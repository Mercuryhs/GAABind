from datetime import datetime
import logging
import os
from tqdm import tqdm


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import set_global_seed, set_logger, Dock_loss
from data.graph_dataset import DockingDataset, DockingValDataset
from data.collator import collator_3d
from option import set_args
from models.DockingPoseModel import DockingPoseModel

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

import warnings
warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')



def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= dist.get_world_size()
    return rt

def run(args, device):
    set_global_seed(args.seed)
    code_root = os.path.dirname(os.path.realpath(__file__))
    timestamp =datetime.now().strftime('%Y%m%d%H%M%S')
    save_folder_name = 'run_' + timestamp
    save_path = os.path.join(code_root, save_folder_name)
    model_path = os.path.join(save_path, 'models')
    code_path = os.path.join(save_path, 'code')
    #print(dist.get_rank())
    if dist.get_rank() == 0:
        os.makedirs(save_path)
        os.makedirs(model_path)
        os.makedirs(code_path)
        os.system(f'cp -r models data docking {code_path} && cp *.py {code_path}')
        set_logger(save_path, name=args.name + f'_{timestamp}')
        logging.info('File saved in ----> {}'.format(save_path))
        logging.info(args)


    model = DockingPoseModel(args).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.99), eps=1e-06)


    train_dataset = DockingDataset(os.path.join(args.data_dir, 'processed'), os.path.join(args.data_dir, 'train.txt'), args.poc_max_len)
    val_dataset = DockingValDataset(os.path.join(args.data_dir, 'processed'), os.path.join(args.data_dir, 'valid.txt'))

    if dist.get_rank() == 0:
        logging.info("number of tersors in model: {}".format(sum([param.nelement() for param in model.parameters()])))
        logging.info(f'train samples: {len(train_dataset)}, val samples: {len(val_dataset)}')
    
    train_sampler = DistributedSampler(train_dataset)
    val_sampler =  DistributedSampler(val_dataset)
   
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler, collate_fn=collator_3d)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=args.num_workers, sampler=val_sampler, collate_fn=collator_3d)
    
    model.train()
    best_valid_loss = 999999
    no_improvement = 0
    best_epoch = 0

    scaler = torch.cuda.amp.GradScaler()
    #mol_token_atoms, mol_edge_type, mol_dist, poc_token_atoms, poc_edge_type, poc_dist, target_mol_dist, target_inter_dist
    for epoch in range(args.epochs):
        train_dataloader.sampler.set_epoch(epoch)
        val_dataloader.sampler.set_epoch(epoch)
        total_cross_loss, total_mol_loss = 0, 0
        total_cross_len, total_mol_len = 0, 0
        total_affi_loss = 0

        for idx, batch_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            for dicts in batch_data[:2]:
                for key in dicts.keys():
                    dicts[key] = dicts[key].to(device)
            
            with torch.cuda.amp.autocast():
                pred = model(batch_data)
                
                cross_loss, mol_loss, distance_mask, holo_distance_mask, affi_loss = Dock_loss(batch_data, pred, args.dist_threshold)
                
                affi_loss_use = affi_loss * (epoch+1) / args.epochs * args.affi_weight
                loss = cross_loss + mol_loss + affi_loss_use
                loss = loss / args.accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (idx+1) % args.accumulation_steps == 0 or (idx+1) == len(train_dataloader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm, norm_type=2)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            #print(cross_loss.item(), mol_loss.item())
            total_cross_loss += (cross_loss * distance_mask.sum()).detach()
            total_mol_loss += (mol_loss * holo_distance_mask.sum()).detach()
            total_cross_len += distance_mask.sum().detach()
            total_mol_len += holo_distance_mask.sum().detach()
            total_affi_loss += affi_loss.detach()

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        total_cross_loss = total_cross_loss / total_cross_len
        total_mol_loss = total_mol_loss / total_mol_len
        total_affi_loss = total_affi_loss / (idx+1)
        total_loss = total_cross_loss + total_mol_loss + total_affi_loss * args.affi_weight
        
        reduced_cross_loss = reduce_tensor(total_cross_loss.data)
        reduced_mol_loss = reduce_tensor(total_mol_loss.data)
        reduced_affi_loss = reduce_tensor(total_affi_loss.data)
        reduced_loss = reduce_tensor(total_loss.data)
        
        
        if dist.get_rank() == 0:
            logging.info(f"epoch {epoch+1:<4d}, train, loss: {reduced_loss:6.3f}, cross_loss: {reduced_cross_loss:6.3f}, mol_loss: {reduced_mol_loss:6.3f}, affinity_loss: {reduced_affi_loss:6.3f}, lr: {lr:10.9f}")
            checkpoints = model.state_dict()
            torch.save(checkpoints, os.path.join(model_path, f'epoch_{epoch+1}.pt'))
        

        val_loss, val_cross_loss, val_mol_loss, val_affi_loss = evaluate(args, model, val_dataloader)
        
        reduced_val_loss = reduce_tensor(val_loss.data)
        reduced_val_cross = reduce_tensor(val_cross_loss.data)
        reduced_val_mol = reduce_tensor(val_mol_loss.data)
        reduced_val_affi = reduce_tensor(val_affi_loss.data)
        
        if reduced_val_loss <= best_valid_loss:
            no_improvement = 0
            best_valid_loss = reduced_val_loss
            best_epoch = epoch
        else:
            no_improvement += 1

        if dist.get_rank() == 0:
            logging.info(f"epoch {epoch+1:<4d}, valid, loss: {reduced_val_loss:6.3f}, cross_loss: {reduced_val_cross:6.3f}, mol_loss: {reduced_val_mol:6.3f}, affinity_loss: {reduced_val_affi:6.3f}, best_epoch: {best_epoch+1}")

def evaluate(args, model, dataloader):
    
    with torch.no_grad():
        model.eval()
        total_cross_loss = 0
        total_mol_loss = 0
        total_cross_len = 0
        total_mol_len = 0
        total_affi_loss = 0
        
        for idx, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
            
            for dicts in batch_data[:2]:
                for key in dicts.keys():
                    dicts[key] = dicts[key].to(device)
    
            with torch.cuda.amp.autocast():
                pred = model(batch_data)
                cross_loss, mol_loss, distance_mask, holo_distance_mask, affi_loss = Dock_loss(batch_data, pred, args.dist_threshold)
            
            total_cross_loss += cross_loss * distance_mask.sum()
            total_mol_loss += mol_loss * holo_distance_mask.sum()
            total_cross_len += distance_mask.sum()
            total_mol_len += holo_distance_mask.sum()
            total_affi_loss += affi_loss

        
        total_cross_loss = total_cross_loss / total_cross_len
        total_mol_loss = total_mol_loss / total_mol_len
        total_affi_loss = total_affi_loss / (idx+1)
        total_loss = total_cross_loss + total_mol_loss + total_affi_loss * args.affi_weight
    
    
    return total_loss, total_cross_loss, total_mol_loss, total_affi_loss


if __name__ == '__main__':
    parser = set_args()
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    run(args, device)
