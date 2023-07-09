import torch
from torch_geometric.loader import DataLoader

from datetime import datetime
import logging
import sys
import os
import numpy as np
import random
import torch.nn.functional as F

from torch_geometric.utils import to_dense_batch



def set_global_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def set_logger(save_path, name=None, print_on_screen=True):
    if name!=None:
        log_file = os.path.join(save_path, name+'.log')
    else:
        log_file = os.path.join(save_path, 'train.log')
    print("logger =>" + log_file)
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S ',
        filename=log_file,
        filemode='a'
    )
    if print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
    # fh = logging.FileHandler(log_file, encoding='utf8')
    # formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(lineno)04d | %(message)s')
    # fh.setFormatter(formatter)
    # logging.getLogger('').addHandler(fh)


def adjust_lr(args, now_update, total_update):
    assert args.warmup_ratio > 0
    warmup_updates = int(args.warmup_ratio * total_update)
    max_lr = args.lr
    end_learning_rate = args.end_learning_rate

    if now_update <= warmup_updates:
        warmup_factor = now_update / float(warmup_updates)
        lr = warmup_factor * max_lr
    elif now_update >= total_update:
        lr = end_learning_rate
    else:
        warmup = warmup_updates
        lr_range = max_lr - end_learning_rate
        pct_remaining = 1 - (now_update - warmup) / (total_update - warmup)
        lr = lr_range * pct_remaining ** args.power + end_learning_rate
    
    return lr



def Dock_loss(data, pred, dist_threshold=8.0):
    #mol_token_atoms, mol_edge_type, mol_dist, poc_token_atoms, poc_edge_type, poc_dist, poc_coords, \
    #        poc_center, target_mol_dist, target_inter_dist, smi_name, poc_name, mol_coords
    lig_dict, poc_dict, infos = data[0], data[1], data[2]
    poc_coord, poc_mask = poc_dict['pos'], poc_dict['x'][:,:,0].ne(0)
    holo_coord, holo_mask = lig_dict['holo_pos'], lig_dict['x'][:,:,0].ne(0)
    mask_2d = poc_mask.unsqueeze(1).int() * holo_mask.unsqueeze(2).int()
    distance_target = (poc_coord.unsqueeze(1) - holo_coord.unsqueeze(2)).norm(dim=-1)
    distance_target = distance_target * mask_2d
    
    holo_mask_2d = holo_mask.unsqueeze(1).int() * holo_mask.unsqueeze(2).int()
    holo_distance_target = (holo_coord.unsqueeze(1) - holo_coord.unsqueeze(2)).norm(dim=-1)
    holo_distance_target = holo_distance_target * holo_mask_2d
    
    cross_distance_predict, holo_distance_predict = pred[0], pred[1]
    distance_mask = distance_target != 0
    distance_mask &= (distance_target < dist_threshold)
    cross_distance_predict = cross_distance_predict.permute(0, 2, 1)
    distance_predict = cross_distance_predict[distance_mask]
    distance_target = distance_target[distance_mask]
    distance_target = distance_target.to(distance_predict.device)
  
    distance_loss = F.mse_loss(distance_predict.float(), distance_target.float(), reduction="mean")

    holo_distance_mask = holo_distance_target != 0
    holo_distance_predict_train = holo_distance_predict[holo_distance_mask]
    holo_distance_target = holo_distance_target[holo_distance_mask]
    holo_distance_target = holo_distance_target.to(holo_distance_predict_train.device)
    holo_distance_loss = F.smooth_l1_loss(
            holo_distance_predict_train.float(),
            holo_distance_target.float(),
            reduction="mean",
            beta=1.0,
        )
    
   
    pred_affi = pred[-1]
    affinity = infos['y'].to(pred_affi.device)
    affinity_loss = F.smooth_l1_loss(pred_affi.float(), affinity.float(), reduction="mean", beta=1.0)
    #affinity_loss = affinity_loss * 0.05

    #loss = distance_loss + holo_distance_loss + affinity_loss

    return distance_loss, holo_distance_loss, distance_mask, holo_distance_mask, affinity_loss
