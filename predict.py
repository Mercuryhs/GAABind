import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import pickle

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch.utils.data import DataLoader


from utils import set_global_seed, Dock_loss
from data.graph_dataset import get_dataset
from data.collator import collator_3d
from option import args
from models.DockingPoseModel import DockingPoseModel


import warnings
warnings.filterwarnings("ignore")


def run(args, device):
    code_root = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(code_root, args.save_path)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(code_root, args.save_path, 'test.out.pkl')
    set_global_seed(args.seed)
    
    ckpt_path = args.ckpt_path
    state_dict = torch.load(ckpt_path, map_location='cpu')
    new_state_dict = dict()
    for key in state_dict.keys():
        name = key[7:]
        new_state_dict[name] = state_dict[key]

    model = DockingPoseModel(args).to(device)
    model.load_state_dict(new_state_dict)
    

    test_dataset = get_dataset(args.data_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=collator_3d)

    outputs = []
    with torch.no_grad():
        model.eval()
        for batch in tqdm(test_dataloader):
            for dicts in batch[:2]:
                for key in dicts.keys():
                    dicts[key] = dicts[key].to(device)

            with torch.cuda.amp.autocast():
                pred = model(batch)

            mol_token_atoms = batch[0]['x'][:,:,0]
            poc_token_atoms = batch[1]['x'][:,:,0]
            poc_coords = batch[1]['pos']
            sample_size = mol_token_atoms.size(0)
            logging_output = {
                "bsz": sample_size,
                "sample_size": 1,
            }
            
            logging_output["smi_name"] = batch[2]['smi_list']
            logging_output["pocket_name"] = batch[2]['pocket_list']
            logging_output['mol'] = batch[2]['mol']
            logging_output["cross_distance_predict"] = pred[0].data.detach().cpu().permute(0, 2, 1)
            logging_output["holo_distance_predict"] = pred[1].data.detach().cpu()
            logging_output["atoms"] = mol_token_atoms.data.detach().cpu()
            logging_output["pocket_atoms"] = poc_token_atoms.data.detach().cpu()
            logging_output["holo_center_coordinates"] = batch[2]['holo_center_list']
            logging_output["pocket_coordinates"] = poc_coords.data.detach().cpu()
            logging_output['pred_affinity'] = pred[-1].data.detach().cpu()
            outputs.append(logging_output)
    
        pickle.dump(outputs, open(save_path, "wb"))
    


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = f'cuda:{str(args.gpu)}' if torch.cuda.is_available() else 'cpu' 
    run(args, device)
    
