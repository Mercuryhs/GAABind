import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import pickle

from multiprocessing import Pool

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader


from utils import set_global_seed
from data.graph_dataset import DockingTestDataset
from data.collator import collator_test_3d
from option import set_args
from models.DockingPoseModel import DockingPoseModel

from docking.docking_utils import (
    docking_data_pre,
    ensemble_iterations,
)

import warnings
warnings.filterwarnings("ignore")


def run_inference(args, device):
    
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    save_path = os.path.join(args.output_path, 'inference.out.pkl')

    set_global_seed(args.seed)
    
    ckpt_path = args.ckpt_path
    state_dict = torch.load(ckpt_path, map_location='cpu')
    new_state_dict = dict()
    for key in state_dict.keys():
        name = key[7:]
        new_state_dict[name] = state_dict[key]

    model = DockingPoseModel(args).to(device)
    model.load_state_dict(new_state_dict)

    test_dataset = DockingTestDataset(args.input_path, args.conf_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=collator_test_3d)

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

            logging_output = {}

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

    parser = set_args()
    parser.add_argument("--input_path", type=str, default="example_processed_data", help="Path of processed dataset")
    parser.add_argument("--ckpt_path", type=str, default="saved_model/best_epoch.pt", help="Model path")
    parser.add_argument("--nthreads", type=int, default=8, help="Number of threads")
    parser.add_argument("--output_path", type=str, default="predict_result", help="Location of the prediction file")
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = f'cuda:{str(args.gpu)}' if torch.cuda.is_available() else 'cpu' 
    run_inference(args, device)
    
    predict_path = os.path.join(args.output_path, 'inference.out.pkl')
    
    mol_list, smi_list, pocket_list, pocket_coords_list, distance_predict_list, holo_distance_predict_list,\
        holo_center_coords_list, pred_affi_list = docking_data_pre(predict_path)


    iterations = ensemble_iterations(mol_list, smi_list, pocket_list, pocket_coords_list, distance_predict_list,\
                                     holo_distance_predict_list, holo_center_coords_list, pred_affi_list)
    
    new_pocket_list = set(pocket_list)
    output_dir = os.path.join(args.output_path, "cache")
    os.makedirs(output_dir, exist_ok=True)

    def dump_file(content):
        pocket = content[2]
        output_name = os.path.join(output_dir, "{}.pkl".format(pocket))
        try:
            os.remove(output_name)
        except:
            pass
        pd.to_pickle(content, output_name)
        return True

    with Pool(args.nthreads) as pool:
        for inner_output in pool.imap(dump_file, iterations):
            if not inner_output:
                print("fail to dump")
    pool.close()

    def single_docking(pocket_name):
        input_name = os.path.join(output_dir, "{}.pkl".format(pocket_name))
        output_ligand_name = os.path.join(args.output_path, pocket_name)
        try:
            os.remove(output_ligand_name)
        except:
            pass
        cmd = "python docking/coordinate_model.py --input {}  --output-path {}".format(
               input_name, output_ligand_name)
        os.system(cmd)
        return True
    

    with Pool(args.nthreads) as pool:
        for inner_output in tqdm(
            pool.imap(single_docking, new_pocket_list), total=len(new_pocket_list)
        ):
            if not inner_output:
                print("fail to docking")
                
    pool.close()
    
