import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import argparse

from multiprocessing import Pool

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr

from utils import set_global_seed
from data.graph_dataset import DockingValDataset
from data.collator import collator_3d
from option import set_args
from models.DockingPoseModel import DockingPoseModel

from docking.docking_utils import (
    docking_data_pre,
    ensemble_iterations,
)

from rdkit import Chem

import warnings
warnings.filterwarnings("ignore")


def run_inference(args, device):
    
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    save_path = os.path.join(args.output_path, 'inference.out.pkl')
    set_global_seed(args.seed)
    
    ckpt_path = args.ckpt_path
    state_dict = torch.load(ckpt_path, map_location='cpu') #module.holo_distance_project.out_proj.bias
    new_state_dict = dict()
    for key in state_dict.keys():
        name = key[7:]
        new_state_dict[name] = state_dict[key]

    model = DockingPoseModel(args).to(device)
    model.load_state_dict(new_state_dict)
    

    test_dataset = DockingValDataset(args.input_path, args.complex_list, conf_size=args.conf_size)
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


def get_metrics(input_dir, pred_dir):
    data = pickle.load(open(input_dir, 'rb'))
    holo_pos = data['holo_coordinates'] 
    pred_mol = Chem.MolFromMolFile(os.path.join(pred_dir, 'ligand.sdf'))
    pred_pos = pred_mol.GetConformer().GetPositions()
    rmsd = np.sqrt(np.sum((pred_pos - holo_pos) ** 2) / holo_pos.shape[0])
    
    pred_affi = float(open(os.path.join(pred_dir, 'pred_affinity.txt')).read().splitlines()[0])
    affi = data['affinity']
    return rmsd, pred_affi, affi
    

def evaluate_rmsd(rmsd_results):
    rmsd_results = np.array(rmsd_results)
    print('Predicted RMSD Statistics:')
    print("25% RMSD   : ", np.percentile(rmsd_results, 25))
    print("50% RMSD   : ", np.percentile(rmsd_results, 50))
    print("75% RMSD   : ", np.percentile(rmsd_results, 75))
    print("Mean RMSD  : ", np.mean(rmsd_results))
    print("RMSD < 2.0 : ", np.mean(rmsd_results < 2.0))
    print("RMSD < 5.0 : ", np.mean(rmsd_results < 5.0))


def evaluate_affinity(affi_list, pred_affi_list):
    mae = mean_absolute_error(affi_list, pred_affi_list)
    rmse = np.sqrt(((np.array(pred_affi_list) - np.array(affi_list)) ** 2).mean())
    pccs = pearsonr(affi_list, pred_affi_list)[0]
    spear = spearmanr(affi_list, pred_affi_list)[0]
    print('Predicted Affinity Statistics:')
    print("Affinity MAE      : ", mae)
    print("Affinity RMSE     : ", rmse)
    print("Affinity Pearson  : ", pccs)
    print("Affinity Spearman : ", spear)


if __name__ == '__main__':
    
    parser = set_args()
    parser.add_argument("--input_path", type=str, default="dataset/PDBBind/processed", help="Path of input dataset")
    parser.add_argument("--complex_list", type=str, default="dataset/PDBBind/test.txt", help="Path of file contains the complex IDs that need to be predicted")
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
    
    rmsds, pred_affi, truth_affi = [], [], []
    for pocket_name in new_pocket_list:
        results = get_metrics(os.path.join(args.input_path, f'{pocket_name}.pkl'), os.path.join(args.output_path, pocket_name))
        rmsds.append(results[0])
        if not np.isnan(results[2]): # in case some examples in Mpro dataset did not contain experimentally measured binding affinity
            pred_affi.append(results[1])
            truth_affi.append(results[2])
    
    evaluate_rmsd(rmsds)
    evaluate_affinity(truth_affi, pred_affi)
