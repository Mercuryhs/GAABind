import copy
import numpy as np
import torch
import pandas as pd
from rdkit import Chem
import pickle
import argparse
import warnings
import os

warnings.filterwarnings(action="ignore")


def single_SF_loss(
    predict_coords,
    pocket_coords,
    distance_predict,
    holo_distance_predict,
    dist_threshold=6.0,
    cross_dist_weight=1.0,
    dist_weight=2.5
):
    dist = torch.norm(predict_coords.unsqueeze(1) - pocket_coords.unsqueeze(0), dim=-1)
    holo_dist = torch.norm(
        predict_coords.unsqueeze(1) - predict_coords.unsqueeze(0), dim=-1
    )
    distance_mask = distance_predict < dist_threshold
    cross_dist_score = (dist[distance_mask] - distance_predict[distance_mask]).abs().mean()

    dist_score = (holo_dist - holo_distance_predict).abs().mean()
    loss = cross_dist_score * cross_dist_weight + dist_score * dist_weight
    return loss


def single_dock_with_gradient(
    pocket_coords,
    distance_predict,
    holo_distance_predict,
    loss_func=single_SF_loss,
    iterations=20000,
    early_stoping=5,
):
    
    coords = torch.randn((holo_distance_predict.shape[0], 3))
    coords.requires_grad = True
    
    # optimizer = torch.optim.LBFGS([coords], lr=1.0)
    optimizer = torch.optim.Adam([coords], lr=0.1)
    bst_loss, times = 10000.0, 0
    #print("=*="*20)
    for i in range(iterations):

        def closure():
            optimizer.zero_grad()
            loss = loss_func(coords, pocket_coords, distance_predict, holo_distance_predict,)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        #print(loss.item(), times, bst_loss, early_stoping, i)
        if loss.item() < bst_loss:
            bst_loss = loss.item()
            times = 0
        else:
            times += 1
            if times > early_stoping:
                break

    return coords.detach().numpy(), loss.detach().numpy()

def set_coord(mol, coords):
    for i in range(coords.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, coords[i].tolist())
    return mol


def add_coord(mol, xyz):
    x, y, z = xyz
    conf = mol.GetConformer(0)
    pos = conf.GetPositions()
    pos[:, 0] += x
    pos[:, 1] += y
    pos[:, 2] += z
    for i in range(pos.shape[0]):
        conf.SetAtomPosition(
            i, Chem.rdGeometry.Point3D(pos[i][0], pos[i][1], pos[i][2])
        )
    return mol


def single_docking(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    content = pd.read_pickle(input_path)
    (
        mol,
        smi,
        pocket,
        pocket_coords_tta,
        distance_predict_tta,
        holo_distance_predict_tta,
        holo_center_coords_tta, 
        pred_affi_list,
    ) = content
            
    sample_times = len(distance_predict_tta)
    bst_predict_coords, bst_loss = None, 1000.0
    for i in range(sample_times):
        pocket_coords = pocket_coords_tta[i]
        distance_predict = distance_predict_tta[i]
        holo_distance_predict = holo_distance_predict_tta[i]
        holo_center_coords = holo_center_coords_tta[i]
        
        predict_coords, loss = single_dock_with_gradient(
                                                pocket_coords,
                                                distance_predict,
                                                holo_distance_predict,
                                                loss_func=single_SF_loss)

        if loss < bst_loss:
            bst_loss = loss
            bst_predict_coords = predict_coords
            pred_affi = pred_affi_list[i]


    mol = Chem.RemoveHs(mol)
    mol = set_coord(mol, bst_predict_coords)
    mol = add_coord(mol, holo_center_coords.numpy())

    output_ligand_path = os.path.join(output_path, 'ligand.sdf')
    Chem.MolToMolFile(mol, output_ligand_path)
    output_affinity_path = open(os.path.join(output_path, 'pred_affinity.txt'), 'w+')
    print(pred_affi.item(), file=output_affinity_path)
    output_affinity_path.close()
    
    return True


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(0)
    parser = argparse.ArgumentParser(description="Docking with gradient")
    parser.add_argument("--input", type=str, help="input file.")
    parser.add_argument(
        "--output-path", type=str, default=None, help="output path."
    )
    args = parser.parse_args()

    single_docking(args.input, args.output_path)
