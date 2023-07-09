import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
import warnings

warnings.filterwarnings(action="ignore")


def get_all_indexes(lst, element):
    return [i for i in range(len(lst)) if lst[i] == element]

def docking_data_pre(predict_path):

    predict = pd.read_pickle(predict_path)
    (
        smi_list,
        pocket_list,
        pocket_coords_list,
        distance_predict_list,
        holo_distance_predict_list,
        holo_center_coords_list,
        pred_affi_list,
        mol_list
    ) = ([], [], [], [], [], [], [], [])
    
    for batch in predict:
        sz = batch["atoms"].size(0)
        for i in range(sz):
            smi_list.append(batch["smi_name"][i])
            pocket_list.append(batch["pocket_name"][i])

            distance_predict = batch["cross_distance_predict"][i]
            token_mask = batch["atoms"][i] != 0
            pocket_token_mask = batch["pocket_atoms"][i] != 0
            distance_predict = distance_predict[token_mask][:, pocket_token_mask]
            pocket_coords = batch["pocket_coordinates"][i]
            pocket_coords = pocket_coords[pocket_token_mask, :]

            holo_distance_predict = batch["holo_distance_predict"][i]
            holo_distance_predict = holo_distance_predict[token_mask][:, token_mask]

            holo_center_coordinates = batch["holo_center_coordinates"][i][:3]

            pocket_coords_list.append(pocket_coords)
            distance_predict_list.append(distance_predict)
            holo_distance_predict_list.append(holo_distance_predict)
            holo_center_coords_list.append(holo_center_coordinates)

            pred_affi_list.append(batch['pred_affinity'][i])
            mol_list.append(Chem.RemoveHs(batch['mol'][i]))
            

    return (
        mol_list,
        smi_list,
        pocket_list,
        pocket_coords_list,
        distance_predict_list,
        holo_distance_predict_list,
        holo_center_coords_list,
        pred_affi_list
    )


def ensemble_iterations(
    mol_list,
    smi_list,
    pocket_list,
    pocket_coords_list,
    distance_predict_list,
    holo_distance_predict_list,
    holo_center_coords_list,
    pred_affi_list
):
    poc_names = set(pocket_list)
    for name in poc_names:
        idx = get_all_indexes(pocket_list, name)
        pocket_coords_tta = [pocket_coords_list[i] for i in idx]
        distance_predict_tta = [distance_predict_list[i] for i in idx]
        holo_distance_predict_tta = [holo_distance_predict_list[i] for i in idx]
        holo_center_coords_tta = [holo_center_coords_list[i] for i in idx]
        pred_affi_tta = [pred_affi_list[i] for i in idx]

        yield [
            mol_list[idx[0]],
            smi_list[idx[0]],
            pocket_list[idx[0]],
            pocket_coords_tta,
            distance_predict_tta,
            holo_distance_predict_tta,
            holo_center_coords_tta,
            pred_affi_tta
        ]
