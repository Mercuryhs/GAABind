import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import glob
import argparse
from docking_utils import (
    docking_data_pre,
    ensemble_iterations,

)
import warnings

warnings.filterwarnings(action="ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="docking")
    parser.add_argument(
        "--predict_path",
        type=str,
        default="./predict_result/test.out.pkl",
        help="Location of the prediction file",
    )
    parser.add_argument("--nthreads", type=int, default=8, help="num of threads")
    
    args = parser.parse_args()
    output_path = os.path.dirname(args.predict_path)

    (
        mol_list,
        smi_list,
        pocket_list,
        pocket_coords_list,
        distance_predict_list,
        holo_distance_predict_list,
        holo_center_coords_list,
        pred_affi_list
    ) = docking_data_pre(args.predict_path)
    
    
   
    
    iterations = ensemble_iterations(
        mol_list,
        smi_list,
        pocket_list,
        pocket_coords_list,
        distance_predict_list,
        holo_distance_predict_list,
        holo_center_coords_list,
        pred_affi_list
        
    )
    
    new_pocket_list = set(pocket_list)
    output_dir = os.path.join(output_path, "cache")
    os.makedirs(output_dir, exist_ok=True)

    def dump(content):
        pocket = content[2]
        output_name = os.path.join(output_dir, "{}.pkl".format(pocket))
        try:
            os.remove(output_name)
        except:
            pass
        pd.to_pickle(content, output_name)
        return True

    with Pool(args.nthreads) as pool:
        for inner_output in tqdm(pool.imap(dump, iterations), total=len(new_pocket_list)):
            if not inner_output:
                print("fail to dump")
                
    pool.close()

    def single_docking(pocket_name):
        input_name = os.path.join(output_dir, "{}.pkl".format(pocket_name))
        output_ligand_name = os.path.join(output_path, "{}.ligand.sdf".format(pocket_name))
        try:
            os.remove(output_ligand_name)
        except:
            pass
        cmd = "python docking/coordinate_model.py --input {}  --output-ligand {}".format(
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
