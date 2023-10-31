import os
from tqdm import tqdm
import pickle
import argparse
from glob import glob
from multiprocessing import Pool

from data.feature_utils import get_ligand_info, get_protein_info, get_chem_feats, read_mol, get_coords

import warnings
warnings.filterwarnings('ignore')


def process_single(items):
    input_path, output_path = items
    new_data = {}
    name = os.path.basename(input_path)
    mol_file = glob(f'{input_path}/ligand.*')[0]
    pro_file = f'{input_path}/receptor.pdb'
    pocket_file = f'{input_path}/pocket.txt'
    poc_res = open(pocket_file).read().splitlines()

    input_mol = read_mol(mol_file)
    try:
        mol, smiles, coordinate_list = get_coords(input_mol)
    except:
        print(f'generate input ligand coords failed for {name}')
        return False
    
    lig_atoms, lig_atom_feats, lig_edges, lig_bonds = get_ligand_info(mol)
    poc_pos, poc_atoms, poc_atom_feats, poc_edges, poc_bonds = get_protein_info(pro_file, poc_res)
    

    new_data.update({'atoms': lig_atoms, 'coordinates': coordinate_list, 'pocket_atoms': poc_atoms,
                     'pocket_coordinates': poc_pos, 'smi': smiles, 'pocket': name,'lig_feats': lig_atom_feats,
                     'lig_bonds': lig_edges, 'lig_bonds_feats': lig_bonds, 'poc_feats': poc_atom_feats, 
                     'poc_bonds': poc_edges, 'poc_bonds_feats': poc_bonds, 'mol': mol})
    
    new_data = get_chem_feats(new_data)

    f_out = open(output_path, 'wb')
    pickle.dump(new_data, f_out)
    f_out.close()
    return True



def main():
    
    parser = argparse.ArgumentParser(description='preprocess the dataset')
    parser.add_argument('--input_path', type=str, default='example_data', help='input path of raw dataset')
    parser.add_argument('--output_path', type=str, default='example_processed_data', help='output path of processed dataset')
    parser.add_argument('--threads', type=int, default=8, help='number of threads to use')
    args = parser.parse_args()
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    input_data_list = os.listdir(args.input_path)
    output_data_list = [os.path.join(args.output_path, f'{x}.pkl') for x in input_data_list]
    input_data_list = [os.path.join(args.input_path, x) for x in input_data_list]
    data_list = [(x, y) for x,y in zip(input_data_list, output_data_list)]
    
    with Pool(args.threads) as pool:
        for inner_output in tqdm(pool.imap(process_single, data_list), total=len(data_list)):
            if not inner_output:
                print("fail to process")


if __name__ == '__main__':
    main()
