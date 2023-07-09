import os
from tqdm import tqdm
import pickle
import argparse
from glob import glob
import warnings
from multiprocessing import Pool

from feature_utils import get_ligand_info, get_protein_info, get_chem_feats, read_mol, get_coords


warnings.filterwarnings('ignore')



def process_single(data_dir):
    name = os.path.basename(data_dir)
    save_dir = os.path.dirname(data_dir)
    
    save_path = f'{save_dir}/{name}.pkl'
    if os.path.exists(save_path): 
        return True
    new_data = {}
    mol_file = glob(f'{data_dir}/ligand.*')[0]
    pro_file = f'{data_dir}/receptor.pdb'
    pocket_file = f'{data_dir}/pocket.txt'
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
    
    
    f_out = open(save_path, 'wb')
    pickle.dump(new_data, f_out)
    f_out.close()
    return True



def main():
    
    parser = argparse.ArgumentParser(description='preprocess the dataset')
    parser.add_argument('--data_path', type=str, default='example_data', help='path of dataset')
    parser.add_argument('--threads', type=int, default=8, help='number of threads to use')
    args = parser.parse_args()

    data_list = list(glob(os.path.join(args.data_path, '*')))
    with Pool(20) as pool:
        for inner_output in tqdm(pool.imap(process_single, data_list), total=len(data_list)):
            if not inner_output:
                print("fail to process")


if __name__ == '__main__':
    main()