import numpy as np
import copy
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans
from rdkit.Chem.rdMolAlign import AlignMolConformers

def read_mol(molfile):
    Chem.WrapLogs()
    if molfile.endswith('.sdf') or molfile.endswith('.mol'):
        mol = Chem.MolFromMolFile(molfile)
        mol = Chem.AddHs(mol)
    elif molfile.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molfile)
        mol = Chem.AddHs(mol)
    elif molfile.endswith('.txt'):
        smiles = open(molfile).read().splitlines()[0].strip()
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
    else:
        print('Unrecognized file format.')

    return mol


def single_conf_gen(tgt_mol, num_confs=1000, seed=42, removeHs=True):
    mol = copy.deepcopy(tgt_mol)
    mol = Chem.AddHs(mol)
    allconformers = AllChem.EmbedMultipleConfs(
        mol, numConfs=num_confs, randomSeed=seed, clearConfs=True
    )
    sz = len(allconformers)
    for i in range(sz):
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=i)
        except:
            continue
    if removeHs:
        mol = Chem.RemoveHs(mol)
    return mol


def clustering_coords(mol, M=100, N=10, seed=42, removeHs=True):
    rdkit_coords_list = []
    rdkit_mol = single_conf_gen(mol, num_confs=M, seed=seed, removeHs=removeHs)
    noHsIds = [
        rdkit_mol.GetAtoms()[i].GetIdx()
        for i in range(len(rdkit_mol.GetAtoms()))
        if rdkit_mol.GetAtoms()[i].GetAtomicNum() != 1
    ]
    ### exclude hydrogens for aligning
    AlignMolConformers(rdkit_mol, atomIds=noHsIds)
    sz = len(rdkit_mol.GetConformers())
    for i in range(sz):
        _coords = rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
        rdkit_coords_list.append(_coords)

    ### exclude hydrogens for clustering
    rdkit_coords_flatten = np.array(rdkit_coords_list)[:, noHsIds].reshape(sz, -1)
    ids = (
        KMeans(n_clusters=N, random_state=seed)
        .fit_predict(rdkit_coords_flatten)
        .tolist()
    )
    N_cluster = len(set(ids))
    coords_list = [rdkit_coords_list[ids.index(i)] for i in range(N_cluster)]
    return coords_list


def get_coords(mol, seed=42):

    M, N = 100, 10
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    lig_atoms = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
    
    coordinate_list = clustering_coords(mol, M=M, N=N, seed=seed, removeHs=False)

    smiles = Chem.MolToSmiles(mol)
    mol_mask_hydrogen = lig_atoms != 'H'
    new_coordinates = [coords[mol_mask_hydrogen] for coords in coordinate_list]
    
    return mol, smiles, new_coordinates

def get_all_atom_feature(mol, protein=True):
    feature_list = []
    res_list = []
    atoms = []
   
    for atom_i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_i)
        feats = [atom.GetSymbol(), atom.GetChiralTag(), atom.GetTotalDegree(), atom.GetFormalCharge(), 
                 atom.GetTotalNumHs(), atom.GetNumRadicalElectrons(), atom.GetHybridization(), 
                 atom.GetIsAromatic(), atom.IsInRing()]
        
        if protein:
            res_info = atom.GetPDBResidueInfo()
            res_name = res_info.GetResidueName()
            res_id = res_info.GetChainId() + '_' + str(res_info.GetResidueNumber()) + '_' + res_name
            res_list.append(res_id)
            feats.append(res_name)
    
        feature_list.append(feats)
        atoms.append(atom.GetSymbol())
        
    if protein:
        return np.array(feature_list),  np.array(atoms), np.array(res_list)
    else:
        return np.array(feature_list), np.array(atoms)


def get_all_bond_feature(mol):
    feature_list = []
    bonded_num = mol.GetNumBonds()
    for i in range(bonded_num):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        feats = [(u,v), str(bond.GetBondType()).split('.')[-1], str(bond.GetBondDir()).split('.')[-1], \
                 str(bond.GetStereo()).split('.')[-1], bond.GetIsConjugated()]
        feature_list.append(feats)
        
    return np.array(feature_list)



def get_protein_info(pdbfile, poc_res):
  
    protein = Chem.MolFromPDBFile(pdbfile)
    if protein is None:
        protein = Chem.MolFromPDBFile(pdbfile, sanitize=False)
        protein = Chem.RemoveHs(protein, sanitize=False)
    atom_feats, atoms, res_list= get_all_atom_feature(protein)
    bond_feats = get_all_bond_feature(protein)
    
    remain_atoms = []
    for i, res_id in enumerate(res_list):
        atom = atoms[i]
        if res_id in poc_res and atom != 'H':
            remain_atoms.append(i)
    atom_feats = atom_feats[remain_atoms]
    atoms = atoms[remain_atoms]
    res_list = res_list[remain_atoms]
    
    remain_bonds = []
    for i, feats in enumerate(bond_feats):
        bond_src, bond_dst = feats[0]
        if bond_src in remain_atoms and bond_dst in remain_atoms:
            remain_bonds.append(i)
    bond_feats = bond_feats[remain_bonds]
    
    all_pos = protein.GetConformers()[0].GetPositions()
    pos = all_pos[remain_atoms]
    
    bond_src = []
    bond_dst = []
    new_bond_feats = []
    map_dict = dict()
    for i, atom in enumerate(remain_atoms):
        map_dict[atom] = i
    for feat in bond_feats:
        src, dst = feat[0]
        new_src, new_dst = map_dict[src], map_dict[dst]
        bond_src.append(new_src)
        bond_dst.append(new_dst)
        new_bond_feats.append(feat[1:])
    edges = np.vstack([np.array(bond_src), np.array(bond_dst)])
    
    return pos, atoms, atom_feats, edges, np.array(new_bond_feats)


def get_ligand_info(mol):
    atom_feats, atoms = get_all_atom_feature(mol, protein=False)
    bond_feats = get_all_bond_feature(mol)
    remain_atoms = [i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetSymbol() != 'H']
    atoms = atoms[remain_atoms]
    atom_feats = atom_feats[remain_atoms]
    
    
    remain_bonds = []
    for i, feats in enumerate(bond_feats):
        bond_src, bond_dst = feats[0]
        if bond_src in remain_atoms and bond_dst in remain_atoms:
            remain_bonds.append(i)
    bond_feats = bond_feats[remain_bonds]
    
    bond_src = []
    bond_dst = []
    new_bond_feats = []
    map_dict = dict()
    for i, atom in enumerate(remain_atoms):
        map_dict[atom] = i
    for feat in bond_feats:
        src, dst = feat[0]
        new_src, new_dst = map_dict[src], map_dict[dst]
        bond_src.append(new_src)
        bond_dst.append(new_dst)
        new_bond_feats.append(feat[1:])
    edges = np.vstack([np.array(bond_src), np.array(bond_dst)])
    
    return atoms, atom_feats, edges, np.array(new_bond_feats)




chira_vocab = ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CCW', 'CHI_TETRAHEDRAL_CW']
degree_vocab = ['0', '1', '2', '3', '4', 'unk']
hybridization_vocab = ['SP3', 'SP3D2', 'SP', 'SP3D', 'SP2', 'UNSPECIFIED']
num_hs_vocab = ['0', '1', '2', '3', '4']

bond_type_vocab = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']


AA_list = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", 
           "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "HOH", "UNK"]

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def get_atom_fea(feats, ligand=True):
    #offset: 
    if ligand:
        offset = [11, 14, 20, 25, 31, 33, 35] 
    else:
        offset = [5, 8, 14, 19, 25, 27, 29]  
    
    chira = np.array([chira_vocab.index(x) for x in feats[:,1]])[:,np.newaxis] + offset[0]
    degree = np.array([safe_index(degree_vocab, x) for x in feats[:,2]])[:,np.newaxis]  + offset[1]
    
    num_hs =  np.array([num_hs_vocab.index(x) for x in feats[:,4]])[:,np.newaxis]  + offset[2]
    hybridization =  np.array([safe_index(hybridization_vocab, x) for x in feats[:,6]])[:,np.newaxis]  + offset[3]
    
    isaromatic = np.array([1 if x=='True' else 0 for x in feats[:,7]])[:,np.newaxis]  + offset[4]
    isinring = np.array([1 if x=='True' else 0 for x in feats[:,8]])[:,np.newaxis]  + offset[5]
    
    new_feats = np.hstack([chira, degree, num_hs, hybridization, isaromatic, isinring]) 
    
    return new_feats

def get_bond_fea(feats):
    bond_type = np.array([bond_type_vocab.index(x) for x in feats[:,0]])[:, np.newaxis]
    conjugated = np.array([int(bool(x)) for x in feats[:,3]])[:,np.newaxis] + 4
    
    new_feats = np.hstack([bond_type, conjugated])
    return new_feats



def mol_vec_index(atom):
    atom_list = ['UNK', 'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'I']
    if atom not in atom_list:
        return atom_list.index('UNK')
    return atom_list.index(atom)

def poc_vec_index(atom):
    atom_list = ['UNK', 'C', 'N', 'O', 'S']
    if atom not in atom_list:
        return atom_list.index('UNK')
    return atom_list.index(atom)

def token_atoms(mol_atoms, poc_atoms):
    mol_token_atoms = np.vectorize(mol_vec_index)(mol_atoms)
    poc_token_atoms = np.vectorize(poc_vec_index)(poc_atoms)
    return mol_token_atoms, poc_token_atoms


def get_res_fea(feats):
    residues = feats[:,-1]
    res_fea = np.array([safe_index(AA_list, res) for res in residues])[:, np.newaxis] + 29
    return res_fea

def get_chem_feats(data):
    lig_feats, poc_feats = data['lig_feats'], data['poc_feats']
    lig_bonds, poc_bonds = data['lig_bonds_feats'], data['poc_bonds_feats']
    
    mol_token_atoms, poc_token_atoms = token_atoms(data['atoms'], data['pocket_atoms'])
    
    
    lig_feats = get_atom_fea(lig_feats)
    poc_feats = get_atom_fea(poc_feats, ligand=False)
    poc_res = get_res_fea(data['poc_feats'])
    lig_bonds = get_bond_fea(lig_bonds)
    poc_bonds = get_bond_fea(poc_bonds)
    
    lig_all_feats = np.hstack([mol_token_atoms[:,np.newaxis], lig_feats])
    poc_all_feats = np.hstack([poc_token_atoms[:,np.newaxis], poc_feats, poc_res])
    
    data['lig_feats'] = lig_all_feats
    data['poc_feats'] = poc_all_feats
    data['lig_bonds_feats'] = lig_bonds
    data['poc_bonds_feats'] = poc_bonds
    
    return data