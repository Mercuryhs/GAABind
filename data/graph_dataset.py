
import os
import pickle
import torch
import numpy as np
from glob import glob

from torch.utils.data import Dataset
from torch_geometric.data import Data


def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index.to(torch.int64), item.x
    N = x.size(0)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = edge_attr + 1
    
    # combine
    item.x = x
    item.attn_edge_type = attn_edge_type
    item.degree = adj.long().sum(dim=1).view(-1) // 2

    return item


def Normalize_data(data):
    dd = data.copy()
    poc_coords_mean = data['pocket_coordinates'].mean(axis=0)
    coordinates = data['holo_coordinates'] - poc_coords_mean
    pocket_coordinates = data['pocket_coordinates'] - poc_coords_mean
    dd['holo_coordinates'] = coordinates.astype(np.float32)
    dd['pocket_coordinates'] = pocket_coordinates.astype(np.float32)
    dd['holo_center_coordinates'] = poc_coords_mean.astype(np.float32)

    return dd

def Normalize_testdata(data):
    dd = data.copy()
    poc_coords_mean = data['pocket_coordinates'].mean(axis=0)
    pocket_coordinates = data['pocket_coordinates'] - poc_coords_mean
    dd['pocket_coordinates'] = pocket_coordinates.astype(np.float32)
    dd['holo_center_coordinates'] = poc_coords_mean.astype(np.float32)

    return dd


def crop_pocket(data, max_len):
    dd = data.copy()
    atoms = dd['pocket_atoms']
    coordinates = dd['pocket_coordinates']
    
    poc_feats = data['poc_feats']
    poc_bonds = data['poc_bonds']
    bond_feats = data['poc_bonds_feats']

    if atoms.shape[0] > max_len:
        distance = np.linalg.norm(coordinates - coordinates.mean(axis=0), axis=1)
        def softmax(x):
            x -= np.max(x)
            x = np.exp(x) / np.sum(np.exp(x))
            return x

        distance += 1  # prevent inf
        weight = softmax(np.reciprocal(distance))
        remain_index = np.random.choice(
            len(atoms), max_len, replace=False, p=weight
        )

        atoms = atoms[remain_index]
        coordinates = coordinates[remain_index]

        poc_feats = poc_feats[remain_index]

        map_dict = dict()
        for i, atom in enumerate(remain_index):
            map_dict[atom] = i
    
        remain_bonds = []
        new_src = []
        new_dst = []
        for i in range(bond_feats.shape[0]):
            bond_src, bond_dst = poc_bonds[0][i], poc_bonds[1][i]
            if bond_src in remain_index and bond_dst in remain_index:
                remain_bonds.append(i)
                new_src.append(map_dict[bond_src])
                new_dst.append(map_dict[bond_dst])
        bond_feats = bond_feats[remain_bonds]
        poc_bonds = np.vstack([np.array(new_src), np.array(new_dst)])
    
    dd['pocket_atoms'] = atoms
    dd['pocket_coordinates'] = coordinates.astype(np.float32)
    
    dd['poc_feats'] = poc_feats
    dd['poc_bonds'] = poc_bonds
    dd['poc_bonds_feats'] = bond_feats

    return dd


def sample_conf(data):
    dd = data.copy()
    conf_size = len(data['coordinates'])
    conf_idx = np.random.randint(conf_size)
    dd['coordinates'] = data['coordinates'][conf_idx]
    return dd


class DockingDataset(Dataset):
    def __init__(self, dir_path, name_list, poc_max_len):
        
        self.dir_path = dir_path
        self.name_list = open(name_list).read().splitlines()
        self.poc_max_len = poc_max_len

    def __len__(self):

        return len(self.name_list)

    def __getitem__(self, idx):
        data_name = self.name_list[idx]
        file_path = open(os.path.join(self.dir_path, f'{data_name}.pkl'), 'rb')
        data = pickle.load(file_path)
        file_path.close()
    
        #random sample conformation
        data = sample_conf(data)
        
        #sample pockets
        data = crop_pocket(data, self.poc_max_len)
        #normalize coords
        data = Normalize_data(data)
       
        lig_chem_feats = torch.from_numpy(data['lig_feats'])
        poc_chem_feats = torch.from_numpy(data['poc_feats'])
        
        #To bi-directional edges
        poc_src = np.hstack([data['poc_bonds'][0], data['poc_bonds'][1]])[np.newaxis, :]
        poc_dst = np.hstack([data['poc_bonds'][1], data['poc_bonds'][0]])[np.newaxis, :]
        poc_edges = np.vstack([poc_src, poc_dst])
        poc_edge_attr = np.vstack([data['poc_bonds_feats'], data['poc_bonds_feats']])
        
        lig_src = np.hstack([data['lig_bonds'][0], data['lig_bonds'][1]])[np.newaxis, :]
        lig_dst = np.hstack([data['lig_bonds'][1], data['lig_bonds'][0]])[np.newaxis, :]
        lig_edges = np.vstack([lig_src, lig_dst])
        lig_edge_attr = np.vstack([data['lig_bonds_feats'], data['lig_bonds_feats']])

        lig_graph = Data()
        poc_graph = Data()
        
        rand_noise = torch.rand((data['coordinates'].shape)) 
        
        lig_graph.__num_nodes__ = lig_chem_feats.shape[0]
        lig_graph.edge_index = torch.from_numpy(lig_edges).to(torch.int64)
        lig_graph.edge_attr = torch.from_numpy(lig_edge_attr).to(torch.int64)
        lig_graph.x = lig_chem_feats.to(torch.int64)
        lig_graph.pos = torch.from_numpy(data['coordinates']) + rand_noise
        lig_graph.holo_pos = torch.from_numpy(data['holo_coordinates'])

        lig_graph = preprocess_item(lig_graph)

        poc_graph.__num_nodes__ = poc_chem_feats.shape[0]
        poc_graph.edge_index = torch.from_numpy(poc_edges).to(torch.int64)
        poc_graph.edge_attr = torch.from_numpy(poc_edge_attr).to(torch.int64)
        poc_graph.x = poc_chem_feats.to(torch.int64)
        poc_graph.pos = torch.from_numpy(data['pocket_coordinates'])

        poc_graph = preprocess_item(poc_graph)

        infos = dict()
        infos['smi'] = data['smi']
        infos['pocket'] = data['pocket']
        infos['holo_center_coordinates'] = torch.from_numpy(data['holo_center_coordinates'])
        infos['y'] = data['affinity']
        infos['mol'] = data['mol']

        return (lig_graph, poc_graph, infos)


class DockingValDataset(Dataset):
    def __init__(self, dir_path, name_list, conf_size=10):
        
        self.dir_path = dir_path
        self.name_list = open(name_list).read().splitlines()
        self.conf_size = conf_size
        

    def __len__(self):
        
        return len(self.name_list) * self.conf_size

    def __getitem__(self, idx):
        data_idx = idx // self.conf_size
        conf_idx = idx % self.conf_size
        data_name = self.name_list[data_idx]
        file_path = open(os.path.join(self.dir_path, f'{data_name}.pkl'), 'rb')
        data = pickle.load(file_path)
        file_path.close()
        data['coordinates'] = data['coordinates'][conf_idx]

        data = Normalize_data(data)

        lig_chem_feats = torch.from_numpy(data['lig_feats'])
        poc_chem_feats = torch.from_numpy(data['poc_feats'])

        poc_src = np.hstack([data['poc_bonds'][0], data['poc_bonds'][1]])[np.newaxis, :]
        poc_dst = np.hstack([data['poc_bonds'][1], data['poc_bonds'][0]])[np.newaxis, :]
        poc_edges = np.vstack([poc_src, poc_dst])
        poc_edge_attr = np.vstack([data['poc_bonds_feats'], data['poc_bonds_feats']])
        
        lig_src = np.hstack([data['lig_bonds'][0], data['lig_bonds'][1]])[np.newaxis, :]
        lig_dst = np.hstack([data['lig_bonds'][1], data['lig_bonds'][0]])[np.newaxis, :]
        lig_edges = np.vstack([lig_src, lig_dst])
        lig_edge_attr = np.vstack([data['lig_bonds_feats'], data['lig_bonds_feats']])

        lig_graph = Data()
        poc_graph = Data()
        
        lig_graph.__num_nodes__ = lig_chem_feats.shape[0]
        lig_graph.edge_index = torch.from_numpy(lig_edges).to(torch.int64)
        lig_graph.edge_attr = torch.from_numpy(lig_edge_attr).to(torch.int64)
        lig_graph.x = lig_chem_feats.to(torch.int64)
        lig_graph.pos = torch.from_numpy(data['coordinates'])
        lig_graph.holo_pos = torch.from_numpy(data['holo_coordinates'])

        lig_graph = preprocess_item(lig_graph)

        poc_graph.__num_nodes__ = poc_chem_feats.shape[0]
        poc_graph.edge_index = torch.from_numpy(poc_edges).to(torch.int64)
        poc_graph.edge_attr = torch.from_numpy(poc_edge_attr).to(torch.int64)
        poc_graph.x = poc_chem_feats.to(torch.int64)
        poc_graph.pos = torch.from_numpy(data['pocket_coordinates'])

        poc_graph = preprocess_item(poc_graph)

        infos = dict()
        infos['smi'] = data['smi']
        infos['pocket'] = data['pocket']
        infos['holo_center_coordinates'] = torch.from_numpy(data['holo_center_coordinates'])
        infos['y'] = data['affinity']
        infos['mol'] = data['mol']

        return lig_graph, poc_graph, infos
    
    
class DockingTestDataset(Dataset):
    def __init__(self, dir_path, conf_size=10):
        
        self.dir_path = dir_path
        self.name_list = glob(f'{self.dir_path}/*.pkl')
        self.conf_size = conf_size
        
    def __len__(self):
    
        return len(self.name_list) * self.conf_size

    def __getitem__(self, idx):
        data_idx = idx // self.conf_size
        conf_idx = idx % self.conf_size
        file_path = open(self.name_list[data_idx], 'rb')
        data = pickle.load(file_path)
        file_path.close()
        data['coordinates'] = data['coordinates'][conf_idx]

        data = Normalize_testdata(data)

        lig_chem_feats = torch.from_numpy(data['lig_feats'])
        poc_chem_feats = torch.from_numpy(data['poc_feats'])

        poc_src = np.hstack([data['poc_bonds'][0], data['poc_bonds'][1]])[np.newaxis, :]
        poc_dst = np.hstack([data['poc_bonds'][1], data['poc_bonds'][0]])[np.newaxis, :]
        poc_edges = np.vstack([poc_src, poc_dst])
        poc_edge_attr = np.vstack([data['poc_bonds_feats'], data['poc_bonds_feats']])
        
        lig_src = np.hstack([data['lig_bonds'][0], data['lig_bonds'][1]])[np.newaxis, :]
        lig_dst = np.hstack([data['lig_bonds'][1], data['lig_bonds'][0]])[np.newaxis, :]
        lig_edges = np.vstack([lig_src, lig_dst])
        lig_edge_attr = np.vstack([data['lig_bonds_feats'], data['lig_bonds_feats']])

        lig_graph = Data()
        poc_graph = Data()
        
        lig_graph.__num_nodes__ = lig_chem_feats.shape[0]
        lig_graph.edge_index = torch.from_numpy(lig_edges).to(torch.int64)
        lig_graph.edge_attr = torch.from_numpy(lig_edge_attr).to(torch.int64)
        lig_graph.x = lig_chem_feats.to(torch.int64)
        lig_graph.pos = torch.from_numpy(data['coordinates'])

        lig_graph = preprocess_item(lig_graph)

        poc_graph.__num_nodes__ = poc_chem_feats.shape[0]
        poc_graph.edge_index = torch.from_numpy(poc_edges).to(torch.int64)
        poc_graph.edge_attr = torch.from_numpy(poc_edge_attr).to(torch.int64)
        poc_graph.x = poc_chem_feats.to(torch.int64)
        poc_graph.pos = torch.from_numpy(data['pocket_coordinates'])

        poc_graph = preprocess_item(poc_graph)

        infos = dict()
        infos['smi'] = data['smi']
        infos['pocket'] = data['pocket']
        infos['holo_center_coordinates'] = torch.from_numpy(data['holo_center_coordinates'])
        infos['mol'] = data['mol']

        return lig_graph, poc_graph, infos
