import os
import ipdb
import torch
import numpy as np
import pickle
import copy
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


    return item


def Normalize_data(data):
    dd = data.copy()
    poc_coords_mean = data['pocket_coordinates'].mean(axis=0)
    pocket_coordinates = data['pocket_coordinates'] - poc_coords_mean
    dd['pocket_coordinates'] = pocket_coordinates.astype(np.float32)
    dd['holo_center_coordinates'] = poc_coords_mean.astype(np.float32)

    return dd



class DockingTestDataset(Dataset):
    def __init__(self, dir_path):
    
        self.dir_path = dir_path
        self.data_list = list(glob(os.path.join(dir_path, '*.pkl')))
        self.data = []
        for dataname in self.data_list:
            data_all = pickle.load(open(dataname, 'rb'))
            confs = len(data_all['coordinates'])
            for conf_idx in range(confs):
                data = copy.deepcopy(data_all)
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
                self.data.append((lig_graph, poc_graph, infos))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataset(data_dir):
    val_dataset = DockingTestDataset(data_dir)
    return val_dataset
