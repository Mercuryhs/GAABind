import torch
import torch.nn as nn
import torch.nn.functional as F

from .MolTransformer import MolTransformer
from .layers.triangle_attn import PairUpdate, Pair2mol
from .layers.utils import NonLinearHead, DistanceHead


class DockingPoseModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mol_model = MolTransformer(args.mol_atom_type, args.mol_node_feats, args)
        self.poc_model = MolTransformer(args.poc_atom_type, args.poc_node_feats, args)
        
        self.inter_blocks = args.pair_blocks
        self.mlp_z = NonLinearHead(args.node_embed_dim, args.edge_embed_dim)
        self.inter_model = nn.ModuleList([PairUpdate(args.edge_embed_dim,
                                                     args.pair_hidden_dim, 
                                                     args.pair_attn_size, 
                                                     args.pair_attn_heads, 
                                                     args.pair_drop_ratio) for _ in range(self.inter_blocks)])
        
        self.pair2mol = nn.ModuleList([Pair2mol(args.edge_embed_dim, 
                                                args.pair_hidden_dim, 
                                                args.pair_attn_size, 
                                                args.pair_attn_heads, 
                                                args.pair_drop_ratio,
                                                False) for _ in range(self.inter_blocks)])
        

        self.poc_atom_linear = nn.Linear(args.node_embed_dim, args.edge_embed_dim)
        self.mol_atom_linear = nn.Linear(args.node_embed_dim, args.edge_embed_dim)
        self.aff_linear = nn.Linear(args.edge_embed_dim*3, 1)
        self.gate_aff = nn.Linear(args.edge_embed_dim*3, 1)
        self.leaky = nn.LeakyReLU()
        self.bias = nn.Parameter(torch.ones(1))
        
        self.cross_distance_project = NonLinearHead(args.edge_embed_dim, 1)
        self.holo_distance_project = DistanceHead(args.edge_embed_dim)

    def forward(self, batch_data):
        
        _, mol_encoder_rep, mol_encoder_pair_rep, mol_padding_mask = self.mol_model(batch_data[0])
        _, pocket_encoder_rep, pocket_encoder_pair_rep, poc_padding_mask = self.poc_model(batch_data[1])
        
        z = torch.einsum("bik,bjk->bijk", pocket_encoder_rep, mol_encoder_rep) #node embedding; [b,poc,mol,dim]
        mol_true = ~mol_padding_mask
        pocket_true = ~poc_padding_mask
        z_mask = mol_true.unsqueeze(1) * pocket_true.unsqueeze(2)
        mol_mask_2d = mol_true.unsqueeze(1) * mol_true.unsqueeze(2)

        z = self.mlp_z(z)
    
        for i in range(self.inter_blocks):
            z = self.inter_model[i](z, z_mask, pocket_encoder_pair_rep, mol_encoder_pair_rep)
            mol_encoder_pair_rep = self.pair2mol[i](mol_encoder_pair_rep, mol_mask_2d, z, z_mask)
            
        mol_encoder_rep = self.mol_atom_linear(mol_encoder_rep)
        pocket_encoder_rep = self.poc_atom_linear(pocket_encoder_rep)
        mol_sz = mol_encoder_rep.size(1)
        poc_sz = pocket_encoder_rep.size(1)

        aff_z = torch.cat([z, mol_encoder_rep.unsqueeze(-3).repeat(1, poc_sz, 1, 1), pocket_encoder_rep.unsqueeze(-2).repeat(1, 1, mol_sz, 1)], dim=-1)
        aff_z = (self.gate_aff(aff_z).sigmoid() * self.aff_linear(aff_z)).squeeze(-1) * z_mask
        pred_affi = self.leaky(self.bias + aff_z.sum(axis=(-1, -2))/z_mask.sum(axis=(-1, -2)))
        
        mol_encoder_pair_rep = mol_encoder_pair_rep * mol_mask_2d.unsqueeze(-1)
        cross_distance_predict = F.elu(self.cross_distance_project(z).squeeze(-1)) + 1.0
        holo_distance_predict = self.holo_distance_project(mol_encoder_pair_rep)


        return cross_distance_predict, holo_distance_predict, pred_affi