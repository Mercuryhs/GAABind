import torch
import torch.nn as nn
import math
from .layers.transfomer_encoder import TransformerEncoder
from .layers.utils import NonLinearHead, GaussianLayer


def init_params(module, n_layers=None):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class AtomFeature(nn.Module):
    """
    Compute atom features for each atom in the molecule.
    """

    def __init__(self, num_atoms, hidden_dim):
        super(AtomFeature, self).__init__()
        self.num_atoms = num_atoms

        # 1 for graph token
        self.atom_encoder = nn.Embedding(num_atoms, hidden_dim, padding_idx=0)
        # self.degree_encoder = nn.Embedding(num_degree, hidden_dim, padding_idx=0)

        self.graph_token = nn.Embedding(1, hidden_dim)

        self.apply(lambda module: init_params(module))

    def forward(self, batched_data):
        """
        input: 
        output: [n_graph, n_node+1, n_feat]
        """
        
        x = batched_data['x']
        n_graph = x.size()[0]

        # node feauture + graph token
        node_feature = self.atom_encoder(x).sum(dim=-2) # [n_graph, n_node, n_feat] - >[n_graph, n_node, n_feat * n_dim]
        # degree_feature = self.degree_encoder(degree)
        # node_feature = node_feature + degree_feature

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        return graph_node_feature


class EdgeFeature(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(self, num_edges, hidden_dim):
        super(EdgeFeature, self).__init__()
        self.hidden_dim = hidden_dim
        self.edge_encoder = nn.Embedding(num_edges, hidden_dim, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(1, hidden_dim)
        self.apply(lambda module: init_params(module))
        
    def forward(self, batched_data):
        """
        input: 
        output: [n_graph, n_feat, n_edge+1, n_edge+1]
        """
        edge_attr = batched_data['attn_edge_type']  # [bs, num_node, num_node, feat]

        bs, n_node = edge_attr.size()[:2]
        graph_attn_bias = torch.zeros([bs, n_node+1, n_node+1], device=edge_attr.device)
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.hidden_dim, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]

        edge_attr = self.edge_encoder(edge_attr).mean(-2).permute(0, 3, 1, 2).contiguous() #b * N * N * D * H
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:,:, 1:, 1:] + edge_attr

        t = self.graph_token_virtual_distance.weight.view(1, self.hidden_dim, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t
        
        return graph_attn_bias

class Edge3DFeature(nn.Module):
    """
        Compute 3D attention bias according to the position information for each head.
        """

    def __init__(self, hidden_dim, num_edge_types, node_embed_dim, num_kernel):
        super(Edge3DFeature, self).__init__()

        self.gbf = GaussianLayer(num_kernel, num_edge_types)
        self.gbf_proj = NonLinearHead(num_kernel, hidden_dim)
        # self.edge_proj = nn.Linear(num_kernel, node_embed_dim)
    

    def forward(self, batched_data):

        pos, x, node_type_edge = batched_data['pos'], batched_data['x'], batched_data['node_type_edge'] 

        padding_mask = x.eq(0).all(dim=-1)
        _, n_node, _ = pos.shape
        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1).view(-1, n_node, n_node)
    
        edge_feature = self.gbf(dist, node_type_edge.long())
        edge_feature = edge_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0
        )

        # sum_edge_features = edge_feature.sum(dim=-2)
        # merge_edge_features = self.edge_proj(sum_edge_features)
        
        
        graph_attn_bias = self.gbf_proj(edge_feature)
        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()

        

        return graph_attn_bias


class MolTransformer(nn.Module):
    def __init__(self, atom_type, node_feats, args, K=128):
        super().__init__()
    
        self.args = args
        self.atom_type = atom_type
        self.node_feats = node_feats
        n_edge_type = self.atom_type * self.atom_type
        
        self.atom_feature = AtomFeature(self.node_feats,
                                        args.node_embed_dim)

        self.edge_feature = EdgeFeature(args.edge_feats,
                                        args.edge_embed_dim)
        
        self.edge_3d_feature = Edge3DFeature(args.edge_embed_dim, 
                                             n_edge_type, 
                                             args.node_embed_dim,
                                             num_kernel=K)
        
        self.encoder = TransformerEncoder(args.encoder_layers,
                                          args.node_emb_dropout,
                                          args.node_embed_dim,
                                          args.node_ffn_dim,
                                          args.node_attn_heads,
                                          args.node_ffn_dropout,
                                          args.node_attn_dropout,
                                          args.edge_embed_dim, 
                                          args.edge_attn_heads,
                                          args.edge_attn_size, 
                                          args.edge_dropout)

    def forward(self, batch_data):
        
        data_x = batch_data["x"]   # Node feature, [bs, num_nodes, num_node_features]
        n_mol = data_x.size()[0]
        padding_mask_atom = (data_x[:,:,0]).eq(0) # B x T 
        padding_mask_cls = torch.zeros(n_mol, 1, device=padding_mask_atom.device, dtype=padding_mask_atom.dtype)
        padding_mask = torch.cat((padding_mask_cls, padding_mask_atom), dim=1) # B x (T+1) 
        node_embed = self.atom_feature(batch_data)

        edge_embed = self.edge_feature(batch_data)
        edge_3d_embed = self.edge_3d_feature(batch_data)
        edge_embed[:, :, 1:, 1:] = edge_embed[:, :, 1:, 1:] + edge_3d_embed
        edge_embed = edge_embed.permute(0, 2, 3, 1).contiguous() #B*N*N*H

        node_embed, edge_embed = self.encoder(node_embed, padding_mask=padding_mask, edge_repr=edge_embed)
        edge_embed[edge_embed == float("-inf")] = 0
        encoder_cls = node_embed[:,0,:]
        return encoder_cls, node_embed[:, 1:, :], edge_embed[:, 1:, 1:, :], padding_mask_atom
    
    
    
