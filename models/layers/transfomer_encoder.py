from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


from .multihead_attn import SelfMultiheadAttention
from .triangle_attn import MolTriangleAttention, MolTriangleUpdate, Transition
from .utils import NonLinearHead

class NodeUpdate(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        post_ln = False,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout

        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = nn.ReLU(inplace=True)

        self.self_attn = SelfMultiheadAttention(
            self.embed_dim,
            attention_heads,
            dropout=attention_dropout,
        )
        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.post_ln = post_ln


    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool=False,
    ) -> torch.Tensor:
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        if not self.post_ln:
            x = self.self_attn_layer_norm(x)
        # new added
        x = self.self_attn(
            query=x,
            key_padding_mask=padding_mask,
            attn_bias=attn_bias,
            return_attn=return_attn,
        )
        if return_attn:
            x, attn_weights, attn_probs = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.self_attn_layer_norm(x)

        residual = x
        if not self.post_ln:
            x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.final_layer_norm(x)
        if not return_attn:
            return x
        else:
            return x, attn_weights, attn_probs
                


class Node2Edge(nn.Module):
    def __init__(self, num_heads, edge_dim) -> None:
        super().__init__()
        self.attn_proj = NonLinearHead(num_heads, edge_dim)
        
    def forward(self, attn_bias):
        attn_bias = self.attn_proj(attn_bias)

        return attn_bias


class EdgeUpdate(nn.Module):
    def __init__(self, edge_dim, attn_size, num_heads, edge_dropout=0.1) -> None:
        super().__init__()
        self.tri_update = MolTriangleUpdate(edge_dim, edge_dim)
        self.tri_attn = MolTriangleAttention(edge_dim, attn_size, num_heads)
        self.transition = Transition(edge_dim, n=2)
        self.dropout = nn.Dropout2d(p=edge_dropout)
        
    def forward(self, edge_repr, edge_mask):
        edge_repr = edge_repr + self.dropout(self.tri_update(edge_repr, edge_mask))
        edge_repr = edge_repr + self.dropout(self.tri_attn(edge_repr, edge_mask))
        edge_repr = self.dropout(self.transition(edge_repr))
        return edge_repr
        

def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
    bsz = padding_mask.size(0)
    seq_len = padding_mask.size(1)
    if attn_mask is not None and padding_mask is not None:
        # merge key_padding_mask and attn_mask
        attn_mask = attn_mask.view(bsz, -1, seq_len, seq_len)
        attn_mask.masked_fill_(
            padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
            fill_val,
        )
        attn_mask = attn_mask.view(-1, seq_len, seq_len)
    return attn_mask


class TransformerBlock(nn.Module):
    def __init__(self, node_embed_dim, node_ffn_dim, node_attn_heads, node_ffn_dropout, 
                 node_attn_dropout, edge_embed_dim, edge_attn_heads,edge_attn_size, edge_dropout):
        super().__init__()
        self.proj_edge_bias = nn.Linear(edge_embed_dim, node_attn_heads)
        self.node_attn = NodeUpdate(node_embed_dim, node_ffn_dim, node_attn_heads,
                                    node_ffn_dropout, node_attn_dropout)
        self.node2edge = Node2Edge(node_attn_heads, edge_embed_dim)
        self.edge_attn = EdgeUpdate(edge_embed_dim, edge_attn_size, edge_attn_heads, edge_dropout)
        
    def forward(self, node_repr, edge_repr, padding_mask):
        bsz = node_repr.size(0)
        seq_len = node_repr.size(1)
        # account for padding while computing the representation
        if padding_mask is not None:
            node_repr = node_repr * (1 - padding_mask.unsqueeze(-1).type_as(node_repr))
            mol_true = ~padding_mask
            edge_mask = mol_true.unsqueeze(1) * mol_true.unsqueeze(2)

        #node self attn with edge bias
        node_attn_bias = self.proj_edge_bias(edge_repr)
        node_attn_bias = fill_attn_mask(node_attn_bias.permute(0, 3, 1, 2).contiguous(), padding_mask)
        node_repr, node_attn_bias, _ = self.node_attn(node_repr, attn_bias=node_attn_bias, return_attn=True)
        node_attn_bias[node_attn_bias == float("-inf")] = 0
        node_attn_bias = node_attn_bias.view(bsz, -1 ,seq_len, seq_len).permute(0, 2, 3, 1).contiguous()

        #edge update and self triangle attn
        edge_repr = edge_repr + self.node2edge(node_attn_bias)
        edge_repr = self.edge_attn(edge_repr, edge_mask)
        
        return node_repr, edge_repr
        

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layers,
        node_emb_dropout,
        node_embed_dim,
        node_ffn_dim,
        node_attn_heads,
        node_ffn_dropout,
        node_attn_dropout,
        edge_embed_dim, 
        edge_attn_heads,
        edge_attn_size, 
        edge_dropout,
        post_ln=False
    ) -> None:

        super().__init__()
        self.node_emb_dropout = node_emb_dropout
        self.node_emb_layer_norm = nn.LayerNorm(node_embed_dim)
        if not post_ln:
            self.final_layer_norm = nn.LayerNorm(node_embed_dim)
        else:
            self.final_layer_norm = None
    
        self.layers = nn.ModuleList([TransformerBlock(node_embed_dim,
                                                      node_ffn_dim,
                                                      node_attn_heads,
                                                      node_ffn_dropout,
                                                      node_attn_dropout,
                                                      edge_embed_dim, 
                                                      edge_attn_heads,
                                                      edge_attn_size, 
                                                      edge_dropout) for _ in range(encoder_layers)])

    def forward(
        self,
        node_repr: torch.Tensor,
        edge_repr: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        node_repr = self.node_emb_layer_norm(node_repr)
        node_repr = F.dropout(node_repr, p=self.node_emb_dropout, training=self.training)

        for i in range(len(self.layers)):
            node_repr, edge_repr = self.layers[i](node_repr, edge_repr, padding_mask)

        if self.final_layer_norm is not None:
            node_repr = self.final_layer_norm(node_repr)

        return node_repr, edge_repr
