import torch
import torch.nn as nn


class Mol2Pair(nn.Module):
    def __init__(self, pair_dim, hidden_dim):
        super().__init__()
        self.layernorm = nn.LayerNorm(pair_dim)
        self.layernorm_hidden = nn.LayerNorm(hidden_dim)
        
        self.gate_linear1 = nn.Linear(pair_dim, hidden_dim)
        self.gate_linear2 = nn.Linear(pair_dim, hidden_dim)
        self.linear1 = nn.Linear(pair_dim, hidden_dim)
        self.linear2 = nn.Linear(pair_dim, hidden_dim)
        
        self.end_gate = nn.Linear(pair_dim, pair_dim)
        self.liner_after_sum = nn.Linear(hidden_dim, pair_dim)
        
    def forward(self, z, z_mask, pocket_pair, mol_pair):
        z_mask = z_mask.unsqueeze(-1)
        z = self.layernorm(z) #z of shape B * L * L * D
    
        pocket_pair = self.layernorm(pocket_pair)
        mol_pair = self.layernorm(mol_pair)
        
        
        ab1 = self.gate_linear1(z).sigmoid() * self.linear1(z) * z_mask
        ab2 = self.gate_linear2(z).sigmoid() * self.linear2(z) * z_mask
        pocket_pair = self.gate_linear2(pocket_pair).sigmoid() * self.linear2(pocket_pair)
        mol_pair = self.gate_linear1(mol_pair).sigmoid() * self.linear1(mol_pair)

        block1 = torch.einsum("bikc,bkjc->bijc", pocket_pair, ab1)
        block2 = torch.einsum("bikc,bjkc->bijc", ab2, mol_pair)
        z = self.end_gate(z).sigmoid() * self.liner_after_sum(self.layernorm_hidden(block1+block2)) * z_mask
    
        return z


class PairAxisAttention(nn.Module):
   
    def __init__(self, pair_dim=128, attention_head_size=32, num_attention_heads=4, row_attn=True):
        super().__init__()
        self.row_attn = row_attn
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
       

        self.layernorm = nn.LayerNorm(pair_dim)

        self.linear_q = nn.Linear(pair_dim, self.all_head_size, bias=False)
        self.linear_k = nn.Linear(pair_dim, self.all_head_size, bias=False)
        self.linear_v = nn.Linear(pair_dim, self.all_head_size, bias=False)

        self.g = nn.Linear(pair_dim, self.all_head_size)
        self.final_linear = nn.Linear(self.all_head_size, pair_dim)

    def reshape_last_dim(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, z, z_mask):
        if not self.row_attn:
            z = z.transpose(1, 2)
            z_mask = z_mask.transpose(1, 2)
        
        z = self.layernorm(z)
        p_length = z.shape[1]
        batch_n = z.shape[0]
        
        z_mask_i = z_mask.view((batch_n, p_length, 1, 1, -1))
        attention_mask_i = (1e9 * (z_mask_i.float() - 1.))
        
        q = self.reshape_last_dim(self.linear_q(z)) * (self.attention_head_size**(-0.5))
        k = self.reshape_last_dim(self.linear_k(z))
        v = self.reshape_last_dim(self.linear_v(z))
        weights = torch.einsum('biqhc,bikhc->bihqk', q, k) + attention_mask_i
        
        weights = nn.Softmax(dim=-1)(weights)
       
        weighted_avg = torch.einsum('bihqk,bikhc->biqhc', weights, v)
        z = self.reshape_last_dim(self.g(z)).sigmoid() * weighted_avg
        
        new_output_shape = z.size()[:-2] + (self.all_head_size,)
        z = z.view(*new_output_shape)

        z = self.final_linear(z) * z_mask.unsqueeze(-1)
        
        if not self.row_attn:
            z = z.transpose(1, 2)
        return z



class PairAxisAttention_new_new_new(nn.Module):
   
    def __init__(self, pair_dim=128, attention_head_size=32, num_attention_heads=4, row_attn=True):
        super().__init__()
        self.row_attn = row_attn
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
       

        self.layernorm = nn.LayerNorm(pair_dim)

        self.linear_q = nn.Linear(pair_dim, self.all_head_size, bias=False)
        self.linear_k = nn.Linear(pair_dim, self.all_head_size, bias=False)
        self.linear_v = nn.Linear(pair_dim, self.all_head_size, bias=False)

        self.g = nn.Linear(pair_dim, self.all_head_size)
        self.final_linear = nn.Linear(self.all_head_size, pair_dim)

    def reshape_last_dim(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, z, z_mask):
        if not self.row_attn:
            z = z.transpose(1, 2)
            z_mask = z_mask.transpose(1, 2)
        
        z = self.layernorm(z)
        p_length = z.shape[1]
        batch_n = z.shape[0]
        
        z_mask_i = z_mask.view((batch_n, p_length, 1, 1, -1))
        attention_mask_i = (1e9 * (z_mask_i.float() - 1.))
        
        q = self.reshape_last_dim(self.linear_q(z)) * (self.attention_head_size**(-0.5))
        k = self.reshape_last_dim(self.linear_k(z))
        v = self.reshape_last_dim(self.linear_v(z))
        
        weights_1 = torch.einsum('biqhc,bikhc->bihqk', q, k) + attention_mask_i
        weights_1 = nn.Softmax(dim=-1)(weights_1)
        weighted_avg_1 = torch.einsum('bihqk,bikhc->biqhc', weights_1, v)
        z_1 = self.reshape_last_dim(self.g(z)).sigmoid() * weighted_avg_1
        
        weights_2 = torch.einsum('biqhc,bikhc->bihqk', q.transpose(1, 2), k.transpose(1, 2)) + attention_mask_i.transpose(1, 4)
        weights_2 = nn.Softmax(dim=-1)(weights_2)
        weighted_avg_2 = torch.einsum('bihqk,bikhc->biqhc', weights_2, v.transpose(1, 2))
        z_2 = self.reshape_last_dim(self.g(z.transpose(1,2))).sigmoid() * weighted_avg_2
        
        z = z_1 + z_2.transpose(1,2)

        new_output_shape = z.size()[:-2] + (self.all_head_size,)
        z = z.view(*new_output_shape)

        z = self.final_linear(z) * z_mask.unsqueeze(-1)
        
        if not self.row_attn:
            z = z.transpose(1, 2)
        return z



class MolTriangleUpdate(nn.Module):
    def __init__(self, pair_dim, hidden_dim):
        super().__init__()
        
        self.layernorm = nn.LayerNorm(pair_dim)
        self.layernorm_hidden = nn.LayerNorm(hidden_dim)
        
        self.gate_linear1 = nn.Linear(pair_dim, hidden_dim)
        self.gate_linear2 = nn.Linear(pair_dim, hidden_dim)
        self.linear1 = nn.Linear(pair_dim, hidden_dim)
        self.linear2 = nn.Linear(pair_dim, hidden_dim)
        
        self.end_gate = nn.Linear(pair_dim, pair_dim)
        self.liner_after_sum = nn.Linear(hidden_dim, pair_dim)
        
    def forward(self, z, z_mask):
        z_mask = z_mask.unsqueeze(-1)
        z = self.layernorm(z) #z of shape B * L * L * D
        
        ab1 = self.gate_linear1(z).sigmoid() * self.linear1(z) * z_mask
        ab2 = self.gate_linear2(z).sigmoid() * self.linear2(z) * z_mask
        
        block = torch.einsum("bikc,bjkc->bijc", ab1, ab2) + torch.einsum("bkic,bkjc->bijc", ab1, ab2)
        
        z = self.end_gate(z).sigmoid() * self.liner_after_sum(self.layernorm_hidden(block)) * z_mask

        return z


class MolTriangleAttention(nn.Module):
   
    def __init__(self, pair_dim=128, attention_head_size=32, num_attention_heads=4):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
       

        self.layernorm = nn.LayerNorm(pair_dim)

        self.linear_q = nn.Linear(pair_dim, self.all_head_size, bias=False)
        self.linear_k = nn.Linear(pair_dim, self.all_head_size, bias=False)
        self.linear_v = nn.Linear(pair_dim, self.all_head_size, bias=False)
        
        self.attn_bias = nn.Linear(pair_dim, self.num_attention_heads, bias=False)

        self.g = nn.Linear(pair_dim, self.all_head_size)
        self.final_linear = nn.Linear(self.all_head_size, pair_dim)

    def reshape_last_dim(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, z, z_mask):
    
        z = self.layernorm(z)
        p_length = z.shape[1]
        batch_n = z.shape[0]
        
        z_mask_i = z_mask.view((batch_n, p_length, 1, 1, -1))
        attention_mask_i = (1e9 * (z_mask_i.float() - 1.))
        
        q = self.reshape_last_dim(self.linear_q(z)) * (self.attention_head_size**(-0.5))
        k = self.reshape_last_dim(self.linear_k(z))
        v = self.reshape_last_dim(self.linear_v(z))
        attn_bias = self.attn_bias(z).permute(0, 3, 1, 2).unsqueeze(-4).contiguous()
        weights_1 = torch.einsum('biqhc,bikhc->bihqk', q, k) + attn_bias
        weights_2 = torch.einsum('biqhc,bikhc->bihqk', q.transpose(1, 2), k.transpose(1, 2)) + attn_bias.transpose(-1, -2)
        # ipdb.set_trace()
        weights = weights_1 + weights_2 + attention_mask_i
        weights = nn.Softmax(dim=-1)(weights)
        weighted_avg = torch.einsum('bihqk,bikhc->biqhc', weights, v + v.transpose(1, 2))
        
        z = self.reshape_last_dim(self.g(z)).sigmoid() * weighted_avg
        new_output_shape = z.size()[:-2] + (self.all_head_size,)
        z = z.view(*new_output_shape)

        z = self.final_linear(z) * z_mask.unsqueeze(-1)
        
        return z



class Transition(nn.Module):
    def __init__(self, embedding_channels=256, n=2):
        super().__init__()
        self.layernorm = nn.LayerNorm(embedding_channels)
        self.linear1 = nn.Linear(embedding_channels, n*embedding_channels)
        self.linear2 = nn.Linear(n*embedding_channels, embedding_channels)
    def forward(self, z):
        z = self.layernorm(z)
        z = self.linear2((self.linear1(z)).relu())
        return z



class PairUpdate(nn.Module):
    def __init__(self, pair_dim, hidden_dim, attn_size, attn_heads, drop_ratio):
        super().__init__()
        self.triangle_update = Mol2Pair(pair_dim, hidden_dim) 
        self.rowwise_attn = PairAxisAttention_new_new_new(pair_dim, attn_size, attn_heads) 
        self.dropout = nn.Dropout2d(p=drop_ratio)
        self.transition = Transition(pair_dim, n=2)
    def forward(self, z, z_mask, pocket_pair, mol_pair):
        z = z + self.dropout(self.triangle_update(z, z_mask, pocket_pair, mol_pair))
        z = z + self.dropout(self.rowwise_attn(z, z_mask))
        z = self.transition(z)
        return z
        
        
class Pair2mol(nn.Module):
    def __init__(self, pair_dim, hidden_dim, attn_size, attn_heads, dropout=0.2, pocket=True):
        super().__init__()
        self.pocket = pocket
        self.norm_z = nn.LayerNorm(pair_dim)
        self.drop_ratio = dropout
        self.dropout = nn.Dropout2d(p=self.drop_ratio)
        
        self.gate_linear1 = nn.Linear(pair_dim, hidden_dim)
        self.linear1 = nn.Linear(pair_dim, hidden_dim)

        self.update_norm = nn.LayerNorm(hidden_dim)
        self.update_linear = nn.Linear(hidden_dim, pair_dim)
        self.gate_mol_pair = nn.Linear(pair_dim, pair_dim)
        
        self.triangle_update = MolTriangleUpdate(pair_dim, pair_dim)
        self.triangle_attn = MolTriangleAttention(pair_dim, attn_size, attn_heads)
        self.transition = Transition(pair_dim, n=2)
        
        
    def forward(self, mol_pair, mol_mask, z, z_mask):
        z_mask = z_mask.unsqueeze(-1)
        z = self.norm_z(z)
        #z=B*M*N*D, z_mask=B*M*N*1
        z = self.gate_linear1(z).sigmoid() * self.linear1(z) * z_mask
        
        if not self.pocket:
            scaling_factor=z_mask[:,:,0,0].sum(dim=1,keepdims=True).unsqueeze(-1).unsqueeze(-1)
            z_for_mol = torch.einsum("bmik,bmjk->bijk", z, z)/torch.sqrt(scaling_factor)
        else:
            scaling_factor=z_mask[:,0,:,0].sum(dim=1,keepdims=True).unsqueeze(-1).unsqueeze(-1)
            z_for_mol = torch.einsum("bimk,bjmk->bijk", z, z)/torch.sqrt(scaling_factor)
        
        del z
        z_for_mol = self.gate_mol_pair(mol_pair).sigmoid() * self.update_linear(self.update_norm(z_for_mol))
        mol_pair = mol_pair + self.dropout(z_for_mol)
        
        del z_for_mol
        mol_pair = mol_pair + self.dropout(self.triangle_update(mol_pair, mol_mask))
        mol_pair = mol_pair + self.dropout(self.triangle_attn(mol_pair, mol_mask))
        mol_pair = self.transition(mol_pair)
        
        return mol_pair