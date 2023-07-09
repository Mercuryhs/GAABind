import argparse

parser = argparse.ArgumentParser(description='')


# Hardware specifications
parser.add_argument('--seed', type=int, default=42,)

# Data specifications
parser.add_argument('--data_dir', type=str, default='example_data')


# Model specifications
parser.add_argument('--mol_atom_type', type=int, default=12)
parser.add_argument('--mol_node_feats', type=int, default=36)
parser.add_argument('--poc_atom_type', type=int, default=6)
parser.add_argument('--poc_node_feats', type=int, default=52)
parser.add_argument('--edge_feats', type=int, default=7)

#attn
parser.add_argument('--encoder_layers', type=int, default=5)
#node attn
parser.add_argument('--node_embed_dim', type=int, default=512)
parser.add_argument('--node_emb_dropout', type=float, default=0.1)
parser.add_argument('--node_ffn_dim', type=int, default=512)
parser.add_argument('--node_attn_heads', type=int, default=64)
parser.add_argument('--node_attn_dropout', type=float, default=0.1)
parser.add_argument('--node_ffn_dropout', type=float, default=0.1)


#edge attn
parser.add_argument('--edge_embed_dim', type=int, default=64)
parser.add_argument('--edge_attn_heads', type=int, default=4)
parser.add_argument('--edge_attn_size', type=int, default=32)
parser.add_argument('--edge_dropout', type=float, default=0.1)


#inter attn
parser.add_argument('--pair_hidden_dim', type=int, default=64)
parser.add_argument('--pair_attn_size', type=int, default=32)
parser.add_argument('--pair_attn_heads', type=int, default=4)
parser.add_argument('--pair_drop_ratio', type=int, default=0.25)
parser.add_argument('--pair_blocks', type=int, default=4)

# Training specifications
parser.add_argument('--save_path', type=str, default='./predict_result')
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--gpu', type=int, default='0')
parser.add_argument('--num_workers', type=int,  default=4)
parser.add_argument('--ckpt_path', type=str,  default='saved_model/best_epoch.pt')


args = parser.parse_args()
