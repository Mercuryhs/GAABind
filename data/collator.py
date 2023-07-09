import torch

def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_pos_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def edge_type(atoms, ligand=True):
    if ligand:
        atom_num = 12
    else:
        atom_num = 6
    atom_offset = atoms.view(-1, 1) * atom_num + atoms.view(1, -1)
    return atom_offset



def pad_graph(items, ligand=True):
    if ligand:
        items = [(item.attn_edge_type, item.x, item.pos) for item in items]
        attn_edge_types, xs, poses = zip(*items)
    else:
        items = [(item.attn_edge_type, item.x, item.pos) for item in items]
        attn_edge_types, xs, poses = zip(*items)
        
    
    max_node_num = max(i.size(0) for i in xs)
    pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    attn_edge_type = torch.cat([pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])

    node_type_edges = []
    for idx in range(len(items)):
        node_atom_type = items[idx][1][:, 0]
        node_atom_type = pad_1d_unsqueeze(node_atom_type, max_node_num)
        node_atom_edge = edge_type(node_atom_type, ligand=ligand)
        node_type_edges.append(node_atom_edge.long().unsqueeze(0))
    node_type_edge = torch.cat(node_type_edges, dim=0)
    
    if ligand:
        item_dict = dict(
        attn_edge_type=attn_edge_type,
        x=x,
        pos=pos,
        node_type_edge=node_type_edge)
    else:
        item_dict = dict(
        attn_edge_type=attn_edge_type,
        x=x,
        pos=pos,
        node_type_edge=node_type_edge)
        
    return item_dict


def pad_info(items):
    smi_list = [item['smi'] for item in items]
    pocket_list = [item['pocket'] for item in items]
    holo_center_list = torch.cat([item['holo_center_coordinates'].unsqueeze(0) for item in items], dim=0)
    mol_list = [item['mol'] for item in items]
    
    info_dict = dict(smi_list=smi_list, pocket_list=pocket_list, holo_center_list = holo_center_list, mol = mol_list)
    return info_dict

def collator_3d(items):
    lig_graph = [item[0] for item in items]
    poc_graph = [item[1] for item in items]
    infos = [item[2] for item in items]
    
    lig_dict = pad_graph(lig_graph)
    poc_dict = pad_graph(poc_graph, ligand=False)
    info_dict = pad_info(infos)

    return lig_dict, poc_dict, info_dict
