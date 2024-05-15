import torch

from typing import Tuple
import asy_tools


def _efficient_cat(data_list):
    data_list = [d for d in data_list if len(d) > 0]
    if len(data_list) == 1:
        return data_list[0]
    return torch.cat(data_list)

def _efficient_cat_unique(data_list):
    # first only keep elements that have len > 0
    data_list_filt = [data for data in data_list if data.shape[0] > 0]
    if len(data_list_filt) == 1:
        return data_list_filt[0]
    elif len(data_list_filt) == 0:
        return data_list[0]
    else:
        return torch.cat(data_list_filt).unique()

def _to_hom(x, ones=None):
    if ones is None or len(ones) < len(x):
        ones = torch.ones_like(x[:,-1:])
    else:
        ones = ones[:len(x)]
    return torch.cat([x, ones], dim=-1)

def _from_hom(x):
    return x[:,:-1] / (x[:,-1:] + 1e-9)

def graph_new_nodes(old_data, new_data):
    return torch.arange(old_data.x.shape[0], new_data.x.shape[0], device=new_data.x.device, dtype=torch.long)

def graph_changed_nodes(old_data, new_data) -> Tuple[torch.Tensor, torch.Tensor]:
    len_x_old = old_data.x.shape[0]
    len_pos_old = old_data.pos.shape[0]
    x_new = new_data.x[:len_x_old] if len_x_old < new_data.x.shape[0] else new_data.x
    pos_new = new_data.pos[:len_pos_old] if len_pos_old < new_data.pos.shape[0] else new_data.pos

    diff_idx = asy_tools.masked_isdiff(new_data.diff_idx, x_new, old_data.x, 1e-8, 1e-5) if new_data.diff_idx.numel() > 0 else new_data.diff_idx
    diff_pos_idx = asy_tools.masked_isdiff(new_data.diff_pos_idx, pos_new, old_data.pos, 1e-8, 1e-5) if new_data.diff_pos_idx.numel() > 0 else new_data.diff_pos_idx

    return diff_idx, diff_pos_idx

def torch_isin(query, database):
    if hasattr(torch, "isin"):
        return torch.isin(query, database)
    else:
        return (query.view(1, -1) == database.view(-1, 1)).any(0)

def __remove_duplicate_from_A(a, b):
    a_in_b = (a.view(2,1,-1) == b.view(2,-1,1)).all(0).any(0)
    return a[:,~a_in_b]