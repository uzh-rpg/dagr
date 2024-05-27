import torch

from torch_geometric.nn import BatchNorm
from torch_geometric.data import Data

import torch_geometric.transforms as T


class BatchNormData(BatchNorm):
    def forward(self, data: Data):
        data.x = BatchNorm.forward(self, data.x)
        return data


class Linear(torch.nn.Module):
    def __init__(self, ic, oc, bias=True):
        torch.nn.Module.__init__(self)
        self.mlp = torch.nn.Linear(ic, oc, bias=bias)

    def forward(self, data: Data):
        data.x = self.mlp(data.x)
        return data


class Cartesian(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        T.Cartesian.__init__(self, *args, **kwargs)

    def forward(self, data):
        if data.edge_index.shape[1] > 0:
            return T.Cartesian.forward(self, data)
        else:
            data.edge_attr = torch.zeros((0, 3), dtype=data.x.dtype, device=data.x.device)
            return data