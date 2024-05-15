import numpy as np
import torch
from torch_geometric.data import Data


def to_data(**kwargs):
    # convert all tracks to correct format
    for k, v in kwargs.items():
        if k.startswith("bbox"):
            kwargs[k] = torch.from_numpy(v)

    xy = np.stack([kwargs['x'], kwargs['y']], axis=-1).astype("int16")
    t = kwargs['t'].astype("int32")
    p = kwargs['p'].reshape((-1,1))

    kwargs['x'] = torch.from_numpy(p)
    kwargs['pos'] = torch.from_numpy(xy)
    kwargs['t'] = torch.from_numpy(t)

    return Data(**kwargs)