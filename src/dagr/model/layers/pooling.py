import torch
import torch_scatter

from torch_cluster import grid_cluster
from torch_geometric.nn.pool.avg_pool import _avg_pool_x
from torch_geometric.nn.pool.pool import pool_pos
from torch_geometric.data import Data, Batch
from dagr.model.layers.components import BatchNormData
from typing import List, Callable


def consecutive_cluster(src):
    unique, inv, counts = torch.unique(src, sorted=True, return_inverse=True, return_counts=True)
    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
    perm = inv.new_empty(unique.size(0)).scatter_(0, inv, perm)
    return unique, inv, perm, counts


class Pooling(torch.nn.Module):
    def __init__(self, size: List[float], width, height, batch_size, transform: Callable[[Data, ], Data], aggr: str = 'max', keep_temporal_ordering=False, dim=2, self_loop=False, in_channels=-1):
        super(Pooling, self).__init__()
        assert aggr in ['mean', 'max']
        self.aggr = aggr
        self.register_buffer("voxel_size", torch.cat([size, torch.Tensor([1])]), persistent=False)

        self.transform = transform
        self.keep_temporal_ordering = keep_temporal_ordering
        self.dim = dim

        self.register_buffer("start", torch.Tensor([0,0,0,0]), persistent=False)
        self.register_buffer("end", torch.Tensor([0.9999999,0.9999999,0.9999999,batch_size-1]), persistent=False)
        self.register_buffer("wh_inv", 1/torch.Tensor([[width, height]]), persistent=False)

        self.max_num_voxels = batch_size * self.num_grid_cells
        self.register_buffer("sorted_cluster", torch.arange(self.max_num_voxels), persistent=False)

        self.self_loop = self_loop

        self.bn = None
        if in_channels > 0:
            self.bn = BatchNormData(in_channels)

    @property
    def num_grid_cells(self):
        return (1/self.voxel_size+1e-3).int().prod()
    
    def round_to_pixel(self, pos, wh_inv):
        torch.div(pos+1e-5, wh_inv, out=pos, rounding_mode='floor')
        return pos * wh_inv

    def forward(self, data: Data):
        if data.x.shape[0] == 0:
            return data

        pos = torch.cat([data.pos, data.batch.float().view(-1,1)], dim=-1)
        cluster = grid_cluster(pos, size=self.voxel_size, start=self.start, end=self.end)
        unique_clusters, cluster, perm, _ = consecutive_cluster(cluster)
        edge_index = cluster[data.edge_index]
        if self.self_loop:
            edge_index = edge_index.unique(dim=-1)
        else:
            edge_index = edge_index[:, edge_index[0]!=edge_index[1]]
            if edge_index.shape[1] > 0:
                edge_index = edge_index.unique(dim=-1)

        batch = None if data.batch is None else data.batch[perm]
        pos = None if data.pos is None else pool_pos(cluster, data.pos)

        if self.keep_temporal_ordering:
            t_max, _ = torch_scatter.scatter_max(data.pos[:,-1], cluster, dim=0)
            t_src, t_dst = t_max[edge_index]
            edge_index = edge_index[:, t_dst > t_src]

        if self.aggr == 'max':
            x, argmax = torch_scatter.scatter_max(data.x, cluster, dim=0)
        else:
            x = _avg_pool_x(cluster, data.x)

        new_data = Batch(batch=batch, x=x, edge_index=edge_index, pos=pos)

        if hasattr(data, "height"):
            new_data.height = data.height
            new_data.width = data.width

        # round x and y coordinates to the center of the voxel grid
        new_data.pos[:,:2] = self.round_to_pixel(new_data.pos[:,:2], wh_inv=self.wh_inv)

        if self.transform is not None:
            if new_data.edge_index.numel() > 0:
                new_data = self.transform(new_data)
            else:
                new_data.edge_attr = torch.zeros(size=(0,pos.shape[1]), dtype=pos.dtype, device=pos.device)

        if self.bn is not None:
            new_data = self.bn(new_data)

        return new_data
