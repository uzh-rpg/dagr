import torch

from torch_geometric.nn.conv import SplineConv
from torch_geometric.data import Data
from torch_geometric.transforms.to_sparse_tensor import ToSparseTensor
from torch_spline_conv import spline_basis


class MySplineConv(SplineConv):
    def __init__(self, in_channels, out_channels, args, bias=False, degree=1, **kwargs):
        self.reproducible = True
        self.to_sparse_tensor = ToSparseTensor(attr="edge_attr", remove_edge_index=False)
        super().__init__(in_channels=in_channels, out_channels=out_channels, bias=bias, degree=degree,
                         dim=args.edge_attr_dim, aggr=args.aggr, kernel_size=args.kernel_size)

    def init_lut(self, height, width, rx=None, Mx=None, ry=None, My=None):
        # attr is assumed to be computed as attr = (x_i - x_j)/(2M) + 0.5
        # where -r <= x_i - x_j <= r. So remapping to integers gives
        # lut_index = 2M*attr - M + r. and 0 <= lut_index <= 2r

        ry = ry or rx
        My = My or Mx
        self.attr_remapping_matrix = torch.Tensor([[2 * Mx * width,               0, - Mx * width + rx],
                                                   [             0, 2 * My * height, - My * height + ry]])

        # generate all possible dx, dy
        dxy = torch.stack(torch.meshgrid(torch.arange(-rx, rx+1), torch.arange(-ry, ry+1))).float()
        dxy[0] = dxy[0] / (2 * Mx * width) + 0.5
        dxy[1] = dxy[1] / (2 * My * height) + 0.5
        edge_attr = dxy.view((2,-1)).t()

        bil_w, indices = spline_basis(edge_attr.to(self.weight.data.device), self.kernel_size, self.is_open_spline, self.degree)
        lut_weights = (bil_w[...,None,None] * self.weight[indices]).sum(1)
        _, cin, cout = lut_weights.shape
        self.lut_weights = lut_weights.view((2 * rx + 1, 2 * ry + 1, cin, cout))

        self.message = self.message_lut

    def message_lut(self, x_j, edge_attr):
        # index = (attr - 0.5) * 2 * M + r
        dx_index = (edge_attr[:,0] * self.attr_remapping_matrix[0,0] + self.attr_remapping_matrix[0,-1]+1e-3).long()
        dy_index = (edge_attr[:,1] * self.attr_remapping_matrix[1,1] + self.attr_remapping_matrix[1,-1]+1e-3).long()

        weights = self.lut_weights[dx_index, dy_index] # N x C_out x C_in
        x_out = torch.einsum("nio,ni->no", weights, x_j)

        return x_out

    def forward(self, data: Data)->Data:
        if self.reproducible:
            # first check we already computed the adjacency matrix
            if not hasattr(data, "adj_t"):
                data.edge_attr = data.edge_attr[:,:self.dim]
                data = self.to_sparse_tensor(data)
            data.x = self._forward(data.x,
                                  edge_index=data.adj_t)
        else:
            data.x = self._forward(data.x,
                                  edge_index=data.edge_index,
                                  edge_attr=data.edge_attr[:, :self.dim],
                                  size=(data.x.shape[0], data.x.shape[0]))
        return data

    def _forward(self, x, edge_index, edge_attr=None, size=None):
        """"""
        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        if edge_index.numel() > 0:
            out = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr, size=size)
        else:
            out = torch.zeros((x.size(0), self.out_channels), dtype=x.dtype, device=x.device)

        if x is not None and self.root_weight:
            out += self.lin(x)

        if self.bias is not None:
            out += self.bias

        return out

def to_dense(self, x, pos, pooling, batch=None, batch_size=None):
    if hasattr(self, "batch_size"):
        B = self.batch_size
    elif batch_size is not None:
        self.batch_size = batch_size
        B = batch_size
    elif batch is None:
        batch = torch.zeros(size=(len(x),), dtype=torch.long, device=x.device)
        B = 1
        self.batch_size = B
    else:
        B = batch.max().item() + 1
        self.batch_size = B

    if not hasattr(self, "dense"):
        W, H = (1 / pooling[:2] + 1e-3).long()
        C = x.shape[-1]
        self.dense = torch.zeros(size=(B, C, H, W), dtype=x.dtype, device=x.device)

    est_x, est_y = (pos[:, :2] / pooling[:2]).t().long()

    self.dense = self.dense.detach()
    self.dense.zero_()

    dense = self.dense[:B] if B < self.dense.shape[0] else self.dense
    dense[batch.long(), :, est_y, est_x] = x

    return dense


class SplineConvToDense(MySplineConv):
    def forward(self, data: Data, batch_size: int=None)->torch.Tensor:
        data = super().forward(data)
        if data.batch is None:
            data.batch = torch.zeros(len(data.x), dtype=torch.long, device=data.x.device)
        return self.to_dense(data.x, data.pos, data.pooling, data.batch, batch_size=batch_size)

    def to_dense(self, x, pos, pooling, batch=None, batch_size=None):
        return to_dense(self, x, pos, pooling, batch, batch_size=batch_size)