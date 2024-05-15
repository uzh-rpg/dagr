import numpy as np
import torch
import torch_geometric
import asy_tools

from torch.nn import Linear
import torch.nn.functional as F
from .base.base import make_asynchronous, add_async_graph
from .base.utils import graph_new_nodes, graph_changed_nodes


def __graph_initialization(module: Linear, data) -> torch.Tensor:
    mask = data.active_clusters if hasattr(data, "active_clusters") else slice(None, None, None)
    x = data.x[mask]
    weight = module.mlp.weight
    bias = module.mlp.bias

    y = torch.zeros(size=(len(data.x), weight.shape[0]), dtype=torch.float32, device=data.pos.device)
    y[mask] = F.linear(x, weight, bias)

    module.asy_graph = data.clone()
    module.graph_out = torch_geometric.data.Data(x=y, pos=data.pos)
    if hasattr(data, "active_clusters"):
        module.graph_out.active_clusters = data.active_clusters

    if module.asy_flops_log is not None:
        flops = int(np.prod(x.size()) * y.size()[-1])
        module.asy_flops_log.append(flops)

    return module.graph_out.clone()

def __graph_processing(module: Linear, data) -> torch.Tensor:
    if len(module.asy_graph.x) < len(data.x):
        diff_idx = graph_new_nodes(module.asy_graph, data)
        diff_pos_idx = diff_idx.clone()
        module.graph_out.x = torch.cat([module.graph_out.x, torch.zeros_like(module.graph_out.x[:len(diff_idx)])])
    else:
        diff_idx, diff_pos_idx = graph_changed_nodes(module.asy_graph, data)

    weight = module.mlp.weight
    bias = module.mlp.bias

    # Update the graph with the new values (only there where it has changed).
    if diff_idx.numel() > 0:
        if bias is not None:
           asy_tools.masked_lin(diff_idx, data.x, module.graph_out.x, weight.data, bias.data, False)
        else:
           asy_tools.masked_lin_no_bias(diff_idx, data.x, module.graph_out.x, weight.data, False)

    # If required, compute the flops of the asynchronous update operation.
    if module.asy_flops_log is not None:
        cin = weight.shape[1]
        cat = hasattr(data, "skipped") and data.skipped
        data.skipped = False

        if cat:
            cin -= data.num_image_channels

        flops = diff_idx.numel() * int(weight.shape[0] * (2*cin-1))
        flops += diff_idx.numel() * weight.shape[0]
        module.asy_flops_log.append(flops)

    data.diff_idx = diff_idx
    data.diff_pos_idx = diff_pos_idx
    data.x = module.graph_out.x

    return data

def __check_support(module: Linear):
    return True


def make_linear_asynchronous(module: Linear, log_flops: bool = False):
    """Module converter from synchronous to asynchronous & sparse processing for linear layers.
    By overwriting parts of the module asynchronous processing can be enabled without the need of re-learning
    and moving its weights and configuration. So, a linear layer can be converted by, for example:

    ```
    module = Linear(4, 2)
    module = make_linear_asynchronous(module)
    ```

    :param module: linear module to transform.
    :param log_flops: log flops of asynchronous update.
    """
    assert __check_support(module)
    module = add_async_graph(module, log_flops=log_flops)
    return make_asynchronous(module, __graph_initialization, __graph_processing)
