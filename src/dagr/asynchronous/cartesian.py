import torch

from torch_geometric.nn.norm import BatchNorm
from .base.base import make_asynchronous, add_async_graph

def __edge_attr(pos, edge_index, norm, max):
    (row, col), pos = edge_index, pos

    cart = pos[row] - pos[col]
    cart = cart.view(-1, 1) if cart.dim() == 1 else cart

    if norm and cart.numel() > 0:
        max_value = cart.abs().max() if max is None else max
        cart = cart / (2 * max_value) + 0.5

    return cart


def __graph_initialization(module: BatchNorm, data) -> torch.Tensor:
    module.asy_graph = data.clone()
    module.graph_out = data.clone()
    module.graph_out.edge_attr = __edge_attr(data.pos, data.edge_index, module.norm, module.max)

    # flops are not counted since BN can be fused with previous conv operator.
    if module.asy_flops_log is not None:
        flops = 2 * len(module.graph_out.edge_attr)
        module.asy_flops_log.append(flops)

    return module.graph_out.clone()


def __graph_processing(module: BatchNorm, data) -> torch.Tensor:
    """Batch norms only execute simple normalization operation, which already is very efficient. The overhead
    for looking for diff nodes would be much larger than computing the dense update.

    However, a new node slightly changes the feature distribution and therefore all activations, when calling
    the dense implementation. Therefore, we approximate the distribution with the initial distribution as
    num_new_events << num_initial_events.
    """
    module.graph_out.pos = torch.cat([module.asy_graph.pos, data.pos])
    module.graph_out.x = torch.cat([module.asy_graph.x, data.x])
    module.graph_out.edge_attr = __edge_attr(module.graph_out.pos, data.edge_index, module.norm, module.max)
    module.graph_out.edge_index = data.edge_index

    # flops are not counted since BN can be fused with previous conv operator.
    if module.asy_flops_log is not None:
        flops = 2 * len(module.graph_out.edge_attr)
        module.asy_flops_log.append(flops)

    if hasattr(data, "diff_idx"):
        module.graph_out.diff_idx = data.diff_idx
        module.graph_out.diff_pos_idx = data.diff_pos_idx

    return module.graph_out


def __check_support(module):
    return True


def make_cartesian_asynchronous(module: BatchNorm, log_flops: bool = False):
    """Module converter from synchronous to asynchronous & sparse processing for cartesian layers.
    By overwriting parts of the module asynchronous processing can be enabled without the need of re-learning
    and moving its weights and configuration. So, a layer can be converted by, for example:

    ```
    module = Cartesian()
    module = make_cartesian_asynchronous(module)
    ```

    :param module: cartesian module to transform.
    :param log_flops: log flops of asynchronous update.
    """
    assert __check_support(module)
    module = add_async_graph(module, log_flops=log_flops)
    return make_asynchronous(module, __graph_initialization, __graph_processing)