import torch
import asy_tools
from torch_geometric.nn.norm import BatchNorm
import torch.nn.functional as F
from .base.base import make_asynchronous, add_async_graph
from .base.utils import graph_changed_nodes, graph_new_nodes


def __sync_forward(m, x):
    return F.batch_norm(x, m.running_mean, m.running_var, m.weight, m.bias, False, m.momentum, m.eps)


def __graph_initialization(module: BatchNorm, data) -> torch.Tensor:
    module.asy_graph = data.clone()
    module.graph_out = data.clone()
    module.graph_out.x = __sync_forward(module.module, data.x)

    # flops are not counted since BN can be fused with previous conv operator.
    if module.asy_flops_log is not None:
        flops = 0
        module.asy_flops_log.append(flops)

    return module.graph_out.clone()

def __graph_processing(module: BatchNorm, data) -> torch.Tensor:
    """Batch norms only execute simple normalization operation, which already is very efficient. The overhead
    for looking for diff nodes would be much larger than computing the dense update.

    However, a new node slightly changes the feature distribution and therefore all activations, when calling
    the dense implementation. Therefore, we approximate the distribution with the initial distribution as
    num_new_events << num_initial_events.
    """
    if len(module.asy_graph.x) < len(data.x):
        diff_idx = graph_new_nodes(module.asy_graph, data)
        module.graph_out.x = torch.cat([module.graph_out.x, torch.zeros_like(data.x[:len(diff_idx)])])
    else:
        diff_idx, _ = graph_changed_nodes(module.asy_graph, data)

    if data.diff_idx.numel()>0:
        asy_tools.masked_inplace_BN(data.diff_idx, data.x,
                                    module.graph_out.x,
                                    module.module.running_mean,
                                    module.module.running_var,
                                    module.module.weight,
                                    module.module.bias,
                                    module.module.eps)

    # If required, compute the flops of the asynchronous update operation.
    if module.asy_flops_log is not None:
        flops = 0
        module.asy_flops_log.append(flops)

    data.x = module.graph_out.x

    return data


def __check_support(module):
    return True


def make_batch_norm_asynchronous(module: BatchNorm, log_flops: bool = False):
    """Module converter from synchronous to asynchronous & sparse processing for batch norm (1d) layers.
    By overwriting parts of the module asynchronous processing can be enabled without the need of re-learning
    and moving its weights and configuration. So, a layer can be converted by, for example:

    ```
    module = BatchNorm(4)
    module = make_batch_norm_asynchronous(module)
    ```

    :param module: batch norm module to transform.
    :param log_flops: log flops of asynchronous update.
    """
    assert __check_support(module)
    module = add_async_graph(module, log_flops=log_flops)
    return make_asynchronous(module, __graph_initialization, __graph_processing)
