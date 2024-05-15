import logging

import torch.nn
import torch_geometric
import inspect

from torch.nn import ModuleList

from .conv import make_conv_asynchronous
from .batch_norm import make_batch_norm_asynchronous
from .linear import make_linear_asynchronous
from .max_pool import make_max_pool_asynchronous
from .cartesian import make_cartesian_asynchronous

from .flops import compute_flops_from_module

from dagr.model.layers.spline_conv import MySplineConv
from dagr.model.layers.pooling import Pooling
from dagr.model.layers.components import BatchNormData, Cartesian, Linear



from torch_geometric.data import Data, Batch
from typing import List


def is_data_or_data_list(ann):
    return ann is Data or ann is Batch or ann is List[Data]

def make_model_synchronous(module: torch.nn.Module):
    module.forward = module.sync_forward
    module.asy_flops_log = []

    for key, nn in module.named_modules():
        if hasattr(nn, "sync_forward"):
            nn.forward = nn.sync_forward
            nn.asy_flops_log = []

    return module

def make_model_asynchronous(module, log_flops: bool = False):
    """Module converter from synchronous to asynchronous & sparse processing for graph convolutional layers.
    By overwriting parts of the module asynchronous processing can be enabled without the need of re-learning
    and moving its weights and configuration. So, a convolutional layer can be converted by, for example:

    ```
    module = GCNConv(1, 2)
    module = make_conv_asynchronous(module)
    ```

    :param module: convolutional module to transform.
    :param grid_size: grid size (grid starting at 0, spanning to `grid_size`), >= `size` for pooling operations,
                      e.g. the image size.
    :param r: update radius around new events.
    :param edge_attributes: function for computing edge attributes (default = None), assumed to be the same over
                            all convolutional layers.
    :param log_flops: log flops of asynchronous update.
    """
    assert isinstance(module, torch.nn.Module), "module must be a `torch.nn.Module`"
    model_forward = module.forward
    module.sync_forward = module.forward

    module.asy_flops_log = [] if log_flops else None

    # Make all layers asynchronous that have an implemented asynchronous function. Otherwise use
    # the synchronous forward function.
    for key, nn in module._modules.items():
        nn_class_name = nn.__class__.__name__
        logging.debug(f"Making layer {key} of type {nn_class_name} asynchronous")

        if isinstance(nn, MySplineConv):
            module._modules[key] = make_conv_asynchronous(nn, log_flops=log_flops)

        elif isinstance(nn, Pooling):
            module._modules[key] = make_max_pool_asynchronous(nn, log_flops=log_flops)

        elif isinstance(nn, BatchNormData):
            module._modules[key] = make_batch_norm_asynchronous(nn, log_flops=log_flops)

        elif isinstance(nn, Cartesian):
            module._modules[key] = make_cartesian_asynchronous(nn, log_flops=log_flops)

        elif isinstance(nn, Linear):
            module._modules[key] = make_linear_asynchronous(nn, log_flops=log_flops)

        elif isinstance(nn, ModuleList):
            module._modules[key] = make_model_asynchronous(nn, log_flops=log_flops)

        else:
            sign = inspect.signature(nn.forward)
            first_arg = list(sign.parameters.values())[0]

            if not is_data_or_data_list(first_arg.annotation):
                continue

            module._modules[key] = make_model_asynchronous(nn, log_flops=log_flops)
            logging.debug(f"Asynchronous module for {nn_class_name} is being made asynchronous recursively.")

    def async_forward(data: torch_geometric.data.Data, *args, **kwargs):
        out = model_forward(data, *args, **kwargs)

        if module.asy_flops_log is not None:
            flops_count = [compute_flops_from_module(layer) for layer in module._modules.values()]
            module.asy_flops_log.append(sum(flops_count))
            logging.debug(f"Model's modules update with overall {sum(flops_count)} flops")

        return out

    module.forward = async_forward
    return module


__all__ = [
    "make_conv_asynchronous",
    "make_linear_asynchronous",
    "make_max_pool_asynchronous",
    "make_model_asynchronous"
]
