import logging
from torch.nn import ModuleList

from .conv import compute_flops_conv, compute_flops_cat


def compute_flops_from_module(module) -> int:
    """Compute flops from a GNN module (after the forward pass).

    Generally, there are two cases. Either the module is an asynchronous module, then it should
    have an `flops_log`, which contains the flops used for the last forward pass. Otherwise, the
    layer's flops are computed from to the synchronous, dense update.

    :param module: module to infer the flops from.
    """
    module_name = module.__class__.__name__

    if hasattr(module, "asy_flops_log") and module.asy_flops_log is not None:
        assert type(module.asy_flops_log) == list, "asyc. flops log must be a list"
        if type(module) is ModuleList:
            flops = sum([compute_flops_from_module(layer) for layer in module._modules.values()])
        else:
            assert len(module.asy_flops_log) > 0, f"asynchronous flops log is empty for module {module.__class__.__name__}"
            flops = module.asy_flops_log[-1]
    else:
        logging.debug(f"Module {module_name} is not asynchronous, using flops = 0")
        return 0

    logging.debug(f"Module {module_name} adds {flops} flops")
    return flops


__all__ = [
    "compute_flops_conv",
    "compute_flops_from_module"
]
