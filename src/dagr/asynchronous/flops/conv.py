import torch


def compute_flops_conv(module: torch.nn.Module, num_times_apply_bias_and_root: int, num_edges: int, concatenation=False, num_image_channels=-1) -> int:
    # Iterate over every different and every new node, and add the number of flops introduced
    # by the node to the overall flops count of the layer.
    ni = num_edges

    m_in = module.in_channels

    if concatenation:
        m_in -= num_image_channels

    m_out = module.out_channels

    flops = ni * (2*m_in-1) * m_out

    if hasattr(module, "root_weight") and module.root_weight:
        flops += num_times_apply_bias_and_root * module.lin.weight.shape[0] * (2*module.lin.weight.shape[1]-1)

    if hasattr(module, "bias") and module.bias is not None:
        flops += num_times_apply_bias_and_root * module.lin.weight.shape[0]

    return flops


def compute_flops_cat(module, num_edges, num_times_apply_bias_and_root, num_image_channels):
    ni = num_edges
    m_in = num_image_channels
    m_out = module.out_channels

    flops =  ni * (2 * m_in - 1) * m_out

    if hasattr(module, "root_weight") and module.root_weight:
        flops += num_times_apply_bias_and_root * module.lin.weight.shape[0] * (2*m_in-1)

    return flops