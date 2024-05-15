import asy_tools
import torch
import torch_geometric.nn.conv

from .base.base import make_asynchronous, add_async_graph
from .base.utils import graph_new_nodes, graph_changed_nodes, _efficient_cat_unique, torch_isin
from .flops import compute_flops_conv, compute_flops_cat
from torch_scatter import scatter_sum


def __conv(x, edge_index, edge_attr, mask, nn):
    if edge_index.numel() > 0:
        x_j = x[edge_index[0, :], :]
        phi = nn.message(x_j, edge_attr=edge_attr[:, :nn.dim])
        y = nn.aggregate(phi, index=edge_index[1, :], ptr=None, dim_size=x.size()[0])
    else:
        y = torch.zeros(size=(x.shape[0], nn.out_channels), dtype=x.dtype, device=x.device)

    if hasattr(nn, "root_weight") and nn.root_weight:
        nn.lin_act = nn.lin(x)
        y[mask] += nn.lin_act[mask]

    if hasattr(nn, "bias") and nn.bias is not None:
        y[mask] += nn.bias

    return y

def __graph_initialization(module, data, *args, **kwargs):
    module.asy_graph = data.clone()
    module.graph_out = data.clone()

    # Concat old and updated feature for output feature vector.
    if hasattr(module.asy_graph, "active_clusters"):
        mask = module.asy_graph.active_clusters
        num_updated_elements = len(mask)
    else:
        mask = slice(None)
        num_updated_elements = len(data.x)

    module.graph_out.x = __conv(data.x, data.edge_index, data.edge_attr, mask, module)

    # If required, compute the flops of the asynchronous update operation. Therefore, sum the flops for each node
    # update, as they highly depend on the number of neighbors of this node.
    if module.asy_flops_log is not None:
        flops = compute_flops_conv(module, num_times_apply_bias_and_root=num_updated_elements, num_edges=data.edge_index.shape[1])
        module.asy_flops_log.append(flops)

    if hasattr(module, "to_dense"):
        mask = module.graph_out.active_clusters
        batch = module.graph_out.batch if module.graph_out.batch is None else module.graph_out.batch[mask]
        if batch is None:
            batch = torch.zeros(len(module.graph_out.pos[mask]), dtype=torch.long, device=data.x.device)
        return module.to_dense(module.graph_out.x[mask],
                               module.graph_out.pos[mask],
                               module.graph_out.pooling,
                               batch)

    return module.graph_out.clone()

def __edges_with_src_node(node_idx, edge_index, edge_attr=None, node_idx_type="src", return_changed_edges=False, return_mask=False):
    if node_idx.numel() == 0:
        outputs = [torch.empty(size=(2,0), dtype=torch.long, device=node_idx.device)]
        if edge_attr is not None:
            outputs.append(torch.empty(size=(0,3), dtype=edge_attr.dtype, device=edge_attr.device))
        if return_mask:
            outputs.append(torch.empty(size=(0,), dtype=torch.bool, device=node_idx.device))
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    if node_idx_type == "src":
        mask = torch_isin(edge_index[0], node_idx)
    elif node_idx_type == "dst":
        mask = torch_isin(edge_index[1], node_idx)
    elif node_idx_type == "both":
        mask = torch_isin(edge_index[0], node_idx) | torch_isin(edge_index[1], node_idx)
    else:
        raise ValueError

    output = [edge_index[:,mask]]
    if edge_attr is not None:
        output.append(edge_attr[mask])
    if return_changed_edges:
        output.append(mask.nonzero().ravel())
    if return_mask:
        output.append(mask.nonzero().ravel())
    if len(output) == 1:
        output = output[0]
    return output

def find_only_x(idx_new_comp, idx_diff, pos_idx_diff, edge):
    return idx_new_comp[torch_isin(idx_new_comp, idx_diff) & ~torch_isin(idx_new_comp, pos_idx_diff) & ~torch_isin(idx_new_comp, edge)]

def __graph_processing(module, data, *args, **kwargs):
    """Asynchronous graph update for graph convolutional layer.

    After the initialization of the graph, only the nodes (and their receptive field) have to updated which either
    have changed (different features) or have been added. Therefore, for updating the graph we have to first
    compute the set of "diff" and "new" nodes to then do the convolutional message passing on this subgraph,
    and add the resulting residuals to the graph.

    :param x: graph nodes features.
    """
    num_edges_image_feat = 0
    num_edges = 0
    num_times_apply_bias_and_root = 0
    new_nodes = len(data.x) > len(module.asy_graph.x)

    # first update the input graph
    if new_nodes:
        idx_new = graph_new_nodes(module.asy_graph, data)

        module.asy_graph.x = torch.cat([module.asy_graph.x, data.x[idx_new]])
        idx_new_comp = idx_new

        # when new edges are added through added events, make sure to add them, otherwise only update the edge attributes
        module.asy_graph.edge_index = torch.cat([module.asy_graph.edge_index, data.edge_index], dim=-1)
        module.asy_graph.edge_attr = torch.cat([module.asy_graph.edge_attr, data.edge_attr], dim=0)

        zero_row = torch.zeros(len(idx_new), module.out_channels, device=data.x.device)
        module.graph_out.x = torch.cat([module.graph_out.x, zero_row])

        data.diff_idx = idx_new_comp
        pos_idx_diff = torch.zeros(size=(0,), dtype=torch.long, device=data.x.device)

        if idx_new_comp.numel() > 0:
            edge_index_new, edge_attr_new = data.edge_index, data.edge_attr
            num_edges += edge_index_new.shape[1]
    else:
        idx_diff, pos_idx_diff = graph_changed_nodes(module.asy_graph, data)
        idx_new_comp = _efficient_cat_unique([pos_idx_diff, idx_diff, data.edge_index[1].unique()])
        data.diff_idx = idx_new_comp

        if idx_new_comp.numel() > 0:
            # find out dests of idx new, idx diff and pos_idx_diff
            edge_index_update_message, mask  = __edges_with_src_node(idx_new_comp, module.asy_graph.edge_index, return_mask=True)
            edge_attr_update_message = module.asy_graph.edge_attr[mask]
            num_edges += edge_index_update_message.shape[1]
            if hasattr(module.asy_graph, "active_clusters") and hasattr(data, "_changed_attr"):
                module.asy_graph.edge_attr[data._changed_attr_indices] = data._changed_attr
                edge_attr_update_message_new = module.asy_graph.edge_attr[mask]
            else:
                edge_attr_update_message_new = edge_attr_update_message

        # when new edges are added through added events, make sure to add them, otherwise only update the edge attributes
        if data.edge_index.numel() > 0:
            module.asy_graph.edge_index = torch.cat([module.asy_graph.edge_index, data.edge_index], dim=-1)
            module.asy_graph.edge_attr = torch.cat([module.asy_graph.edge_attr, data.edge_attr], dim=0)

        if idx_new_comp.numel() > 0 and edge_index_update_message.numel() > 0:
            # first compute update to y
            x_old = module.asy_graph.x[edge_index_update_message[0], :]
            phi_old = module.message(x_old, edge_attr=edge_attr_update_message)

            # new messages
            x_new = data.x[edge_index_update_message[0], :]
            phi_new = module.message(x_new, edge_attr=edge_attr_update_message_new)
            scatter_sum(phi_new-phi_old, index=edge_index_update_message[1],out=module.graph_out.x, dim=0, dim_size=len(module.graph_out.x))

            data.diff_idx = _efficient_cat_unique([data.diff_idx, edge_index_update_message[1]])
            num_edges += edge_index_update_message.shape[1]

        only_x = find_only_x(idx_new_comp, idx_diff, pos_idx_diff, data.edge_index[1])
        if only_x is not None and len(only_x) > 0:
            idx_new_comp = idx_new_comp[~torch_isin(idx_new_comp, only_x)]
            generalized_lin(module, data.x - module.asy_graph.x, module.graph_out.x, only_x)
            num_times_apply_bias_and_root += len(only_x)

        if idx_new_comp.numel() > 0:

            # edge and attrs for newly computed
            edge_index_new, edge_attr_new = __edges_with_src_node(idx_new_comp, edge_index=module.asy_graph.edge_index,
                                                                  edge_attr=module.asy_graph.edge_attr,
                                                                  node_idx_type="dst")

            edge_index_pos, _ = __edges_with_src_node(pos_idx_diff, edge_index=module.asy_graph.edge_index,
                                                      edge_attr=module.asy_graph.edge_attr,
                                                      node_idx_type="dst")
            num_edges_image_feat = edge_index_pos.shape[1]

            num_edges += edge_index_new.shape[1]
            module.graph_out.x[idx_new_comp] = 0

    if idx_new_comp.numel() > 0:
        if edge_index_new.shape[1] > 0:
            num_edges += edge_index_new.shape[1]
            # next compute all messages for computing new index
            x_j = data.x[edge_index_new[0, :], :]
            phi = module.message(x_j, edge_attr=edge_attr_new[:,:module.dim])
            scatter_sum(phi, out=module.graph_out.x, index=edge_index_new[1], dim=0, dim_size=len(module.graph_out.x))

        num_times_apply_bias_and_root += len(idx_new_comp)
        generalized_lin(module, data.x, module.graph_out.x, idx_new_comp)

    data.x = module.graph_out.x
    data.diff_pos_idx = pos_idx_diff

    # If required, compute the flops of the asynchronous update operation. Therefore, sum the flops for each node
    # update, as they highly depend on the number of neighbors of this node.
    if module.asy_flops_log is not None:
        cat = hasattr(data, "skipped") and data.skipped
        data.skipped = False
        flops = compute_flops_conv(module, num_times_apply_bias_and_root=len(idx_new_comp), num_edges=num_edges,
                                   concatenation=cat, num_image_channels=getattr(data, "num_image_channels", -1))

        if cat:
            flops += compute_flops_cat(module, num_edges=num_edges_image_feat,
                                       num_times_apply_bias_and_root=num_times_apply_bias_and_root, num_image_channels=getattr(data, "num_image_channels", -1))


        module.asy_flops_log.append(flops)

    if hasattr(module, "to_dense"):
        if pos_idx_diff.numel() > 0 or idx_new_comp.numel() > 0:
            mask = data.active_clusters
            batch = data.batch if data.batch is None else data.batch[mask]
            if batch is None:
                batch = torch.zeros(len(module.graph_out.pos[mask]), dtype=torch.long, device=data.x.device)

            return module.to_dense(data.x[mask],
                                   data.pos[mask],
                                   data.pooling,
                                   batch)
        else:
            return module.dense[:1]

    return data

def generalized_lin(module, input, output, idx):
    uses_bias = hasattr(module, "bias") and module.bias is not None
    uses_weight = hasattr(module, "root_weight") and module.root_weight
    if not uses_weight:
        return

    if uses_bias:
        asy_tools.masked_lin(idx, input, output, module.lin.weight.data, module.bias.data, True)
    else:
        asy_tools.masked_lin_no_bias(idx, input, output, module.lin.weight.data, True)

def __check_support(module) -> bool:
    if isinstance(module, torch_geometric.nn.conv.GCNConv):
        if module.normalize is True:
            raise NotImplementedError("GCNConvs with normalization are not yet supported!")
    return True


def make_conv_asynchronous(module, log_flops: bool = False):
    """Module converter from synchronous to asynchronous & sparse processing for graph convolutional layers.
    By overwriting parts of the module asynchronous processing can be enabled without the need of re-learning
    and moving its weights and configuration. So, a convolutional layer can be converted by, for example:

    ```
    module = GCNConv(1, 2)
    module = make_conv_asynchronous(module)
    ```

    :param module: convolutional module to transform.
    :param r: update radius around new events.
    :param edge_attributes: function for computing edge attributes (default = None).
    :param is_initial: layer initial layer of sequential or deeper (default = False).
    :param log_flops: log flops of asynchronous update.
    """
    assert __check_support(module)

    module = add_async_graph(module, log_flops=log_flops)
    return make_asynchronous(module, __graph_initialization, __graph_processing)
