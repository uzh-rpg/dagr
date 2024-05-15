import logging
import torch

from torch_geometric.data import Data
from torch_scatter import scatter_max, scatter_sum

from .base.base import add_async_graph, make_asynchronous
from .base.utils import graph_changed_nodes, graph_new_nodes, _efficient_cat_unique, torch_isin, _efficient_cat
from .conv import __edges_with_src_node
from .base.utils import _to_hom, _from_hom, __remove_duplicate_from_A


def pool_edge(cluster, edge_index, self_loop):
    edge_index = cluster[edge_index]
    if self_loop:
        edge_index = edge_index.unique(dim=-1)
    else:
        edge_index = edge_index[:,edge_index[0]!=edge_index[1]].unique(dim=-1)

    if len(edge_index) > 0:
        return edge_index
    return torch.zeros((2,0), dtype=torch.long, device=cluster.device)


def compute_attrs(transform, edge_index, pos):
    return (pos[edge_index[0]] - pos[edge_index[1]])  /  (2 * transform.max) + 0.5


def __dense_process(module, data: Data, *args, **kwargs) -> Data:
    # compute the cache to compute the output graph. This contains
    # 1. the cluster assignment for each input feature -> dim num_input_nodes
    # 2. the sum of positions for each feature in each cluster -> max_num_clusters
    # 3. the count of positions for each feature -> max_num_clusters
    # 4. which input nodes went to the computation of which output_node -> max_num_clusters x num_output
    cluster_index = __get_global_cluster_index(module, pos=data.pos[:,:module.dim])
    x, pos = data.x, data.pos
    edge_index = pool_edge(cluster_index, data.edge_index, module.self_loop)

    if hasattr(module.asy_graph, "active_clusters"):
        active_cluster_index = cluster_index[module.asy_graph.active_clusters]
        new_cluster_index = torch.full_like(cluster_index, fill_value=-1)
        new_cluster_index[module.asy_graph.active_clusters] = active_cluster_index
        cluster_index = new_cluster_index
        x = x[module.asy_graph.active_clusters]
        pos = pos[module.asy_graph.active_clusters]
    else:
        active_cluster_index = cluster_index

    pos_hom = scatter_sum(_to_hom(pos[:,:module.dim]), active_cluster_index, dim=0, dim_size=module.num_grid_cells)
    output_pos = _from_hom(pos_hom)

    module.wh_inv = 1/ torch.Tensor([data.width[0], data.height[0]]).to(output_pos.device).view(1,-1)
    output_pos[:,:2] = module.round_to_pixel(output_pos[:,:2], wh_inv=module.wh_inv)

    active_clusters = torch.unique(active_cluster_index)

    cache = Data(cluster_index=cluster_index, pos_hom=pos_hom)

    if module.aggr == 'max':
        output_x = torch.full(size=(module.num_grid_cells, x.shape[1]), fill_value=-torch.inf, device=x.device)
        _, output_argmax = scatter_max(x, active_cluster_index, dim=0, out=output_x, dim_size=module.num_grid_cells)
        cache.output_argmax = output_argmax
    else:
        x_hom = _to_hom(x)
        cache.output_x_hom = scatter_sum(x_hom, active_cluster_index, dim=0, dim_size=module.num_grid_cells)
        output_x = _from_hom(cache.output_x_hom)

    module.ones = torch.ones_like(output_x[:,:1])

    # construct output. This contains:
    # the output graph -> has num_unique_clusters nodes
    if module.keep_temporal_ordering:
        t = pos[:, -1] if pos.shape[-1] > 2 else data.t_max[active_cluster_index]
        output_t = torch.full(size=(module.num_grid_cells,), fill_value=-torch.inf, device=x.device)
        t_max, _ = scatter_max(t, active_cluster_index, dim=0, out=output_t, dim_size=module.num_grid_cells)
        if edge_index.shape[1] > 0:
            t_src, t_dst = t_max[edge_index]
            edge_index = edge_index[:, t_dst > t_src]

    output_graph = Data(x=output_x,
                        pos=output_pos,
                        edge_index=edge_index,
                        active_clusters=active_clusters,
                        width=data.width,
                        height=data.height)

    if module.keep_temporal_ordering:
        output_graph.t_max = output_t

    if module.transform is not None:
        output_graph = module.transform(output_graph)

    return output_graph, cache

def __graph_initialization(module, data: Data, *args, **kwargs) -> Data:
    """Graph initialization for asynchronous update.

    Both the input as well as the output graph have to be stored, in order to avoid repeated computation. The
    input graph is used for spotting changed or new nodes (as for other asyn. layers), while the output graph
    is compared to the set of diff & new nodes, in order to be updated. Depending on the type of pooling (max, mean,
    average, etc) not only the output voxel feature have to be stored but also aggregations over all nodes in
    one output voxel such as the sum or count.

    Next to the features the node positions are averaged over all nodes in the voxel, as well. To do so,
    position aggregations (count, sum) are stored and updated, too.
    """
    module.asy_graph = data.clone()
    module.graph_out, module.cache  = __dense_process(module, data)
    module.graph_out.pooling = module.voxel_size

    logging.debug(f"Resulting in coarse graph {module.graph_out}")

    # Compute number of floating point operations (no cat, flatten, etc.).
    if module.asy_flops_log is not None:
        unique_clusters = len(module.graph_out.active_clusters)
        flops = 6 * unique_clusters # pos and scatter with index
        flops += module.graph_out.x.shape[1] * unique_clusters + module.graph_out.edge_index.numel()  # every edge has to be re-assigned
        module.asy_flops_log.append(flops)

    return module.graph_out.clone()

#@profile
def __graph_process(module, data, *args, **kwargs) -> Data:
    new_nodes = len(data.x) > len(module.asy_graph.x)

    if new_nodes:
        new_idx = graph_new_nodes(module.asy_graph, data)

        module.asy_graph.x = torch.cat([module.asy_graph.x, data.x[new_idx]])
        module.asy_graph.pos = torch.cat([module.asy_graph.pos, data.pos[new_idx]])

        new_cluster_idx = __get_global_cluster_index(module, data.pos[new_idx, :module.dim])

        # add to active clusters
        if new_idx.numel() > 0:
            module.graph_out.active_clusters = torch.cat([new_cluster_idx, module.graph_out.active_clusters]).sort().values.unique()

        module.cache.cluster_index = torch.cat([module.cache.cluster_index, new_cluster_idx])
        diff_pos_idx = new_idx
        new_pos_hom = _to_hom(data.pos[new_idx, :module.dim], module.ones)
        recomp_pos_new = new_cluster_idx
        recomp_x_new = new_cluster_idx
        if recomp_x_new.numel() > 0:
            recomp_x_new = recomp_x_new#.clone()

        num_diff_x = 0#len(diff_idx)
        num_new = len(new_idx)
        scatter_sum(new_pos_hom, new_cluster_idx, out=module.cache.pos_hom, dim=0)

        if recomp_x_new.numel() > 0:
            if module.aggr == "max":
                mask = torch.cat([module.cache.output_argmax[recomp_x_new].ravel(), new_idx]).unique()
            else:
                mask = torch_isin(module.cache.cluster_index, recomp_x_new)

    else:
        num_new = 0

        diff_idx, diff_pos_idx = graph_changed_nodes(module.asy_graph, data)
        num_diff_x = len(diff_idx)

        recomp_x_new = None
        recomp_pos_new = None

        if diff_pos_idx.numel()> 0:
            inactive = torch_isin(diff_pos_idx, module.asy_graph.active_clusters)
            old_pos = module.asy_graph.pos[diff_pos_idx[inactive], :module.dim]
            module.asy_graph.pos[diff_pos_idx] = data.pos[diff_pos_idx]

            old_pos_hom = _to_hom(old_pos, module.ones)
            old_cluster_idx_pos = __get_global_cluster_index(module, old_pos)
            new_pos_hom = _to_hom(data.pos[diff_pos_idx, :module.dim], module.ones)
            all_pos = torch.cat([-old_pos_hom, new_pos_hom])
            new_cluster_idx_pos = __get_global_cluster_index(module, data.pos[diff_pos_idx, :module.dim])
            module.cache.cluster_index[diff_pos_idx] = new_cluster_idx_pos

            recomp_x_new = new_cluster_idx_pos
            recomp_pos_new = _efficient_cat([old_cluster_idx_pos, new_cluster_idx_pos])
            # todo stupid bug, shallow copy could occur
            if recomp_pos_new.numel()>0 and recomp_pos_new.data_ptr() == recomp_x_new.data_ptr():
                recomp_pos_new = recomp_pos_new#.clone()
            scatter_sum(all_pos, recomp_pos_new, out=module.cache.pos_hom, dim=0)

        if diff_idx.numel() > 0:
            cluster_idx_x = __get_global_cluster_index(module, module.asy_graph.pos[diff_idx, :module.dim])
            recomp_x_new = cluster_idx_x if recomp_x_new is None else _efficient_cat_unique([recomp_x_new, cluster_idx_x])

        if recomp_x_new is not None and recomp_x_new.numel() > 0:
            mask = torch_isin(module.cache.cluster_index, recomp_x_new)
            if module.aggr == "max":
                module.graph_out.x[recomp_x_new] = -torch.inf

    if recomp_x_new is not None and recomp_x_new.numel() > 0:
        if module.aggr == "max":
            scatter_max(data.x[mask], module.cache.cluster_index[mask], out=module.graph_out.x, dim=0)
        else:
            delta_x_hom = _to_hom(data.x[mask], module.ones) #
            valid = ~torch.isinf(module.asy_graph.x[mask][:,0])
            delta_x_hom[valid] -= _to_hom(module.asy_graph.x[mask][valid], module.ones)
            scatter_sum(delta_x_hom, module.cache.cluster_index[mask], out=module.cache.output_x_hom, dim=0)
            module.graph_out.x[recomp_x_new] = _from_hom(module.cache.output_x_hom[recomp_x_new])

    # find the edges which are associated with changed positions since these need their attrs updated
    # however, here we can only look at the x,y values. If only the third attr changes, then we don't need to do anything
    if recomp_pos_new is not None and recomp_pos_new.numel() > 0:
        # update pos with the updated positions
        new_pos = _from_hom(module.cache.pos_hom[recomp_pos_new])
        new_pos[:,:2] = module.round_to_pixel(new_pos[:,:2], wh_inv=module.wh_inv)
        module.graph_out.pos[recomp_pos_new,:module.dim] = new_pos
        update_edge_index, changed_edges = __edges_with_src_node(recomp_pos_new, module.graph_out.edge_index, node_idx_type="both", return_changed_edges=True)
        if module.transform is not None:
            module.graph_out._changed_attr = compute_attrs(module.transform, update_edge_index, module.graph_out.pos)
            module.graph_out._changed_attr_indices = changed_edges

    # also handle edges which come from new connections at the input. These first need to be pooled
    # then check if they are actually new.
    if data.edge_index.numel() > 0:
        coarse_edge_index = pool_edge(module.cache.cluster_index, data.edge_index, module.self_loop)
        module.graph_out.edge_index = __remove_duplicate_from_A(coarse_edge_index, module.graph_out.edge_index)
    else:
        module.graph_out.edge_index = data.edge_index#torch.empty((2, 0), dtype=torch.long, device=data.x.device)

    if module.transform is not None:
        if module.graph_out.edge_index.numel() > 0:
            module.graph_out.edge_attr = compute_attrs(module.transform, module.graph_out.edge_index, module.graph_out.pos)
        else:
            module.graph_out.edge_attr = data.edge_attr

    module.graph_out.diff_idx = recomp_x_new.unique() if recomp_x_new is not None else diff_idx
    module.graph_out.diff_pos_idx = recomp_pos_new.unique() if recomp_pos_new is not None else diff_pos_idx

    if module.asy_flops_log is not None:
        num_recomp_x = 0 if recomp_x_new is None else len(recomp_x_new)
        num_recomp_pos = 0 if recomp_pos_new is None else len(recomp_pos_new)
        flops = 0
        flops += num_recomp_x * module.graph_out.x.shape[1]  # perform max
        flops += num_recomp_pos                       # recompute pos
        flops += 4 * len(diff_pos_idx)                        # subtract and add pos twice
        flops += len(diff_pos_idx) + num_diff_x          # get cluster center for each index
        flops += num_new * 2                             # add twice, also compute cluster center
        module.asy_flops_log.append(flops)

    return module.graph_out

def __get_global_cluster_index(module, pos) -> torch.LongTensor:
    n_pos_dim = 2#pos.shape[1]
    voxel_size = module.voxel_size[:n_pos_dim]#, device=pos.device)
    pos_vertex = (pos[:,:2] / voxel_size).long()
    x_v, y_v = pos_vertex.t()
    grid_size = (1 / voxel_size + 1e-3).long()
    cluster_idx = x_v + grid_size[0] * y_v
    return cluster_idx


def make_max_pool_asynchronous(module, log_flops: bool = False):
    """Module converter from synchronous to asynchronous & sparse processing for graph max pooling layer.
    By overwriting parts of the module asynchronous processing can be enabled without the need re-creating the
    object. So, a max pooling layer can be converted by, for example:

    ```
    module = MaxPool([4, 4])
    module = make_max_pool_asynchronous(module)
    ```

    :param module: standard max pooling module.
    :param grid_size: grid size (grid starting at 0, spanning to `grid_size`), >= `size`.
    :param r: update radius around new events.
    :param log_flops: log flops of asynchronous update.
    """

    module = add_async_graph(module, log_flops=log_flops)
    module = make_asynchronous(module, __graph_initialization, __graph_process)
    return module
