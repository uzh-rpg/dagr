import torch
import ev_graph_cuda
from typing import Union


def _insert_events_into_queue(batch, pos, indices, queue: torch.LongTensor):
    if len(batch) > 1:
        height, width = queue.shape[-2:]
        lin_coords = pos[:,0] + width * pos[:,1] + width*height*batch
        sorted_lin_coords, sort_index = torch.sort(lin_coords, stable=True, descending=False)
        sorted_indices = indices[sort_index].int()
        unique_coords, unique_counter = torch.unique_consecutive(sorted_lin_coords, return_counts=True)
        cumsum_counter = torch.cumsum(unique_counter, dim=0).int()
        queue = ev_graph_cuda.insert_in_queue_cuda(sorted_indices, unique_coords, cumsum_counter, queue)
    else:
        queue = ev_graph_cuda.insert_in_queue_single_cuda(indices, pos, queue)

    return queue

def _search_for_edges(batch, pos, all_timestamps, queue, indices, max_num_neighbors, radius, delta_t_us, edges, min_index):
    ev_graph_cuda.fill_edges_cuda(batch, pos, all_timestamps, queue, indices, max_num_neighbors, radius, delta_t_us, edges, min_index)
    edges = edges[:,(edges[1]>=0)]
    return edges
