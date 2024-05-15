import torch
from .utils import _insert_events_into_queue, _search_for_edges


def move_to_cuda(func):
    def wrapper(self, x, *args, **kwargs):
        device = x.device
        on_cpu = device == "cpu"
        if on_cpu:
            x = x.to("cuda")
        ret = func(self, x, *args, **kwargs)
        if on_cpu:
            ret = ret.cpu()
        return ret
    return wrapper
        

class AsyncGraph:
    def __init__(self, width=640,
                 height=480,
                 batch_size=1,
                 max_num_neighbors=16,
                 max_queue_size=512,
                 radius=7, 
                 delta_t_us=600000):
        self.radius = radius
        self.delta_t_us = delta_t_us
        self.event_queue = None

        self.max_index = 0
        self.min_index = 0
        self.max_queue_size = max_queue_size
        self.max_num_neighbors = max_num_neighbors
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.device = None

        self.edges = torch.zeros((2,0), dtype=torch.long)
        self.all_timestamps = torch.zeros((0,), dtype=torch.int32)
        self.new_indices = None
        self.edge_buffer = None
        self.event_queue = None
    
    def initialize(self, n_ev, device):
        self.edges = torch.zeros((2,0), dtype=torch.long, device=device)
        self.all_timestamps = torch.zeros((0,), dtype=torch.int32, device=device)
        self.new_indices = torch.arange(n_ev, dtype=torch.int32, device=device)
        self.edge_buffer = torch.full((2, self.max_num_neighbors * n_ev), dtype=torch.int64, fill_value=-1, device=device)
        self.event_queue = torch.full((self.batch_size, self.max_queue_size, self.height, self.width), fill_value=-1, device=device, dtype=torch.int32)

    def reset(self):
        self.edges = torch.zeros((2,0), dtype=torch.long, device=self.device)
        self.all_timestamps = torch.zeros((0,), dtype=torch.int32, device=self.device)
        self.max_index = 0
        self.min_index = 0
        if self.edge_buffer is not None:
            self.edge_buffer.fill_(-1)
        if self.event_queue is not None:
            self.event_queue.fill_(-1)
        
    @move_to_cuda
    def forward(self, batch, pos, collect_edges=True):
        n_ev = len(batch)

        if self.device is None:
            self.device = batch.device
            self.initialize(n_ev, self.device)

        if len(batch) == 0:
            return torch.zeros((2,0), device=self.device, dtype=torch.int32)

        assert type(batch) is torch.Tensor and batch.dtype == torch.int32, [type(batch), batch.dtype]

        self.all_timestamps = torch.cat([self.all_timestamps, pos[:,2]])

        # insert events into queue, they have an ever growing index
        if n_ev > len(self.new_indices):
            self.new_indices = torch.arange(0, n_ev, dtype=torch.int32, device=self.device)
            self.edge_buffer = torch.full((2, self.max_num_neighbors * n_ev), dtype=torch.int64, fill_value=-1, device=self.device)

        indices = self.max_index + self.new_indices[:n_ev]
        self.max_index += n_ev

        self.event_queue = _insert_events_into_queue(batch, pos, indices=indices, queue=self.event_queue)

        # read out edges from event queue, they need to correspond to indices
        # from the current nodes
        self.edge_buffer.fill_(-1)
        edge_indices = _search_for_edges(batch, pos,
                                         all_timestamps=self.all_timestamps.contiguous(),
                                         indices=indices,
                                         queue=self.event_queue,
                                         max_num_neighbors=self.max_num_neighbors,
                                         radius=self.radius,
                                         delta_t_us=self.delta_t_us,
                                         edges=self.edge_buffer,
                                         min_index=self.min_index)
        
        if collect_edges:
            self.edges = torch.cat([self.edges, edge_indices], dim=-1)

        return edge_indices


class SlidingWindowGraph(AsyncGraph):
    def __init__(self, width=640,
                 height=480,
                 batch_size=1,
                 max_num_neighbors=16,
                 max_queue_size=1024,
                 radius=7, 
                 delta_t_us=600000):
        AsyncGraph.__init__(self, width, height, batch_size, max_num_neighbors, 
                            max_queue_size, radius, delta_t_us)

    @property
    def init(self):
        return len(self.all_timestamps) > 0

    def delete_nodes(self, n_delete, delete_edges=True, return_edges=True):
        # delete nodes
        self.all_timestamps = self.all_timestamps[n_delete:]
        self.min_index += n_delete

        # the current edges do not correspond to
        # the nodes anymore, so they need to be decremented
        if delete_edges:
            mask = (self.edges[0] < n_delete) | (self.edges[1] < n_delete)
            deleted_edges = self.edges[:,mask].clone()
            self.edges = self.edges[:,~mask]

        self.edges.add_(-n_delete)

        if delete_edges and return_edges:
            return deleted_edges
    
    @move_to_cuda
    def forward(self, batch, pos, return_node_counts=False, return_total_edges=False, delete_nodes=True, collect_edges=True):
        n_delete = len(batch) if self.init else 0

        # first find the interactions
        edges = AsyncGraph.forward(self, batch, pos, collect_edges=collect_edges)

        if return_total_edges:
            total_edges = self.edges.clone()
        
        if return_node_counts:
            tot_nodes = len(self.all_timestamps)

        ret = [edges]

        if delete_nodes:
            deleted_edges = self.delete_nodes(n_delete)
            ret.append(deleted_edges)

        if return_total_edges:
            ret.append(total_edges)

        if return_node_counts:
            ret.append([n_delete, len(batch), tot_nodes])

        if len(ret) == 1:
            ret = ret[0]

        return ret
