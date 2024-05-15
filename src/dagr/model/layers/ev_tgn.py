import torch

from torch_geometric.data import Batch, Data
from dagr.graph.ev_graph import SlidingWindowGraph


def _get_value_as_int(obj, key):
    val = getattr(obj, key)
    return val if type(val) is int else val[0]

def denormalize_pos(events):
    if hasattr(events, "pos_denorm"):
        return events.pos_denorm

    denorm = torch.tensor([int(events.width[0]), int(events.height[0]), int(events.time_window[0])], device=events.pos.device)
    return (denorm.view(1,-1) * events.pos + 1e-3).int()


class EV_TGN(torch.nn.Module):
    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.radius = args.radius
        self.max_neighbors = args.max_neighbors
        self.max_queue_size = 128
        self.graph_creators = None

    def init_graph_creator(self, data):
        delta_t_us = int(self.radius * _get_value_as_int(data, "time_window"))
        radius = int(self.radius * _get_value_as_int(data, "width")+1)
        batch_size = data.num_graphs
        width = int(_get_value_as_int(data, "width"))
        height = int(_get_value_as_int(data, "height"))
        self.graph_creators = SlidingWindowGraph(width=width, height=height,
                                                 max_num_neighbors=self.max_neighbors,
                                                 max_queue_size=self.max_queue_size,
                                                 batch_size=batch_size,
                                                 radius=radius, delta_t_us=delta_t_us)

    def forward(self, events: Data, reset=True):
        if events.batch is None:
            events = Batch.from_data_list([events])

        # before we start, are the new events used to generate the graph, or are the new nodes attached to the network?
        # if the first, then don't delete old events, if the second, delete as many events as are coming in.
        if self.graph_creators is None:
            self.init_graph_creator(events)
        else:
            if reset:
                self.graph_creators.reset()

        pos = denormalize_pos(events)
        #pos = torch.cat([events.batch.view(-1,1), pos, events.x.int()], dim=1).int()
        # properties of the edges
        # src_i <= dst_i
        # dst_i <= dst_j if i<j
        events.edge_index = self.graph_creators.forward(events.batch.int(), pos, delete_nodes=False, collect_edges=reset)
        events.edge_index = events.edge_index.long()

        return events