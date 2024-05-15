import torch

from torch_geometric.data import Batch, Data
from typing import List, Tuple
from collections import OrderedDict

from . import make_model_asynchronous, make_model_synchronous


def split_data(data: Data, index: int)->Tuple[Data, Data]:
    kwargs = dict(time_window=data.time_window, width=data.width, height=data.height)

    if hasattr(data, "image"):
        kwargs['image'] = data.image

    data1 = Data(pos=data.pos[:index], x=data.x[:index], **kwargs)
    data2 = Data(pos=data.pos[index:], x=data.x[index:], **kwargs)

    if hasattr(data, "pos_denorm"):
        data1.pos_denorm = data.pos_denorm[:index]
        data2.pos_denorm = data.pos_denorm[index:]

    return data1, data2

def forward_hook(inst, inp, out):
    inp = inp[0]

    if type(inp) is list:
        inp = inp[0].clone()
    elif type(inp) is tuple or type(inp) is dict:
        return
    else:
        inp = inp.clone()

    if type(out) is list:
        out = out[0].clone()
    elif type(out) is tuple or type(out) is dict:
        return
    else:
        out = out.clone()

    if not hasattr(inst, "activations"):
        inst.activations = []

    if type(inp) is torch.Tensor:
        inp = inp if len(inp.shape) == 2 else inp[0]
        inp = Data(x=inp)
    if type(out) is torch.Tensor:
        out = out if len(out.shape) == 2 else out[0]
        out = Data(x=out)

    if hasattr(inp, "active_clusters") and not hasattr(out, "active_clusters"):
        out.active_clusters = inp.active_clusters
    elif hasattr(out, "active_clusters") and not hasattr(inp, "active_clusters"):
        inp.active_clusters = out.active_clusters

    inp = _mask_if_possible(inp)
    out = _mask_if_possible(out)

    inst.activations.append((inp, out))

def _mask_if_possible(data):
    mask = slice(None, None, None)
    if hasattr(data,"active_clusters") and len(data.x) > data.active_clusters.max():
        mask = data.active_clusters
    masked = Data()
    if hasattr(data, "x"):
        masked.x = data.x[mask]
    if hasattr(data, "pos") and data.pos is not None:
        masked.pos = data.pos[mask]
    if hasattr(data, "edge_index"):
        masked.edge_index = data.edge_index
        masked.edge_attr = data.edge_attr
    return masked

def denorm(data):
    denorm = torch.tensor([int(data.width), int(data.height), int(data.time_window)], device=data.pos.device)
    data.pos_denorm = (denorm.view(1,-1) * data.pos + 1e-3).int()
    data.batch = data.batch.int()
    return data

def evaluate_flops(model: torch.nn.Module, batch: Data, dense=False,
                   check_consistency=False,
                   return_all_samples=False) -> OrderedDict:

    flops_per_layer_batch = []

    # for loop over batch
    for i, data in enumerate(batch.to_data_list()):
        events_initial, events_new = split_data(data, -1)

        events_initial = Batch.from_data_list([events_initial])
        events_new = Batch.from_data_list([events_new])
        data = Batch.from_data_list([data])

        # prepare data for fast inference
        data = denorm(data)
        events_new = denorm(events_new)
        events_initial = denorm(events_initial)

        # make a deep copy asynchronous version
        handles = []
        if check_consistency:
            for m in model.modules():
                handle = m.register_forward_hook(forward_hook)
                handles.append(handle)

            with torch.no_grad():
                model.forward(data, reset=True, return_targets=False)

        model = make_model_asynchronous(model, log_flops=True)

        try:
            with torch.no_grad():
                model.forward(events_initial, reset=True, return_targets=False)
                model.forward(events_new, reset=False, return_targets=False)

        except Exception as e:
            print(f"Crashed at index {i} with message {e}")
            raise e

        index = 0 if dense else 1
        flops_per_layer = OrderedDict(
            [
                (name, module.asy_flops_log[index]) for name, module in model.named_modules() \
                if hasattr(module, "asy_flops_log") and module.asy_flops_log is not None and len(
                module.asy_flops_log) > 0
            ]
        )

        flops_per_layer = _filter_non_leaf_nodes(flops_per_layer)
        flops_per_layer = _merge_to_level_flops(flops_per_layer, level=3)

        if not check_consistency:
            flops_per_layer_batch.append(flops_per_layer)

        model = make_model_synchronous(model)

        if check_consistency:
            # tests if outputs from 0th and 2nd run are equal
            max_mistake_x_layer, max_mistake_pos_layer, global_summary = test_and_compare_activations(model, runs=[0,2])
            if max_mistake_x_layer[0] > 1e-3 or max_mistake_pos_layer[1] > 1e-3:
                print(global_summary)
                print(f"AssertionError(Failed at index {i}.)")
            else:
                flops_per_layer_batch.append(flops_per_layer)
                print(global_summary)

            for handle in handles:
                handle.remove()
            for m in model.modules():
                if hasattr(m, "activations"):
                    del m.activations

    if len(flops_per_layer_batch) == 0:
        return None

    # global average
    flops_per_layer = _merge_list_flops(flops_per_layer_batch)

    output = {"flops_per_layer": flops_per_layer, "total_flops": sum(flops_per_layer.values())}
    if return_all_samples:
        output['flops_per_layer_batch'] = flops_per_layer_batch

    return output

def _filter_non_leaf_nodes(flops_per_layer: OrderedDict)->OrderedDict:
    filter_keys = []
    for q_name in flops_per_layer:
        for name in flops_per_layer:
            if q_name in name and q_name != name:
                filter_keys.append(q_name)
                break
    for f in filter_keys:
        flops_per_layer.pop(f)
    return flops_per_layer

def _merge_to_level_flops(flops_per_layer: OrderedDict, level=2)->OrderedDict:
    known_flops = []
    known_keys = []
    for name, flops in flops_per_layer.items():
        layers = name.split(".")
        layers_up_to_level = ".".join(layers[:level])
        if layers_up_to_level not in known_keys:
            known_keys.append(layers_up_to_level)
            known_flops.append(0)
        index = known_keys.index(layers_up_to_level)
        known_flops[index] += flops

    return OrderedDict(zip(known_keys, known_flops))

def _merge_list_flops(flops_per_layer_batch: List[OrderedDict])->OrderedDict:
    return OrderedDict([(key, sum([f[key] for f in flops_per_layer_batch]) / len(flops_per_layer_batch)) for key in flops_per_layer_batch[0]])

def _summary(est, gt, prefix):
    if len(est) != len(gt):
        return "\tCannot compare since x do not have same length\n", None
    max_diff, max_rel_diff, ind, max_ind = max_abs_diff(gt, est, threshold=1e-6)

    summary = f"\t{prefix} MAX DIFF: {max_diff} MAX REL DIFF: {max_rel_diff}\n"
    if ind.numel() > 0:
        summary += f"\t{prefix} IND: {max_ind.cpu().numpy().ravel().tolist()}\n"
    return summary, max_diff


def max_rel_diff(x, y, threshold=None):
    return error_above_threshold((x-y).abs() / (x.abs()+1e-6), threshold)

def error_above_threshold(error, mag, threshold):
    if threshold is None:
        return error.max()
    else:
        error_ravel = error.ravel()
        arg = error_ravel.argmax()
        return error_ravel[arg], error_ravel[arg] / mag.ravel()[arg], (error > threshold).nonzero()[:,0].unique(), error.max(-1).values.argmax()

def max_abs_diff(x, y, threshold=None, alpha=0):
    error = (x-y).abs()-x.abs()*alpha
    return error_above_threshold(error, x.abs(), threshold)

def _print_summary_for_one(target, estimate, prefix=""):
    max_diff_pos = None
    if type(target) is torch.Tensor:
        summary, max_diff_x = _summary(target, estimate, prefix)
    else:
        summary = ""
        if target.pos is not None and estimate.pos is not None:
            sub_summary, max_diff_pos = _summary(target.pos[:,:2], estimate.pos[:,:2], f"{prefix} POS")
            summary += sub_summary

        sub_summary, max_diff_x = _summary(target.x, estimate.x, prefix=f"{prefix} X")
        summary += sub_summary

    return summary, max_diff_x, max_diff_pos

def print_summary_of_module(activations, runs=[0,2]):
    target, estimate = [activations[i][1] for i in runs]
    return _print_summary_for_one(target, estimate, "OUT")

def test_and_compare_activations(model, runs=[0,2]):
    num_mistakes = []
    global_summary = ""
    for name, module in model.named_modules():
        if not hasattr(module, "activations"):
            continue
        else:
            if len(module.activations) <= max(runs):
                continue

            summary, max_diff_x, max_diff_pos = print_summary_of_module(module.activations, runs)
            if max_diff_x is not None and max_diff_pos is not None:
                num_mistakes.append([max_diff_x, max_diff_pos, name])
            global_summary += f"Inspecting {name}\n{summary}\n\n"

    max_mistake_x_layer = max(num_mistakes, key=lambda x: x[0])
    max_mistake_pos_layer = max(num_mistakes, key=lambda x: x[1])
    global_summary += f"Maximum mistakes: \n" \
                      f"\t{max_mistake_x_layer}\n" \
                      f"\t{max_mistake_pos_layer}"

    return max_mistake_x_layer, max_mistake_pos_layer, global_summary
