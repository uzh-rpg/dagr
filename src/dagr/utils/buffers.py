import numpy as np
import torch

from typing import List, Dict
from pathlib import Path

from .coco_eval import evaluate_detection


def diag_filter(bbox, height: int, width: int, min_box_diagonal: int = 30, min_box_side: int = 20):
    bbox[..., 0::2] = torch.clamp(bbox[..., 0::2], 0, width - 1)
    bbox[..., 1::2] = torch.clamp(bbox[..., 1::2], 0, height - 1)
    w, h = (bbox[..., 2:] - bbox[..., :2]).t()
    diag = torch.sqrt(w ** 2 + h ** 2)
    mask = (diag > min_box_diagonal) & (w > min_box_side) & (h > min_box_side)
    return mask


def filter_bboxes(detections: List[Dict[str, torch.Tensor]], height: int, width: int, min_box_diagonal: int = 30,
                  min_box_side: int = 20):
    filtered_bboxes = []
    for d in detections:
        bbox = d["boxes"]

        # first clamp boxes to image
        mask = diag_filter(bbox, height, width, min_box_diagonal, min_box_side)
        bbox = {k: v[mask] for k, v in d.items()}

        filtered_bboxes.append(bbox)

    return filtered_bboxes

def format_data(data, normalizer=None):
    if normalizer is None:
        normalizer = torch.stack([data.width[0], data.height[0], data.time_window[0]], dim=-1)

    if hasattr(data, "image"):
        data.image = data.image.float() / 255.0

    data.pos = torch.cat([data.pos, data.t.view((-1,1))], dim=-1)
    data.t = None
    data.x = data.x.float()
    data.pos = data.pos / normalizer
    return data

def bbox_t_to_ndarray(bbox, t):
    dtype = [('t', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1')]
    if len(bbox) == 3:
        dtype.append(('class_confidence', '<f4'))

    boxes = bbox['boxes'].numpy()
    labels = bbox['labels'].numpy()

    output = np.zeros(shape=(len(boxes),), dtype=dtype)
    output['t'] = t
    output['x'] = boxes[:, 0]
    output['y'] = boxes[:, 1]
    output['w'] = boxes[:, 2] - boxes[:, 0]
    output['h'] = boxes[:, 3] - boxes[:, 1]
    output['class_id'] = labels

    if len(bbox) == 3:
        output['class_confidence'] = bbox["scores"].numpy()

    return output


def compile(detections, sequences, timestamps):
    output = {}
    for det, s, t in zip(detections, sequences, timestamps):
        if s not in output:
            output[s] = []
        output[s].append(bbox_t_to_ndarray(det, t))

    if len(output) > 0:
        output = {k: np.concatenate(v) for k, v in output.items() if len(v) > 0}

    return output

def to_cpu(data_list: List[Dict[str, torch.Tensor]]):
    return [{k: v.cpu() for k, v in d.items()} for d in data_list]

class Buffer:
    def __init__(self):
        self.buffer = []

    def extend(self, elements: List[Dict[str, torch.Tensor]]):
        self.buffer.extend(to_cpu(elements))

    def clear(self):
        self.buffer.clear()

    def __iter__(self):
        return iter(self.buffer)

    def __next__(self):
        return next(self.buffer)



class DetectionBuffer:
    def __init__(self, height: int, width: int, classes: List[str]):
        self.height = height
        self.width = width
        self.classes = classes
        self.detections = Buffer()
        self.ground_truth = Buffer()

    def compile(self, sequences, timestamps):
        detections = compile(self.detections, sequences, timestamps)
        groundtruth = compile(self.ground_truth, sequences, timestamps)
        return detections, groundtruth

    def update(self, detections: List[Dict[str, torch.Tensor]], groundtruth: List[Dict[str, torch.Tensor]], dataset: str, height=None, width=None):
        self.detections.extend(detections)
        self.ground_truth.extend(groundtruth)

    def compute(self)->Dict[str, float]:
        output =  evaluate_detection(self.ground_truth.buffer, self.detections.buffer, height=self.height, width=self.width, classes=self.classes)
        output = {k.replace("AP", "mAP"): v for k, v in output.items()}
        self.detections.clear()
        self.ground_truth.clear()
        return output


class DictBuffer:
    def __init__(self):
        self.running_mean = None
        self.n = 0

    def __recursive_mean(self, mn: float, s: float):
        return self.n / (self.n + 1) * mn + s / (self.n + 1)

    def update(self, dictionary: Dict[str, float]):
        if self.running_mean is None:
            self.running_mean = {k: 0 for k in dictionary}

        self.running_mean = {k: self.__recursive_mean(self.running_mean[k], dictionary[k]) for k in dictionary}
        self.n += 1

    def save(self, path):
        torch.save(self.running_mean, path)

    def compute(self)->Dict[str, float]:
        return self.running_mean

