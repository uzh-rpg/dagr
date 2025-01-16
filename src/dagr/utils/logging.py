import torch
import wandb
import os

from typing import List, Dict, Optional
from torch_geometric.data import Batch
from pathlib import PosixPath
from pprint import pprint
from pathlib import Path

from torch_geometric.data import Data


def set_up_logging_directory(dataset, task, output_directory):
    project = f"low_latency-{dataset}-{task}"

    output_directory = output_directory / dataset / task
    output_directory.mkdir(parents=True, exist_ok=True)
    wandb.init(project=project, entity="rpg", save_code=True, dir=str(output_directory))

    name = wandb.run.name
    output_directory = output_directory / name
    output_directory.mkdir(parents=True, exist_ok=True)
    os.system(f"cp -r {os.path.join(os.path.dirname(__file__), '../../low_latency_object_detection')} {str(output_directory)}")

    return output_directory

def log_hparams(args):
    hparams = {k: str(v) if type(v) is PosixPath else v for k, v in vars(args).items()}
    pprint(hparams)
    wandb.log(hparams)

def log_bboxes(data: Batch,
               targets: List[Dict[str, torch.Tensor]],
               detections: List[Dict[str, torch.Tensor]],
               class_names: List[str],
               bidx: int,
               key: str):

    gt_bbox = []
    det_bbox = []
    images = []
    for b, datum in enumerate(data.to_data_list()):
        image = visualize_events(datum)
        image = torch.cat([image, image], dim=1)
        images.append(image)

        if len(detections) > 0:
            det = detections[b]
            det = torch.cat([det['boxes'], det['labels'].view(-1,1), det['scores'].view(-1,1)], dim=-1)
            det[:, [0, 2]] += b * datum.width
            det_bbox.append(det)

        if len(targets) > 0:
            tar = targets[b]
            tar = torch.cat([tar['boxes'], tar['labels'].view(-1, 1), torch.ones_like(tar['labels'].view(-1, 1))], dim=-1)
            tar[:, [0, 2]] += b * datum.width
            tar[:, [1, 3]] += datum.height
            gt_bbox.append(tar)

        if b == bidx-1:
            break

    pred_bbox = torch.cat(det_bbox)
    gt_bbox = torch.cat(gt_bbox)
    images = torch.cat(images, dim=-1)

    bidx = min([bidx, len(data)])

    gt_bbox[:,[0,2]] /= (bidx * datum.width)
    gt_bbox[:,[1,3]] /= (2 * datum.height)

    pred_bbox[:,[0,2]] /= (bidx * datum.width)
    pred_bbox[:,[1,3]] /= (2 * datum.height)

    image = __convert_to_wandb_data(images.detach().float().cpu(),
                                    gt_bbox.detach().cpu(),
                                    pred_bbox.detach().cpu(),
                                    class_names)

    wandb.log({key: image})

def visualize_events(data: Data)->torch.Tensor:
    x, y = data.pos[:,:2].long().t()
    p = data.x[:,0].long()

    if hasattr(data, "image"):
        image = data.image[0].clone()
    else:
        image = torch.full(size=(3, data.height, data.width), fill_value=255, device=p.device, dtype=torch.uint8)

    is_pos = p == 1
    image[:, y[is_pos], x[is_pos]] = torch.tensor([[0],[0],[255]], dtype=torch.uint8, device=p.device)
    image[:, y[~is_pos], x[~is_pos]] = torch.tensor([[255],[0],[0]], dtype=torch.uint8, device=p.device)

    return image

def __convert_to_wandb_data(image: torch.Tensor, gt: torch.Tensor, p: torch.Tensor, class_names: List[str])->wandb.Image:
    return wandb.Image(image, boxes={
        "predictions": __parse_bboxes(p, class_names, suffix="P"),
        "ground_truth": __parse_bboxes(gt, class_names)
    })

def __parse_bboxes(bboxes: torch.Tensor, class_names: List[str], suffix: str="GT"):
    # bbox N x 6 -> xyxycs
    return {
        "box_data": [__parse_bbox(bbox, class_names, suffix) for bbox in bboxes],
        "class_labels": dict(enumerate(class_names))
    }

def __parse_bbox(bbox: torch.Tensor, class_names: List[str], suffix: str="GT"):
    # bbox xyxycs
    return {
        "position": {
            "minX": float(bbox[0]),
            "minY": float(bbox[1]),
            "maxX": float(bbox[2]),
            "maxY": float(bbox[3])
        },
        "class_id": int(bbox[-2]),
        "scores": {
            "object score": float(bbox[-1])
        },
        "bbox_caption": f"{suffix} - {class_names[int(bbox[-2])]}"
    }


