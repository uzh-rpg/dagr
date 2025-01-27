import torch
import wandb
import os

from typing import List, Dict, Optional
from torch_geometric.data import Batch
from pathlib import PosixPath
from pprint import pprint
from pathlib import Path

from torch_geometric.data import Data


class Checkpointer:
    def __init__(self, output_directory: Optional[Path] = None, args=None, optimizer=None, scheduler=None, ema=None, model=None):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.model = model

        self.mAP_max = 0
        self.output_directory = output_directory
        self.args = args

    def restore_if_existing(self, folder, resume_from_best=False):
        checkpoint = self.search_for_checkpoint(folder, best=resume_from_best)
        if checkpoint is not None:
            print(f"Found existing checkpoint at {checkpoint}, resuming...")
            self.restore_checkpoint(folder, best=resume_from_best)

    def mAP_from_checkpoint_name(self, checkpoint_name: Path):
        return float(str(checkpoint_name).split("_")[-1].split(".pth")[0])

    def search_for_checkpoint(self, resume_checkpoint: Path, best=False):
        checkpoints = list(resume_checkpoint.glob("*.pth"))
        if len(checkpoints) == 0:
            return None

        if not best:
            if resume_checkpoint / "last_model.pth" in checkpoints:
                return resume_checkpoint / "last_model.pth"

        # remove "last_model.pth" from checkpoints
        if resume_checkpoint / "last_model.pth" in checkpoints:
            checkpoints.remove(resume_checkpoint / "last_model.pth")

        checkpoints = sorted(checkpoints, key=lambda x: self.mAP_from_checkpoint_name(x.name))
        return checkpoints[-1]


    def restore_if_not_none(self, target, source):
        if target is not None:
            target.load_state_dict(source)

    def restore_checkpoint(self, checkpoint_directory, best=False):
        path = self.search_for_checkpoint(checkpoint_directory, best)
        assert path is not None, "No checkpoint found in {}".format(checkpoint_directory)
        print("Restoring checkpoint from {}".format(path))
        checkpoint = torch.load(path)

        checkpoint['model'] = self.fix_checkpoint(checkpoint['model'])
        checkpoint['ema'] = self.fix_checkpoint(checkpoint['ema'])

        if self.ema is not None:
            self.ema.ema.load_state_dict(checkpoint.get('ema', checkpoint['model']))
            self.ema.updates = checkpoint.get('ema_updates', 0)
        self.restore_if_not_none(self.model, checkpoint['model'])
        self.restore_if_not_none(self.optimizer, checkpoint['optimizer'])
        self.restore_if_not_none(self.scheduler, checkpoint['scheduler'])
        return checkpoint['epoch']

    def fix_checkpoint(self, state_dict):
        return state_dict

    def checkpoint(self, epoch: int, name: str=""):
        self.output_directory.mkdir(exist_ok=True, parents=True)

        checkpoint = {
            "ema": self.ema.ema.state_dict(),
            "ema_updates": self.ema.updates,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "args": self.args
        }

        torch.save(checkpoint, self.output_directory / f"{name}.pth")

    def process(self, data: Dict[str, float], epoch: int):
        mAP = data['mAP']
        data = {f"validation/metric/{k}": v for k, v in data.items()}
        data['epoch'] = epoch
        wandb.log(data)

        if mAP > self.mAP_max:
            self.checkpoint(epoch, name=f"best_model_mAP_{mAP}")
            self.mAP_max = mAP


def set_up_logging_directory(dataset, task, output_directory, exp_name="temp"):
    project = f"low_latency-{dataset}-{task}"

    output_directory = output_directory / dataset / task
    output_directory.mkdir(parents=True, exist_ok=True)
    wandb.init(project=project, id=exp_name, entity="danielgehrig18", save_code=True, dir=str(output_directory))

    name = wandb.run.id
    output_directory = output_directory / name
    output_directory.mkdir(parents=True, exist_ok=True)

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


