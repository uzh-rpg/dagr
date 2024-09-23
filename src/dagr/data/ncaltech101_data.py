import numpy as np
import torch
import hdf5plugin
import h5py

from pathlib import Path
from typing import Optional, Callable
from torch.utils.data import Dataset
from torch_geometric.data import Data
from dagr.data.augment import init_transforms
from dagr.data.utils import to_data


class NCaltech101(Dataset):

    def __init__(self, root: Path, split, transform=Optional[Callable[[Data,], Data]], num_events: int=50000):
        super().__init__()
        self.load_dir = root / split
        self.classes = sorted([d.name for d in self.load_dir.glob("*")])
        self.num_classes = len(self.classes)
        self.files = sorted(list(self.load_dir.rglob("*.h5")))
        self.height = 180
        self.width = 240
        if transform is not None and hasattr(transform, "transforms"):
            init_transforms(transform.transforms, self.height, self.width)
        self.transform = transform
        self.time_window = 1000000
        self.num_events = num_events

    def __len__(self):
        return len(self.files)

    def preprocess(self, data):
        data.t -= (data.t[-1] - self.time_window + 1)
        return data

    def load_events(self, f_path):
        return _load_events(f_path, self.num_events)

    def __getitem__(self, idx):
        f_path = self.files[idx]
        target = self.classes.index(str(f_path.parent.name))

        events = self.load_events(f_path)
        data = to_data(**events,  bbox=self.load_bboxes(f_path, target),
                       t0=events['t'][0], t1=events['t'], width=self.width, height=self.height,
                       time_window=self.time_window)

        data = self.preprocess(data)

        data = self.transform(data) if self.transform is not None else data

        if not hasattr(data, "t"):
            data.t = data.pos[:, -1:]
            data.pos = data.pos[:, :2].type(torch.int16)

        return data

    def load_bboxes(self, raw_file: Path, class_id):
        rel_path = str(raw_file.relative_to(self.load_dir))
        rel_path = rel_path.replace("image_", "annotation_").replace(".h5", ".bin")
        annotation_file = self.load_dir / "../annotations" / rel_path
        with annotation_file.open() as fh:
            annotations = np.fromfile(fh, dtype=np.int16)
            annotations = np.array(annotations[2:10])

        return np.array([
            annotations[0], annotations[1],  # upper-left corner
            annotations[2] - annotations[0],  # width
            annotations[5] - annotations[1],  # height
            class_id,
            1
        ]).astype("float32").reshape((1,-1))

def _load_events(f_path, num_events):
    with h5py.File(str(f_path)) as fh:
        fh = fh['events']
        x = fh["x"][-num_events:]
        y = fh["y"][-num_events:]
        t = fh["t"][-num_events:]
        p = fh["p"][-num_events:]
    return dict(x=x, y=y, t=t, p=p)
