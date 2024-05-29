import torch
from dagr.utils.logging import log_bboxes
from dagr.utils.buffers import DetectionBuffer, format_data
import tqdm

def to_npy(detections):
    return [{k: v.cpu().numpy() for k, v in d.items()} for d in detections]

def format_detections(sequences, t, detections):
    detections = to_npy(detections)
    for i, det in enumerate(detections):
        det['sequence'] = sequences[i]
        det['t'] = t[i]
    return detections

def run_test_with_visualization(loader, model, dataset: str, log_every_n_batch=-1, name="", compile_detections=False,
                                no_eval=False):
    model.eval()

    if not no_eval:
        mapcalc = DetectionBuffer(height=loader.dataset.height, width=loader.dataset.width,
                                  classes=loader.dataset.classes)

    counter = 0
    if compile_detections:
        compiled_detections = []

    for i, data in enumerate(tqdm.tqdm(loader, desc=f"Testing {name}")):
        data = data.cuda(non_blocking=True)
        data_for_visualization = data.clone()

        data = format_data(data)
        detections, targets = model(data.clone())

        if compile_detections:
            compiled_detections.extend(format_detections(data.sequence, data.t1, detections))

        if log_every_n_batch > 0 and counter % log_every_n_batch == 0:
            log_bboxes(data_for_visualization, targets=targets, detections=detections, bidx=4,
                       class_names=loader.dataset.classes, key="testing/evaluated_bboxes")

        if not no_eval:
            mapcalc.update(detections, targets, dataset, data.height[0], data.width[0])

        if i % 5 == 0:
            torch.cuda.empty_cache()

        counter += 1

    torch.cuda.empty_cache()

    data = None
    if not no_eval:
        data = mapcalc.compute()

    return (data, compiled_detections) if compile_detections else data