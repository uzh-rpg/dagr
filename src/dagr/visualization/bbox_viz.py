import numpy as np
import cv2
import torchvision
import torch


_COLORS = np.array([[0.000, 0.8, 0.1], [1, 0.67, 0.00]])
class_names = ["car", "pedestrian"]


def draw_bbox_on_img(img, x, y, w, h, labels, scores=None, conf=0.5, nms=0.45, label="", linewidth=2):
    if scores is not None:
        mask = filter_boxes(x, y, w, h, labels, scores, conf, nms)
        x = x[mask]
        y = y[mask]
        w = w[mask]
        h = h[mask]
        labels = labels[mask]
        scores = scores[mask]

    for i in range(len(x)):
        if scores is not None and scores[i] < conf:
            continue

        x0 = int(x[i])
        y0 = int(y[i])
        x1 = int(x[i] + w[i])
        y1 = int(y[i] + h[i])
        cls_id = int(labels[i])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()

        text = f"{label}-{class_names[cls_id]}"

        if scores is not None:
            text += f":{scores[i] * 100: .1f}"

        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, linewidth)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        txt_height = int(1.5*txt_size[1])
        cv2.rectangle(
            img,
            (x0, y0 - txt_height),
            (x0 + txt_size[0] + 1, y0 + 1),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]-txt_height), font, 0.4, txt_color, thickness=1)
    return img

def filter_boxes(x, y, w, h, labels, scores, conf, nms):
    mask = scores > conf

    x1, y1 = x + w, y + h
    box_coords = np.stack([x, y, x1, y1], axis=-1)

    nms_out_index = torchvision.ops.batched_nms(
        torch.from_numpy(box_coords),
        torch.from_numpy(np.ascontiguousarray(scores)),
        torch.from_numpy(labels),
        nms
    )

    nms_mask = np.ones_like(mask) == 0
    nms_mask[nms_out_index] = True

    return mask & nms_mask
