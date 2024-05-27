import cv2
import argparse

from pathlib import Path
import numpy as np

from dsec_det.directory import DSECDirectory
from dsec_det.io import extract_from_h5_by_timewindow, extract_image_by_index, load_start_and_end_time
from dsec_det.preprocessing import compute_index

from dagr.visualization.bbox_viz import draw_bbox_on_img
from dagr.visualization.event_viz import draw_events_on_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser("""Visualization script to show bounding boxes""")
    parser.add_argument("--detections_folder", help="Path to folder with detections.", type=Path, default="/data/scratch1/daniel/logs/dsec/detection/graceful-snowball-1298")
    parser.add_argument("--dataset_directory", help="Path to DSEC folder including which split.", type=Path, default="/data/scratch1/daniel/datasets/DSEC_fragment/test")
    parser.add_argument("--vis_time_step_us", help="Number of microseconds to step each iteration.", type=int, default=1000)
    parser.add_argument("--event_time_window_us", help="Length of sliding event time window for visualization.", type=int, default=5000)
    parser.add_argument("--sequence", help="Sequence to visualize. Must be an official DSEC sequence e.g. zurich_city_13_b", default="zurich_city_13_b", type=str)
    parser.add_argument("--write_to_output", help="Whether to save images in folder ${detections_folder}/visualization. Otherwise, just cv2.imshow is used.", action="store_true")
    args = parser.parse_args()

    assert args.detections_folder.exists()
    assert args.dataset_directory.exists()
    assert (args.detections_folder / f"detections_{args.sequence}.npy").exists()
    assert args.vis_time_step_us > 0
    assert args.event_time_window_us > 0

    if args.write_to_output:
        output_path = args.detections_folder / "visualization"
        output_path.mkdir(parents=True, exist_ok=True)

    detections_file = args.detections_folder / f"detections_{args.sequence}.npy"
    detections = np.load(detections_file)
    detection_timestamps = np.unique(detections['t'])

    dsec_directory = DSECDirectory(args.dataset_directory / args.sequence)

    t0, t1 = load_start_and_end_time(dsec_directory)

    vis_timestamps = np.arange(t0, t1, step=args.vis_time_step_us)
    step_index_to_image_index = compute_index(dsec_directory.images.timestamps, vis_timestamps)
    step_index_to_boxes_index = compute_index(detection_timestamps, vis_timestamps)

    scale = 2

    for step, t in enumerate(vis_timestamps):

        # find most recent image
        image_index = step_index_to_image_index[step]
        image = extract_image_by_index(dsec_directory.images.image_files_distorted, image_index)

        # find events within time window [image_timestamps, t]
        events = extract_from_h5_by_timewindow(dsec_directory.events.event_file, t-args.event_time_window_us, t)

        # find most recent bounding boxes
        boxes_index = step_index_to_boxes_index[step]
        boxes_timestamp = detection_timestamps[boxes_index]
        boxes = detections[detections['t'] == boxes_timestamp]

        # draw them on one image
        scale = 2
        image = draw_events_on_image(image, events['x'], events['y'], events['p'])
        image = draw_bbox_on_img(image, scale*boxes['x'], scale*boxes['y'], scale*boxes['w'], scale*boxes["h"],
                                 boxes["class_id"], boxes['class_confidence'], conf=0.3, nms=0.65)

        if args.write_to_output:
            cv2.imwrite(str(output_path / ("%06d.png" % step)), image)
        else:
            cv2.imshow("DSEC Det: Visualization", image)
            cv2.waitKey(3)

