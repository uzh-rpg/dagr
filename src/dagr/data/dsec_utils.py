import numpy as np
import h5py


def construct_pairs(indices, n=2):
    indices = np.sort(indices)
    indices = np.stack([indices[i:i+1-n] for i in range(n-1)] + [indices[n-1:]])
    mask = np.ones_like(indices[0]) > 0
    for i, row in enumerate(indices):
        mask = mask & (indices[0] + i == row)
    indices = indices[...,mask].T
    return indices

def rescale_tracks(tracks, scale):
    tracks = tracks.copy()
    for k in "xywh":
        tracks[k] /= scale
    return tracks

def crop_tracks(tracks, width, height):
    tracks = tracks.copy()
    x1, y1 = tracks['x'], tracks['y']
    x2, y2 = x1 + tracks['w'], y1 + tracks['h']

    x1 = np.clip(x1, 0, width-1)
    x2 = np.clip(x2, 0, width-1)

    y1 = np.clip(y1, 0, height-1)
    y2 = np.clip(y2, 0, height-1)

    tracks['x'] = x1
    tracks['y'] = y1
    tracks['w'] = x2-x1
    tracks['h'] = y2-y1

    return tracks

def map_classes(class_ids, old_to_new_mapping):
    new_class_ids = old_to_new_mapping[class_ids]
    mask = new_class_ids > -1
    return new_class_ids, mask

def filter_small_bboxes(w, h, bbox_height=20, bbox_diag=30):
    """
    Filter out tracks that are too small.
    """
    diag = np.sqrt(h ** 2 + w ** 2)
    return (diag > bbox_diag) & (w > bbox_height) & (h > bbox_height)

def filter_tracks(dataset, image_width, image_height, class_remapping, min_bbox_height=0, min_bbox_diag=0, scale=1, only_perfect_tracks=False):
    image_index_pairs = {}
    track_masks = {}

    for directory_path in dataset.subsequence_directories:
        tracks = dataset.directories[directory_path.name].tracks.tracks
        image_timestamps = dataset.directories[directory_path.name].images.timestamps

        tracks_rescaled = rescale_tracks(tracks, scale)
        tracks_rescaled = crop_tracks(tracks_rescaled, image_width, image_height)

        _, class_mask = map_classes(tracks_rescaled['class_id'], class_remapping)
        size_mask = filter_small_bboxes(tracks_rescaled['w'], tracks_rescaled['h'], min_bbox_height, min_bbox_diag)
        final_mask = size_mask & class_mask

        # 1. stores indices of images which are valid, i.e. survived all filters above
        valid_image_indices = np.unique(np.nonzero(np.isin(image_timestamps, tracks_rescaled[final_mask]['t']))[0])
        valid_image_index_pairs = construct_pairs(valid_image_indices, 2)

        if only_perfect_tracks:
            valid_image_timestamp_brackets = image_timestamps[valid_image_index_pairs]
            img_idx_to_track_idx = compute_img_idx_to_track_idx(tracks['t'], valid_image_timestamp_brackets)
            mask = filter_by_only_perfect_tracks(tracks_rescaled, img_idx_to_track_idx, tracks_mask=final_mask)
            valid_image_index_pairs = valid_image_index_pairs[mask]

        image_index_pairs[directory_path.name] = valid_image_index_pairs
        track_masks[directory_path.name] = final_mask

    return image_index_pairs, track_masks

def _load_events(file, t0, num_events=None, num_us=None, height=None, time_window=None):
    with h5py.File(file, 'r') as f:
        ms = int((t0 - f['t_offset'][()]) / 1e3)
        idx0 = int(f['ms_to_idx'][ms])

        if num_events is not None:
            idx1 = idx0 + num_events
        if num_us is not None:
            idx1 = int(f['ms_to_idx'][ms + int(num_us / 1e3)])

        idx0, idx1 = sorted([idx0, idx1])
        idx0 = idx0 if idx0 >= 0 else 0
        idx1 = idx1 if idx1 >= 0 else 0

        # load all events
        events = {k: f[f'events/{k}'][idx0:idx1] for k in "xytp"}

        tq = events['t'][-1] if idx1 > idx0 else f[f'events/t'][max([idx1 - 1, idx0])]

        # cast to desired types
        p = 2 * events["p"][..., None].astype("int8") - 1
        t_ev = events['t'][..., None]
        xy = np.stack([events['x'], events['y']], axis=-1).astype("int16")

        if time_window is not None:
            t = (time_window - tq + t_ev).astype('int32')
        else:
            t = tq.copy()

        # we have to add the offset here
        tq += f['t_offset'][()]
        tq = tq.astype("int64")

        # crop events to crop height
        mask = (t[:, 0] > 0)
        if height is not None:
            mask &= (xy[:, 1] < height)

        events = (xy[mask], t[mask], p[mask])

        return events, tq


def filter_by_only_perfect_tracks(tracks, img_idx_to_track_idx, tracks_mask=None):
    i0, i1 = img_idx_to_track_idx
    mask = np.ones_like(i0[0]) > 0
    for i in range(i0.shape[1]):
        track = [tracks[i0[j][i]:i1[j][i]] for j in range(len(i0))]
        if tracks_mask is not None:
            track_mask = [tracks_mask[i0[j][i]:i1[j][i]] for j in range(len(i0))]
            track = [t[m] for t, m in zip(track, track_mask)]
        mask[i] = not is_invalid_track(track)
    return mask

def is_invalid_track(track):
    track = [tr[tr['track_id'].argsort()] for tr in track]

    i_tr = track[0]
    for c_tr in track[1:]:
        if len(i_tr) != len(c_tr):
            return True
        if not (c_tr['track_id'] == i_tr['track_id']).all():
            return True
        iou = compute_iou(i_tr, c_tr)
        min_iou = np.min(iou)
        if min_iou < 0.10:
            return True
    else:
        return False

def compute_iou(track0, track1):
    x1, x2 = track0['x'], track0['x'] + track0['w']
    y1, y2 = track0['y'], track0['y'] + track0['h']

    x1g, x2g = track1['x'], track1['x'] + track1['w']
    y1g, y2g = track1['y'], track1['y'] + track1['h']

    # Intersection keypoints
    xkis1 = np.max(np.stack([x1, x1g]), axis=0)
    ykis1 = np.max(np.stack([y1, y1g]), axis=0)
    xkis2 = np.min(np.stack([x2, x2g]), axis=0)
    ykis2 = np.min(np.stack([y2, y2g]), axis=0)

    intsct = np.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsct[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    union = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsct + 1e-9
    iou = intsct / union

    return iou


def compute_indices_for_contiguous_parts(x):
    x, counts = np.unique(x, return_counts=True)
    idx = np.concatenate([np.array([0]), counts]).cumsum()
    return np.stack([idx[:-1], idx[1:]], axis=-1)

def _compute_img_idx_to_track_idx(t, t_query):
    new_img_idx = compute_indices_for_contiguous_parts(t)
    mask  = np.isin(np.unique(t), t_query)
    new_img_idx = new_img_idx[mask].T
    return new_img_idx

def compute_img_idx_to_track_idx(t, t_query):
    return np.stack([_compute_img_idx_to_track_idx(t, t_q) for t_q in t_query.T])

def compute_class_mapping(classes, all_classes, mapping):
    output_mapping = []
    for i, c in enumerate(all_classes):
        mapped_class = mapping[c]
        output_mapping.append(classes.index(mapped_class) if mapped_class in classes else -1)
    return np.array(output_mapping)
