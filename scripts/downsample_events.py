import argparse
import tqdm
import hdf5plugin
import h5py
import weakref
import numba

import numpy as np

from pathlib import Path

from dsec_det.io import extract_from_h5_by_index, get_num_events


def _compression_opts():
    compression_level = 1  # {0, ..., 9}
    shuffle = 2  # {0: none, 1: byte, 2: bit}
    # From https://github.com/Blosc/c-blosc/blob/7435f28dd08606bd51ab42b49b0e654547becac4/blosc/blosc.h#L66-L71
    # define BLOSC_BLOSCLZ   0
    # define BLOSC_LZ4       1
    # define BLOSC_LZ4HC     2
    # define BLOSC_SNAPPY    3
    # define BLOSC_ZLIB      4
    # define BLOSC_ZSTD      5
    compressor_type = 5
    compression_opts = (0, 0, 0, 0, compression_level, shuffle, compressor_type)
    return compression_opts


H5_BLOSC_COMPRESSION_FLAGS = dict(
    compression=32001,
    compression_opts=_compression_opts(),  # Blosc
    chunks=True
)

def create_ms_to_idx(t_us):
    t_ms = t_us // 1000
    x, counts = np.unique(t_ms, return_counts=True)
    ms_to_idx = np.zeros(shape=(t_ms[-1] + 2,), dtype="uint64")
    ms_to_idx[x + 1] = counts
    ms_to_idx = ms_to_idx[:-1].cumsum()
    return ms_to_idx

class H5Writer:
    def __init__(self, outfile):
        assert not outfile.exists()

        self.h5f = h5py.File(outfile, 'a')
        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

        self.t_offset = None
        self.num_events = 0

        # create hdf5 datasets
        shape = (2 ** 16,)
        maxshape = (None,)

        self.h5f.create_dataset(f'events/x', shape=shape, dtype='u2', maxshape=maxshape, **H5_BLOSC_COMPRESSION_FLAGS)
        self.h5f.create_dataset(f'events/y', shape=shape, dtype='u2', maxshape=maxshape, **H5_BLOSC_COMPRESSION_FLAGS)
        self.h5f.create_dataset(f'events/p', shape=shape, dtype='u1', maxshape=maxshape, **H5_BLOSC_COMPRESSION_FLAGS)
        self.h5f.create_dataset(f'events/t', shape=shape, dtype='u4', maxshape=maxshape, **H5_BLOSC_COMPRESSION_FLAGS)

    def create_ms_to_idx(self):
        t_us = self.h5f['events/t'][()]
        self.h5f.create_dataset(f'ms_to_idx', data=create_ms_to_idx(t_us), dtype='u8', **H5_BLOSC_COMPRESSION_FLAGS)

    @staticmethod
    def close_callback(h5f: h5py.File):
        h5f.close()

    def add_data(self, events):
        if self.t_offset is None:
            self.t_offset = events['t'][0]
            self.h5f.create_dataset(f't_offset', data=self.t_offset, dtype='i8')

        events['t'] -= self.t_offset
        size = len(events['t'])
        self.num_events += size

        self.h5f[f'events/x'].resize(self.num_events, axis=0)
        self.h5f[f'events/y'].resize(self.num_events, axis=0)
        self.h5f[f'events/p'].resize(self.num_events, axis=0)
        self.h5f[f'events/t'].resize(self.num_events, axis=0)

        self.h5f[f'events/x'][self.num_events-size:self.num_events] = events['x']
        self.h5f[f'events/y'][self.num_events-size:self.num_events] = events['y']
        self.h5f[f'events/p'][self.num_events-size:self.num_events] = events['p']
        self.h5f[f'events/t'][self.num_events-size:self.num_events] = events['t']


def downsample_events(events, input_height, input_width, output_height, output_width, change_map=None):
    # this subsamples events if they were generated with cv2.INTER_AREA
    if change_map is None:
        change_map = np.zeros((output_height, output_width), dtype="float32")

    fx = int(input_width / output_width)
    fy = int(input_height / output_height)

    mask = np.zeros(shape=(len(events['t']),), dtype="bool")
    mask, change_map = _filter_events_resize(events['x'], events['y'], events['p'], mask, change_map, fx, fy)

    events = {k: v[mask] for k, v in events.items()}
    events['x'] = (events['x'] / fx).astype("uint16")
    events['y'] = (events['y'] / fy).astype("uint16")

    return events, change_map


@numba.jit(nopython=True, cache=True)
def _filter_events_resize(x, y, p, mask, change_map, fx, fy):
    # iterates through x,y,p of events, and increments cells of size fx x fy by 1/(fx*fy)
    # if one of these cells reaches +-1, then reset the cell, and pass through that event.
    # for memory reasons, this only returns the True/False for every event, indicating if
    # the event was skipped or passed through.
    for i in range(len(x)):
        x_l = x[i] // fx
        y_l = y[i] // fy
        change_map[y_l, x_l] += p[i] * 1.0 / (fx * fy)

        if np.abs(change_map[y_l, x_l]) >= 1:
            mask[i] = True
            change_map[y_l, x_l] -= p[i]

    return mask, change_map



if __name__ == '__main__':
    parser = argparse.ArgumentParser("""Downsample events""")
    parser.add_argument("--input_path", type=Path, required=True, help="Path to input events.h5. ")
    parser.add_argument("--output_path", type=Path, required=True, help="Path where output events.h5 will be written.")
    parser.add_argument("--input_height", type=int, default=480, help="Height of the input events resolution.")
    parser.add_argument("--input_width", type=int, default=640, help="Width of the input events resolution")
    parser.add_argument("--output_height", type=int, default=240, help="Height of the output events resolution.")
    parser.add_argument("--output_width", type=int, default=320, help="Width of the output events resolution.")
    args = parser.parse_args()

    num_events = get_num_events(args.input_path)
    num_events_per_chunk = 100000
    num_iterations = num_events // num_events_per_chunk

    writer = H5Writer(args.output_path)

    change_map = None
    pbar = tqdm.tqdm(total=num_iterations+1)
    for i in range(num_iterations):
        events = extract_from_h5_by_index(args.input_path, i * num_events_per_chunk, (i+1) * num_events_per_chunk)
        events['p'] = 2 * events['p'].astype("int8") - 1
        downsampled_events, change_map = downsample_events(events, change_map=change_map, input_height=args.input_height, input_width=args.input_width,
                                                      output_height=args.output_height, output_width=args.output_width)
        writer.add_data(downsampled_events)
        pbar.update(1)

    events = extract_from_h5_by_index(args.input_path, num_iterations * num_events_per_chunk, num_events)
    downsampled_events, change_map = downsample_events(events, change_map=change_map, input_height=args.input_height,
                                                       input_width=args.input_width,
                                                       output_height=args.output_height, output_width=args.output_width)
    writer.add_data(downsampled_events)
    pbar.update(1)

    writer.create_ms_to_idx()




