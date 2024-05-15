#!/bin/bash

DSEC_ROOT=$1
for split in train test; do
    for sequence in $DSEC_ROOT/$split/*/; do
        infile=$sequence/events/left/events.h5
        outfile=$sequence/events/left/events_2x.h5
        python scripts/downsample_events.py --input_path $infile --output_path $outfile
    done
done