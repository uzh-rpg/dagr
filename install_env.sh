#! /usr/bin/env bash

TORCH=$(python -c "import torch; print(torch.__version__)")
CUDA=$(python -c "import torch; print(torch.version.cuda)")
URL=https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

pip install --no-cache-dir torch-scatter -f $URL;
pip install --no-cache-dir torch-cluster -f $URL;
pip install --no-cache-dir torch-spline-conv -f $URL;
pip install --no-cache-dir torch-sparse -f $URL;
pip install torch-geometric;
pip install wandb numba hdf5plugin plotly matplotlib pycocotools opencv-python scikit-video pandas ruamel.yaml
