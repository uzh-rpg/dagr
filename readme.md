# Low Latency Automotive Vision with Event Cameras

<p align="center">
<a href="https://youtu.be/dwzGhMQCc4Y">
  <img src="assets/Nature_Gehrig_YouTube_cover_yt.jpg" alt="DAGR" width="500"/>
</a>
</p>

This repository contains code from our 2024 Nature paper which can be accessed for free here [PDF Open Access](https://www.nature.com/articles/s41586-024-07409-w).
**_Low Latency Automotive Vision with Event Cameras_** by [Daniel Gehrig](https://danielgehrig18.github.io/) and [Davide Scaramuzza](http://rpg.ifi.uzh.ch/people_scaramuzza.html). 
If you use our code or refer to this project, please cite it using

```bibtex
@Article{Gehrig24nature,
  author    = {Gehrig, Daniel and Scaramuzza, Davide},
  title     = {Low Latency Automotive Vision with Event Cameras},
  booktitle = {Nature},
  year      = {2024}
}
```

## Installation
First, download the github repository and its dependencies
```bash
WORK_DIR=/path/to/work/directory/
cd $WORK_DIR
git clone git@github.com:uzh-rpg/dagr.git
DAGR_DIR=$WORK_DIR/dagr

cd $DAGR_DIR 

```
Then start by installing the main libraries. Make sure Anaconda (or better Mamba), PyTorch, and CUDA is installed. 
```bash
cd $DAGR_DIR
conda create -y -n dagr python=3.8 
conda activate dagr
conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
Then install the pytorch-geometric libraries. This may take a while.
```bash
bash install_env.sh
```
The above bash file will figure out the CUDA and Torch version, and install the appropriate pytorch-geometric packages.
Then, download and install additional dependencies locally 
```bash
bash download_and_install_dependencies.sh
```
Finally, install the dagr package
```bash
pip install -e .
```

## Run Example
After installing, you can download a data fragment, and checkpoint with 
```bash
bash download_example_data.sh
```
This will download a checkpoint and data fragment of DSEC-Detection on which you can test the code. 
Once downloaded, run the following command
```bash 
LOG_DIR=/path/to/log
DEVICE=1
CUDA_VISIBLE_DEVICES=$DEVICE python scripts/run_test_interframe.py --config config/dagr-s-dsec.yaml \
                                                                   --use_image \
                                                                   --img_net resnet50 \
                                                                   --checkpoint data/dagr_s_50.pth \
                                                                   --batch_size 8 \
                                                                   --dataset_directory data/DSEC_fragment \                                                        
                                                                   --output_directory $LOG_DIR
```
note the wandb directory as `$WANDB_DIR` and then visualize the detections with 
```bash
python scripts/visualize_detections.py --detections_folder $LOG_DIR/$WANDB_DIR \
                                       --dataset_directory data/DSEC_fragment/test \
                                       --vis_time_step_us 1000 \ 
                                       --event_time_window_us 5000 \
                                       --sequence zurich_city_13_b
```

## Test on DSEC
Start by downloading the DSEC dataset and the additional labelled data introduced in this work. 
To do so, follow [these instructions](https://github.com/uzh-rpg/dsec-det?tab=readme-ov-file#download-dsec). They are based on the scripts 
of [dsec-det](https://github.com/uzh-rpg/dsec-det), which can be found in `libs/dsec-det/scripts`.
To continue, complete sections [Download DSEC](https://github.com/uzh-rpg/dsec-det?tab=readme-ov-file#download-dsec) until [Test Alignment](https://github.com/uzh-rpg/dsec-det?tab=readme-ov-file#test-alignment). 
If you already downloaded DSEC, make sure `$DSEC_ROOT` points to it, and instead start at section [Download DSEC-extra
](https://github.com/uzh-rpg/dsec-det?tab=readme-ov-file#download-dsec-extra).  

After downloading all the data, change back to $DAGR_DIR, and start by downsampling the events 
```bash
cd $DAGR_DIR
bash scripts/downsample_all_events.sh $DSEC_ROOT
```

### Running Evaluation
This repository implements three scripts for running evaluation of the model on DSEC-Det. 
The first, evaluates the detection performance of the model after seeing one image, and the subsequent 50 milliseconds of events.
To run it, specify a device, and logging directory with  type 
```bash 
LOG_DIR=/path/to/log
DEVICE=1
CUDA_VISIBLE_DEVICES=$DEVICE python scripts/run_test.py --config config/dagr-s-dsec.yaml \
                                                        --use_image \
                                                        --img_net resnet50 \
                                                        --checkpoint data/dagr_s_50.pth \
                                                        --batch_size 8 \
                                                        --dataset_directory $DSEC_ROOT \
                                                        --output_directory $LOG_DIR
```
Then, to evaluate the number of FLOPS generated in asynchronous mode, run 
```bash 
LOG_DIR=/path/to/log
DEVICE=1
CUDA_VISIBLE_DEVICES=$DEVICE python scripts/count_flops.py --config config/eagr-s-dsec.yaml \
                                                           --use_image \
                                                           --img_net resnet50 \
                                                           --checkpoint data/dagr_s_50.pth \
                                                           --batch_size 8 \
                                                           --dataset_directory $DSEC_ROOT \
                                                           --output_directory $LOG_DIR
```
Finally, to evaluate the interframe detection performance of our method run
```bash
LOG_DIR=/path/to/log
DEVICE=1
CUDA_VISIBLE_DEVICES=$DEVICE python scripts/run_test_interframe.py --config config/eagr-s-dsec.yaml \
                                                                   --use_image \
                                                                   --img_net resnet50 \
                                                                   --checkpoint data/dagr_s_50.pth \
                                                                   --batch_size 8 \
                                                                   --dataset_directory $DSEC_ROOT \
                                                                   --output_directory $LOG_DIR \
                                                                   --num_interframe_steps 10
```
This last script will write the high-rate detections from our method into the folder `$LOG_DIR/$WANDB_DIR`, 
where `$WANDB_DIR` is the automatically generated folder created by wandb. 
To visualize the detections, use the following script: 
```bash
python scripts/visualize_detections.py --detections_folder $LOG_DIR/$WANDB_DIR \
                                       --dataset_directory $DSEC_ROOT/test/ \
                                       --vis_time_step_us 1000 \ 
                                       --event_time_window_us 5000 \
                                       --sequence zurich_city_13_b
                                       
```
This will start a visualization window showing the detections over a given sequence. If you want to save the detections 
to a video, use the `--write_to_output` flag, which will create a video in the folder `$LOG_DIR/$WANDB_DIR/visualization}`.  
