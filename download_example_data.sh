#! /usr/bin/env bash
DAGR_DIR=$(pwd)
DATA_DIR=$DAGR_DIR/data

mkdir $DATA_DIR
wget https://download.ifi.uzh.ch/rpg/dagr/data/dagr_s_50.pth -O $DATA_DIR/dagr_s_50.pth

wget https://download.ifi.uzh.ch/rpg/dagr/data/DSEC_fragment.zip -O $DATA_DIR/DSEC_fragment.zip
unzip $DATA_DIR/DSEC_fragment.zip -d $DATA_DIR
rm -rf $DATA_DIR/DSEC_fragment.zip