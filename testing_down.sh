#!/bin/bash

# Set the environment variable to use the specified GPU
export CUDA_VISIBLE_DEVICES=0

# Define the parameters
TEST_MODE="down"
NAME="down_80_all"
SCALE=4
CROP=130
TEST_LR="/media/ml/Data Disk/CapsNetwork/Unsupervised/DATASET/Test/HR"
GPU="cuda:0"
SR_MODEL="gan"
TRAINING_TYPE="gan"
SAVE_RESULTS="--save_results"
REALSR="--realsr"

# Execute the Python script with the specified parameters
python predict.py \
  --test_mode $TEST_MODE \
  --name $NAME \
  --scale $SCALE \
  --resume_down "/media/ml/Data Disk/CapsNetwork/Unsupervised/MDA-SR-GAN/experiment/qkv_fusion_block/models/model_down_0080.pth" \
  --patch_size_down 512 \
  --test_lr "/media/ml/Data Disk/CapsNetwork/Unsupervised/DATASET/Conventional/Train/Cropped_HR" \
  --gpu $GPU \
  --sr_model $SR_MODEL \
  --training_type $TRAINING_TYPE \
  $SAVE_RESULTS \
  $REALSR
  