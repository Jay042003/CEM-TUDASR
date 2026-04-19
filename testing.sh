#!/bin/bash

# Set the environment variable to use the specified GPU
export CUDA_VISIBLE_DEVICES=0

# Define the parameters
TEST_MODE="sr_patch"
NAME="ablation_equal_loss"
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
  --pretrain_sr "/media/ml/Data Disk/CapsNetwork/Unsupervised/CEM-TUDASR/experiment/ablation_equal_loss/models/best_sr_model.pth" \
  --crop $CROP \
  --test_lr "/media/ml/Data Disk/CapsNetwork/Unsupervised/DATASET/Test/HR" \
  --gpu $GPU \
  --sr_model $SR_MODEL \
  --training_type $TRAINING_TYPE \
  $SAVE_RESULTS \
  $REALSR