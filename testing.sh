#!/bin/bash

# Set the environment variable to use the specified GPU
export CUDA_VISIBLE_DEVICES=0

# Define the parameters
TEST_MODE="sr_patch"
NAME="NAME OF THE TEST FOLDER"
SCALE=4
CROP=130
TEST_LR="PATH TO DATA"
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
  --pretrain_sr "PATH TO TRAINED MODEL" \
  --crop $CROP \
  --test_lr "PATH TO DATA" \
  --gpu $GPU \
  --sr_model $SR_MODEL \
  --training_type $TRAINING_TYPE \
  $SAVE_RESULTS \
  $REALSR
