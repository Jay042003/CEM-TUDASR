#!/bin/bash

# Set the visible CUDA device
export CUDA_VISIBLE_DEVICES=0

# Define variables
EXP_PATH=$'./EXPERIMEN NAME'

# Execute the training script with the provided arguments
python main.py --name ${EXP_PATH} \
                --scale 4 \
                --adv_w 0.001 \
                --batch_size 4 \
                --patch_size_down 256 \
                --decay_batch_size_sr 400000 \
                --decay_batch_size_down 50000 \
                --epochs_sr_start 1 \
                --gpu cuda:0 \
                --sr_model gan \
                --training_type gan \
                --joint \
                --save_results \
                --save_log  \
                --resume_down 'PATH TO RESUME CHECKPOINT FOR DOWN MODEL' 
