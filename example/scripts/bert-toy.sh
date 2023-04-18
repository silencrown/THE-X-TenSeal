#!/bin/bash

python ../bert-toy.py \
    --softmax_hidden_size 128 \
    --softmax_num_samples 1000000 \
    --softmax_input_size 128 \
    --softmax_batch_size 32 \
    --softmax_lr 0.001 \
    --softmax_num_epochs 50
