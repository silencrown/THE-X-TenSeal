#!/bin/bash

python ../bert-toy.py \
    --is_softmax 1 \
    --softmax_hidden_size 128 \
    --softmax_num_samples 1000000 \
    --softmax_input_size 128 \
    --softmax_batch_size 128 \
    --softmax_lr 0.00001 \
    --softmax_num_epochs 500
