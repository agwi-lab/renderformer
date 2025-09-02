#!/bin/bash

# Количество GPU
NUM_GPUS=1

# Запуск распределенного обучения
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train.py \
    --config configs/config.yml