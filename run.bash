#! /bin/bash
model_name='deepfm'
batch_size=8192
learning_rate=0.001
epoch=1
device_tab=2
embedding_dim=16
CUDA_LAUNCH_BLOCKING=1 python main.py -m ${model_name} -bs ${batch_size} -lr ${learning_rate} -lr-type none -epoch ${epoch} -use-cuda -device-tab ${device_tab} -num-workers 8 -log -em-dim ${embedding_dim} -online
#CUDA_LAUNCH_BLOCKING=1 python grid_search.py -m ${model_name} -epoch ${epoch} -use-cuda -device-tab ${device_tab} -num-workers 8 -log

