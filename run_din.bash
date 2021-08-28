#! /bin/bash
model_name='xdeepfm'
batch_size=8192
learning_rate=0.001
epoch=1
device_tab=0
embedding_dim=64
lambda=0.01
CUDA_LAUNCH_BLOCKING=1 python main.py -m ${model_name} -bs ${batch_size} -lr ${learning_rate} -epoch ${epoch} -use-cuda -device-tab ${device_tab} -num-workers 8 -log -em-dim ${embedding_dim} -online
CUDA_LAUNCH_BLOCKING=1 python main.py -m ${model_name} -bs ${batch_size} -lr ${learning_rate} -epoch ${epoch} -use-cuda -device-tab ${device_tab} -num-workers 8 -log -em-dim ${embedding_dim} -online
CUDA_LAUNCH_BLOCKING=1 python main.py -m ${model_name} -bs ${batch_size} -lr ${learning_rate} -epoch ${epoch} -use-cuda -device-tab ${device_tab} -num-workers 8 -log -em-dim ${embedding_dim} -online
CUDA_LAUNCH_BLOCKING=1 python main.py -m ${model_name} -bs ${batch_size} -lr ${learning_rate} -epoch ${epoch} -use-cuda -device-tab ${device_tab} -num-workers 8 -log -em-dim ${embedding_dim} -online
CUDA_LAUNCH_BLOCKING=1 python main.py -m ${model_name} -bs ${batch_size} -lr ${learning_rate} -epoch ${epoch} -use-cuda -device-tab ${device_tab} -num-workers 8 -log -em-dim ${embedding_dim} -online