#! /bin/bash
model_name='deepfm'
batch_size=2048
learning_rate=0.001
epoch=2
device_tab=2
embedding_dim=32
CUDA_LAUNCH_BLOCKING=1 python main.py -m ${model_name} -bs ${batch_size} -lr ${learning_rate} -lr-type none -epoch ${epoch} -use-cuda -device-tab ${device_tab} -num-workers 8 -log -em-dim ${embedding_dim}