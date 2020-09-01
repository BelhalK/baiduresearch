#!/bin/bash
export LD_LIBRARY_PATH=/home/work/cuda-9.0/lib64/:/home/work/cuda-9.0/lib/:/home/work/cuda-9.0/extras/CUPTI/lib64:/home/work/cudnn/\
cudnn_v7/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/home/work/cudnn/cudnn_v7/cuda/include:$CPATH
export LIBRARY_PATH=/home/work/cudnn/cudnn_v7/cuda/lib64:$LIBRARY_PATH

mpirun /opt/conda/envs/py36/bin/pip install mpi4py

set -x
mpirun /opt/conda/envs/py36/bin/python -u maindams.py --backend mpi --optimizer sgd --checkpoint_dir ./checkpoints\
    --batch_size 256 --lr 0.1 --num_dataloader_workers 10 \
    --num_epochs 5 --nesterov True --warmup True --push_sum False \
    --graph_type 1 --schedule 30 0.1 60 0.1 80 0.1 \
    --train_fast False --master_port 40100 \
    --tag 'DPSGD_IB' --print_freq 100 --verbose False \
    --all_reduce False --seed 1 \
    --network_interface_type 'infiniband'