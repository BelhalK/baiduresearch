#!/bin/bash
export LD_LIBRARY_PATH=/home/work/cuda-9.0/lib64/:/home/work/cuda-9.0/lib/:/home/work/cuda-9.0/extras/CUPTI/lib64:/home/work/cudnn/\
cudnn_v7/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/home/work/cudnn/cudnn_v7/cuda/include:$CPATH
export LIBRARY_PATH=/home/work/cudnn/cudnn_v7/cuda/lib64:$LIBRARY_PATH

mpirun /opt/conda/envs/py36/bin/pip install mpi4py

set -x
mpirun /opt/conda/envs/py36/bin/python -u test.py > runing_log.txt