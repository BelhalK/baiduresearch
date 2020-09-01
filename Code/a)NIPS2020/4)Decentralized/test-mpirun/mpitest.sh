#!/bin/bash
set -x
# 使用 mpirun 执行一个使用 mpi4py 的 mpi 程序
mpirun python -u each_node.py
if [[ $? -ne 0 ]]; then
    echo "train failed"
    exit 1
fi
exit 0