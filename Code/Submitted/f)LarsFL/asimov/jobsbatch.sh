#!/bin/bash
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#SBATCH --job-name=signum-tinyimagenet1080Ti
#SBATCH --output=out/signum-tinyimagenet1080Ti.out
#SBATCH --error=out/signum-tinyimagenet1080Ti.err
#SBATCH --nodes=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00
#SBATCH --partition=1080Ti_slong

srun python working_dist.py ./data/tiny-imagenet-200 \
        --dist-backend nccl --multiprocessing-distributed --dataset cifar -b 128 \
        --optimizer signum --signum --compress --all_reduce --epochs 3 --lr 0.005