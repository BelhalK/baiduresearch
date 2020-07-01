#!/bin/bash
#
#SBATCH --job-name=DPSGD_ETH
#SBATCH --output=DPSGD_ETH.out
#SBATCH --error=DPSGD_ETH.err
#SBATCH --nodes=5
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=30:00:00

# Replace NB_NODES with the number of nodes to use
# Load any modules and activate your conda environment here

srun python3 -u maindams.py --optimizer damsgrad --checkpoint_dir /home/belhal_karimi_gmail_com/dams/checkpoints\
    --batch_size 256 --lr 0.1 --num_dataloader_workers 16 \
    --num_epochs 90 --nesterov True --warmup True --push_sum False \
    --graph_type 1 --schedule 30 0.1 60 0.1 80 0.1 \
    --train_fast False --master_port 40100 \
    --tag 'DPSGD_ETH' --print_freq 100 --verbose False \
    --all_reduce False --seed 1 \
    --network_interface_type 'ethernet'
