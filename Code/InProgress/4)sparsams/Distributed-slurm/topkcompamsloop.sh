for lr in 0.0001 0.0005 0.001 0.005 0.01
do
    srun -p P100 --gres gpu:2 python working_dist.py ./data/tiny-imagenet-200 \
        --dist-backend nccl --multiprocessing-distributed --dataset cifar -b 128 \
        --optimizer compams --signum --sparsity 0.01 --method topk --epochs 20 --lr ${lr}
done