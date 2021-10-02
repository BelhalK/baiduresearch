srun -p TitanXx8_short --gres gpu:2 python working_dist.py ./data/tiny-imagenet-200 \
        --dist-backend nccl --multiprocessing-distributed --dataset cifar -b 256 \
        --optimizer compams --signum --sparsity 0.01 --method topk --epochs 200 --lr 0.0001


srun -p TitanXx8_short --gres gpu:2 python working_dist.py ./data/tiny-imagenet-200 \
        --dist-backend nccl --multiprocessing-distributed --dataset cifar -b 256 \
        --optimizer compams --signum --sparsity 0.01 --method full --epochs 200 --lr 0.0001


srun -p TitanXx8_short --gres gpu:2 python working_dist.py ./data/tiny-imagenet-200 \
        --dist-backend nccl --multiprocessing-distributed --dataset cifar -b 256 \
        --optimizer signum --signum --compress --sparsity 0.01 --method full --epochs 200 --lr 0.0001

srun -p TitanXx8_short --gres gpu:2 python working_dist.py ./data/tiny-imagenet-200 \
        --dist-backend nccl --multiprocessing-distributed --dataset cifar -b 256 \
        --optimizer signum --signum --compress --sparsity 0.01 --method full --epochs 200 --lr 0.0005