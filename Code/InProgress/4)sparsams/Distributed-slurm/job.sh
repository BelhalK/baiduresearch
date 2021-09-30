# signum mnist vanilla net
srun -p 1080Ti_slong --gres gpu:2 python working_dist.py ~/data/ILSVRC/Data/CLS-LOC \
        --dist-backend nccl --multiprocessing-distributed --dataset mnist -b 256 \
        --optimizer signum --signum --compress --all_reduce --epochs 3


# signum tinyimagenet resnet18
srun -p 1080Ti_slong --gres gpu:4 python working_dist.py ./data/tiny-imagenet-200 \
        --dist-backend nccl --multiprocessing-distributed --dataset tinyimagenet -b 128 \
        --optimizer signum --signum --compress --all_reduce --epochs 3 --lr 0.001

# signum imagenet resnet50
srun -p 1080Ti_slong --gres gpu:2 python working_dist.py ./data/tiny-imagenet-200 \
        --dist-backend nccl --multiprocessing-distributed --dataset tinyimagenet -b 64 \
        --optimizer signum --signum --compress --all_reduce --epochs 3
