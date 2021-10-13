lr=0.0001

#######MNIST######
# signum mnist vanilla net
srun -p 1080Ti_slong --gres gpu:2 python working_dist.py ~/data/ILSVRC/Data/CLS-LOC \
        --dist-backend nccl --multiprocessing-distributed --dataset mnist -b 256 \
        --optimizer signum --signum --compress --all_reduce --epochs 3 --lr ${lr}



####### TINY IMAGENET######
# signum tinyimagenet resnet18
srun -p 1080Ti_slong --gres gpu:4 python working_dist.py ./data/tiny-imagenet-200 \
        --dist-backend nccl --multiprocessing-distributed --dataset tinyimagenet -b 128 \
        --optimizer signum --signum --compress --all_reduce --epochs 3 --lr ${lr}
# signum tinyimagenet resnet50
srun -p 1080Ti_slong --gres gpu:2 python working_dist.py ./data/tiny-imagenet-200 \
        --dist-backend nccl --multiprocessing-distributed --dataset tinyimagenet -b 64 \
        --optimizer signum --signum --compress --all_reduce --epochs 3 --lr ${lr}; 

####### CIFAR######
# signum cifar resnet18

srun -p TitanXx8_short --gres gpu:2 python working_dist.py ./data/tiny-imagenet-200 \
        --dist-backend nccl --multiprocessing-distributed --dataset cifar -b 128 \
        --optimizer compams --signum --sparsity 0.01 --method topk --epochs 3 --lr 0.001


#### IMAGENET###
# signSGD resnet50 (no --compress)
srun -p 1080Ti_slong --gres gpu:2 python working_dist.py ~/data/ILSVRC/Data/CLS-LOC \
        --dist-backend nccl --multiprocessing-distributed --dataset imagenet -b 64 \
        --optimizer signum --signum --all_reduce --epochs 3 --lr ${lr}

# majority resnet50 (with --compress)
srun -p 1080Ti_slong --gres gpu:2 python working_dist.py ~/data/ILSVRC/Data/CLS-LOC \
        --dist-backend nccl --multiprocessing-distributed --dataset imagenet -b 64 \
        --optimizer signum --signum --compress  --all_reduce --epochs 3 --lr ${lr}

# compams resnet50
srun -p 1080Ti_slong --gres gpu:2 python working_dist.py ~/data/ILSVRC/Data/CLS-LOC \
        --dist-backend nccl --multiprocessing-distributed --dataset imagenet -b 64 \
        --optimizer compams --method topk  --all_reduce --epochs 3 --lr ${lr}

# Distributed AMSGrad resnet50
srun -p 1080Ti_slong --gres gpu:2 python working_dist.py ~/data/ILSVRC/Data/CLS-LOC \
        --dist-backend nccl --multiprocessing-distributed --dataset imagenet -b 64 \
        --optimizer compams --method full  --all_reduce --epochs 3 --lr ${lr}