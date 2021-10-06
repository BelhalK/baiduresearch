for lr in 0.00001 0.00003
do
    srun -p TitanXx8_short --gres gpu:2 python main_fedlamb.py ./data/tiny-imagenet-200 \
        --dist-backend nccl --multiprocessing-distributed --dataset cifar -b 128 \
        --optimizer fedlamb --signum --epochs 20 --lr ${lr}
done