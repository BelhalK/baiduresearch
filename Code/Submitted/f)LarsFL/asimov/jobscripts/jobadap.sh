for lr in 0.0005
do
    srun -p TitanXx8_short --gres gpu:2 python main_fedlamb.py ~/data/tiny-imagenet-200 \
        --dist-backend nccl --multiprocessing-distributed --dataset cifar -b 128 \
        --optimizer reddi --compress --all_reduce --lambda0 0 --epochs 100 --lr ${lr}
done
