for lr in 0.001 0.0001 0.005
do
    srun -p TitanXx8_short --gres gpu:2 python main_fedlamb.py ./data/tiny-imagenet-200 \
        --dist-backend nccl --multiprocessing-distributed --dataset cifar -b 128 \
        --optimizer fedlamb --compress --all_reduce --lambda0 0 --epochs 5 --lr ${lr}
done
