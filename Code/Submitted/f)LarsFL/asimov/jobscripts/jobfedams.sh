for lr in 0.0001
do
    srun -p 1080Ti --gres gpu:2 python main_fedlamb.py ~/data/tiny-imagenet-200 \
        --dist-backend nccl --multiprocessing-distributed --dataset cifar -b 128 \
        --optimizer fedams --compress --all_reduce --lambda0 0 --epochs 100 --lr ${lr}
done
