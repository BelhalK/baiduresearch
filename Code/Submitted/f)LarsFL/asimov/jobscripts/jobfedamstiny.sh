for lr in 5e-2 1e-1
do
    srun -p 1080Ti_mlong --gres gpu:2 python main_fedlamb.py ~/data/tiny-imagenet-200 \
        --dist-backend nccl --multiprocessing-distributed --dataset tinyimagenet -b 64 \
        --optimizer fedsgd --compress --all_reduce --lambda0 0 --epochs 100 --lr ${lr}
done
