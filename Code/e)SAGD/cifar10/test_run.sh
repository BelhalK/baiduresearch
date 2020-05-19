sh env.sh
python/bin/python3 -u run_nn_cifar10.py --model resnet --cuda-id 0 --lr 0.2 --batch-size 128 --epochs 100 --repeat 1  --LR-decay True --decay-epoch 30 --optim padam
