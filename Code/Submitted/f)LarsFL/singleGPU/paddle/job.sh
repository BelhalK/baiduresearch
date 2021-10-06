export LD_LIBRARY_PATH=/home/work/cuda-9.0/lib64/:/home/work/cuda-9.0/lib/:/home/work/cuda-9.0/extras/CUPTI/lib64:/home/work/cudnn/\
cudnn_v7/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/home/work/cudnn/cudnn_v7/cuda/include:$CPATH
export LIBRARY_PATH=/home/work/cudnn/cudnn_v7/cuda/lib64:$LIBRARY_PATH
mkdir log
mkdir local_data


/opt/conda/envs/py36/bin/pip install matplotlib
/opt/conda/envs/py36/bin/pip install collections-extended
/opt/conda/envs/py36/bin/pip install sklearn



/opt/conda/envs/py36/bin/python -u main2.py --dataset cifar --model cnn --epochs 100 --optim sgd --num_users 10 --gpu -1 \
--frac 0.5 --local_ep 1 --lr 0.05 --customarg 1 > run_sgd.txt

/opt/conda/envs/py36/bin/python -u main2.py --dataset cifar --model cnn --epochs 100 --optim sgd --num_users 10 --gpu -1 \
--frac 0.5 --local_ep 1 --lr 0.005 --customarg 1 --LAMB --lambda0 0 > run_sgdlamb.txt
/opt/conda/envs/py36/bin/python -u main2.py --dataset cifar --model cnn --epochs 100 --optim sgd --num_users 10 --gpu -1 \
--frac 0.5 --local_ep 1 --lr 0.01 --customarg 1 --LAMB --lambda0 0 > run_sgdlamb.txt



/opt/conda/envs/py36/bin/python -u main2.py --dataset cifar --model cnn --epochs 100 --optim sgd --num_users 10 --gpu -1 \
--frac 0.5 --local_ep 3 --lr 0.05 --customarg 1 > run_sgd.txt

/opt/conda/envs/py36/bin/python -u main2.py --dataset cifar --model cnn --epochs 100 --optim sgd --num_users 10 --gpu -1 \
--frac 0.5 --local_ep 3 --lr 0.005 --customarg 1 --LAMB --lambda0 0 > run_sgdlamb.txt
/opt/conda/envs/py36/bin/python -u main2.py --dataset cifar --model cnn --epochs 100 --optim sgd --num_users 10 --gpu -1 \
--frac 0.5 --local_ep 3 --lr 0.01 --customarg 1 --LAMB --lambda0 0 > run_sgdlamb.txt




/opt/conda/envs/py36/bin/python -u main2.py --dataset cifar --model cnn --epochs 100 --optim sgd --num_users 50 --gpu -1 \
--frac 0.5 --local_ep 1 --lr 0.005 --customarg 1 --LAMB --lambda0 0 > run_sgdlamb.txt
/opt/conda/envs/py36/bin/python -u main2.py --dataset cifar --model cnn --epochs 100 --optim sgd --num_users 50 --gpu -1 \
--frac 0.5 --local_ep 3 --lr 0.005 --customarg 1 --LAMB --lambda0 0 > run_sgdlamb.txt


/opt/conda/envs/py36/bin/python -u main2.py --dataset cifar --model cnn --epochs 100 --optim sgd --num_users 50 --gpu -1 \
--frac 0.5 --local_ep 1 --lr 0.01 --customarg 1 --LAMB --lambda0 0 > run_sgdlamb.txt
/opt/conda/envs/py36/bin/python -u main2.py --dataset cifar --model cnn --epochs 100 --optim sgd --num_users 50 --gpu -1 \
--frac 0.5 --local_ep 3 --lr 0.01 --customarg 1 --LAMB --lambda0 0 > run_sgdlamb.txt





tar -czvf checkpoints.tar.gz checkpoints/

