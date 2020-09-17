export LD_LIBRARY_PATH=/home/work/cuda-9.0/lib64/:/home/work/cuda-9.0/lib/:/home/work/cuda-9.0/extras/CUPTI/lib64:/home/work/cudnn/\
cudnn_v7/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/home/work/cudnn/cudnn_v7/cuda/include:$CPATH
export LIBRARY_PATH=/home/work/cudnn/cudnn_v7/cuda/lib64:$LIBRARY_PATH
mkdir log
mkdir local_data

/opt/conda/envs/py36/bin/pip install blitz
/opt/conda/envs/py36/bin/pip install matplotlib


for lr in 0.001 0.01 0.1
do
/opt/conda/envs/py36/bin/python -u cifar10_bvgg.py --optimizer sgd --epochs 100 --lr lr > runingsgd_log.txt
/opt/conda/envs/py36/bin/python -u cifar10_bvgg.py --optimizer hwa --epochs 100 --lr lr > runinghwa_log.txt
done