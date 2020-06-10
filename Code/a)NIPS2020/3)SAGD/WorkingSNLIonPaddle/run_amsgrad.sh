export LD_LIBRARY_PATH=/home/work/cuda-9.0/lib64/:/home/work/cuda-9.0/lib/:/home/work/cuda-9.0/extras/CUPTI/lib64:/home/work/cudnn/\
cudnn_v7/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/home/work/cudnn/cudnn_v7/cuda/include:$CPATH
export LIBRARY_PATH=/home/work/cudnn/cudnn_v7/cuda/lib64:$LIBRARY_PATH
mkdir log
mkdir local_data
/opt/conda/envs/py36/bin/pip install torchtext
/opt/conda/envs/py36/bin/pip install dill
/opt/conda/envs/py36/bin/pip install time
/opt/conda/envs/py36/bin/pip install prettytable
/opt/conda/envs/py36/bin/pip install tqdm
/opt/conda/envs/py36/bin/pip install datetime

for lr in 0.001 0.002
do
/opt/conda/envs/py36/bin/python -u train.py --lr $lr --batch_size 256 --seed 141 --epoch 50 --repeat 3 --optimizer amsgrad
done

