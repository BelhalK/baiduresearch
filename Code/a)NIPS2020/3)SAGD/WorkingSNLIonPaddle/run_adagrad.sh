export LD_LIBRARY_PATH=/home/work/cuda-9.0/lib64/:/home/work/cuda-9.0/lib/:/home/work/cuda-9.0/extras/CUPTI/lib64:/home/work/cudnn/\
cudnn_v7/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/home/work/cudnn/cudnn_v7/cuda/include:$CPATH
export LIBRARY_PATH=/home/work/cudnn/cudnn_v7/cuda/lib64:$LIBRARY_PATH
mkdir log
mkdir local_data
tar -zxvf `pwd`/v_karimibelhal/final_paddle_sagd.tar.gz -C `pwd`/local_data/ > log/log_tar.txt
ls `pwd`/data/snli > log/log_snlidata.txt
/opt/conda/envs/py36/bin/pip install torchtext
/opt/conda/envs/py36/bin/pip install dill
/opt/conda/envs/py36/bin/pip install time
/opt/conda/envs/py36/bin/pip install prettytable
/opt/conda/envs/py36/bin/pip install tqdm
/opt/conda/envs/py36/bin/pip install datetime


for lr in 0.1 0.5
do
/opt/conda/envs/py36/bin/python -u train.py --lr $lr --batch_size 256 --seed 141 --epoch 50 --repeat 3 --optimizer adagrad
done

