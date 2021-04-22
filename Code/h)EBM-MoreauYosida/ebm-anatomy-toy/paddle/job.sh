export LD_LIBRARY_PATH=/home/work/cuda-9.0/lib64/:/home/work/cuda-9.0/lib/:/home/work/cuda-9.0/extras/CUPTI/lib64:/home/work/cudnn/\
cudnn_v7/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/home/work/cudnn/cudnn_v7/cuda/include:$CPATH
export LIBRARY_PATH=/home/work/cudnn/cudnn_v7/cuda/lib64:$LIBRARY_PATH
mkdir log
mkdir local_data


/opt/conda/envs/py36/bin/pip install matplotlib
/opt/conda/envs/py36/bin/pip install scipy

/opt/conda/envs/py36/bin/python -u train_toy.py --th 0.05 --eps 0.1 --mcmcmethod anilangevin > log_ani.txt
/opt/conda/envs/py36/bin/python -u train_toy.py --eps 0.1 --mcmcmethod langevin > log_langevin.txt

tar -czvf out_toy.tar.gz out_toy/
sshpass -p "Afghanistan15$" scp out_toy.tar.gz belhal@yq01-gpu-86-74-13-00.epc.baidu.com:/home/belhal/belhal/
