export LD_LIBRARY_PATH=/home/work/cuda-9.0/lib64/:/home/work/cuda-9.0/lib/:/home/work/cuda-9.0/extras/CUPTI/lib64:/home/work/cudnn/\
cudnn_v7/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/home/work/cudnn/cudnn_v7/cuda/include:$CPATH
export LIBRARY_PATH=/home/work/cudnn/cudnn_v7/cuda/lib64:$LIBRARY_PATH
mkdir log
mkdir local_data


/opt/conda/envs/py36/bin/pip install matplotlib
/opt/conda/envs/py36/bin/pip install scipy

/opt/conda/envs/py36/bin/python -u train_data.py --th 0.0002 --eps 0.01 --mcmcmethod anilangevin > log_ani.txt
/opt/conda/envs/py36/bin/python -u train_data.py --th 0.0003 --eps 0.01 --mcmcmethod anilangevin > log_ani.txt



tar -czvf out_data.tar.gz out_data/
