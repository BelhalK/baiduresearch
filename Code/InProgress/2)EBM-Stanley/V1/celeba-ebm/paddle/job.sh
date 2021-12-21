export LD_LIBRARY_PATH=/home/work/cuda-9.0/lib64/:/home/work/cuda-9.0/lib/:/home/work/cuda-9.0/extras/CUPTI/lib64:/home/work/cudnn/\
cudnn_v7/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/home/work/cudnn/cudnn_v7/cuda/include:$CPATH
export LIBRARY_PATH=/home/work/cudnn/cudnn_v7/cuda/lib64:$LIBRARY_PATH
mkdir log
mkdir local_data



/opt/conda/envs/py36/bin/python -u run_test_anisotropic.py --eps 0.01 --th 0.0001 > ebm_anila_log_1.txt
/opt/conda/envs/py36/bin/python -u run_test_anisotropic.py --eps 0.01 --th 0.0002 > ebm_anila_log_2.txt
/opt/conda/envs/py36/bin/python -u run_test_anisotropic.py --eps 0.01 --th 0.00008 > ebm_anila_log_3.txt

tar -czvf alloutputs.tar.gz alloutputs/