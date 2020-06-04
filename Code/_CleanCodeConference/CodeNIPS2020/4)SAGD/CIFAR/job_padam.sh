#!/bin/bash
echo "==============JOB BEGIN============"

# /home/HGCP_Program/software-install/openmpi-1.8.5/bin/mpirun mkdir ./workspace/
# /home/HGCP_Program/software-install/openmpi-1.8.5/bin/mpirun hadoop fs -get /app/idl/users/ml/lixu/p40_gpu_cluster/dual_environments/ ./workspace/
# /home/HGCP_Program/software-install/openmpi-1.8.5/bin/mpirun hadoop fs -get /app/idl/users/ml/lixu/p40_gpu_cluster/ltp_data_v3.4.0/ ./
# /home/HGCP_Program/software-install/openmpi-1.8.5/bin/mpirun -bind-to none sh run.sh
# /home/HGCP_Program/software-install/openmpi-1.8.5/bin/mpirun -bind-to none sh run_mcts_coco.sh
#/home/HGCP_Program/software-install/openmpi-1.8.5/bin/mpirun -bind-to none sh run_emnlp.sh
# /home/HGCP_Program/software-install/openmpi-1.8.5/bin/mpirun ./hadoop-client-idl-ml/hadoop/bin/hadoop fs -get /app/idl/users/ml/zhouxin/train_data_v2_kg51.json ./san_kg/kg_data/
# /home/HGCP_Program/software-install/openmpi-1.8.5/bin/mpirun ./hadoop-client-idl-ml/hadoop/bin/hadoop fs -get /app/idl/users/ml/zhouxin/dev_data_v2_kg51.json ./san_kg/kg_data/
# /home/HGCP_Program/software-install/openmpi-1.8.5/bin/mpirun ./hadoop-client-idl-ml/hadoop/bin/hadoop fs -get /app/idl/users/ml/zhouxin/meta_v2_kg51.pick  ./san_kg/kg_data/

sh /home/HGCP_Program/software-install/afs_mount/bin/afs_mount.sh MAP_KM_Data MAP_km_2018 `pwd`/data afs://xingtian.afs.baidu.com:9902/user/MAP_KM_Data/yjx/datas/cifar10
ls `pwd`/data > logs/log_data.txt

sh /home/HGCP_Program/software-install/afs_mount/bin/afs_mount.sh MAP_KM_Data MAP_km_2018 `pwd`/pythons afs://xingtian.afs.baidu.com:9902/user/MAP_KM_Data/yjx/pythons
unzip -d `pwd`/ `pwd`/pythons/python.zip > logs/unzip.txt
ls `pwd`/python/bin > logs/log_python.txt

/home/HGCP_Program/software-install/openmpi-1.8.5/bin/mpirun -bind-to none sh run_padam.sh
# /home/HGCP_Program/software-install/openmpi-1.8.5/bin/mpirun -bind-to none sh run_mcts_coco_bleu_reward.sh

echo "===============JOB END============="
