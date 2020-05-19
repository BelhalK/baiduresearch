#! /bin/bash

HGCP_CLIENR_BIN=/home/yujinxing/gpuserver/software-install/HGCP_client/bin

optim=$1

${HGCP_CLIENR_BIN}/submit \
        --hdfs hdfs://nmg01-mulan-hdfs.dmop.baidu.com:54310 \
        --hdfs-user idl-ml \
        --hdfs-passwd 1234567890 \
        --hdfs-path /app/idl/users/ml/lixu/v100_gpu_cluster/yujinxing-vgg19-$optim \
        --file-dir ./ \
        --job-name cifar10_vgg19_$optim \
        --submitter yujinxing \
        --queue-name yq01-v100-box-2-8 \
        --num-nodes 1 \
        --num-task-pernode 1 \
        --gpu-pnode 1 \
        --time-limit 0 \
        --job-script ./job_$optim.sh
