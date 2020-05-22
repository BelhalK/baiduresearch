#! /bin/bash

HGCP_CLIENR_BIN=/home/belhal/belhal/gpuserver/software-install/HGCP_client/bin

optim=$1

${HGCP_CLIENR_BIN}/submit \
        --hdfs hdfs://nmg01-mulan-hdfs.dmop.baidu.com:54310 \
        --hdfs-user idl-ml \
        --hdfs-passwd 1234567890 \
        --hdfs-path /app/idl/users/ml/lixu/v100_gpu_cluster/yujinxing-ptb-datasize-layer3-$optim \
        --file-dir ./ \
        --job-name bilstm-snli-$optim-lr20 \
        --submitter bkarimi \
        --queue-name yq01-gpu-86-74-13-00 \
        --num-nodes 1 \
        --num-task-pernode 1 \
        --gpu-pnode 1 \
        --time-limit 0 \
        --job-script ./run_$optim.sh
