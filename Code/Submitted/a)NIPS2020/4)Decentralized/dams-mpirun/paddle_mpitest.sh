#!/bin/bash
###############################################################
##                  注意--注意--注意                         ##
##                  pytorch NCCL2多机作业演示             ##
###############################################################
cur_time=`date  +"%Y%m%d%H%M"`
job_name=dams_job${cur_time}
 
# 作业参数
group_name="ccl-32g-0-yq01-k8s-gpu-v100-8"                   # 将作业提交到group_name指定的组，必填                                 
job_version="custom-framework"
start_cmd="sh job.sh"
wall_time="2:00:00"
file_dir="."
k8s_gpu_cards=1
k8s_priority="normal"
k8s_trainers=2
image_addr=registry.baidu.com/lixu13/base_image:6.2


paddlecloud job train --job-name ${job_name} \
        --group-name ${group_name} \
        --job-conf config.ini \
        --start-cmd "${start_cmd}" \
        --file-dir ${file_dir} \
        --job-version ${job_version}  \
        --k8s-trainers ${k8s_trainers} \
        --k8s-gpu-cards ${k8s_gpu_cards} \
        --k8s-priority ${k8s_priority} \
        --wall-time ${wall_time} \
        --is-standalone 0 \
        --image-addr ${image_addr} \
        --distribute-job-type "NCCL2"