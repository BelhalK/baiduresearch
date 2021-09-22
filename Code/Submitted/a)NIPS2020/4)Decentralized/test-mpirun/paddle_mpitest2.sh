#!/bin/bash
###############################################################
##                  注意--注意--注意                         ##
##                  pytorch NCCL2多机作业演示             ##
###############################################################
cur_time=`date  +"%Y%m%d%H%M"`
job_name=demo_job_2_${cur_time}
 
# 作业参数
group_name="ccl-32g-0-yq01-k8s-gpu-v100-8"
job_version="pytorch-1.4.0"
start_cmd="sh job.sh"
wall_time="2:00:00"
file_dir="."
k8s_gpu_cards=1
k8s_priority="normal"
k8s_trainers=2
 
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
        --distribute-job-type "NCCL2"