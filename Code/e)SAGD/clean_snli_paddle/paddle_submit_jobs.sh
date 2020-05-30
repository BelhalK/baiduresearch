#!/bin/bash
###############################################################
##                  注意-- 注意--注意                        ##
##                 K8S 单机作业示例                          ##
##                 请将下面的 ak/sk 替换成自己的 ak/sk              ##
###############################################################
cur_time=`date  +"%Y%m%d%H%M"`
job_name=snli-bilstm-all-${cur_time}

# 作业参数
group_name="ccl-32g-0-yq01-k8s-gpu-v100-8"                   # 将作业提交到group_name指定的组，必填
job_version="paddle-fluid-v1.5.2"
start_cmd="sh paddle_all_jobs.sh"
k8s_gpu_cards=1
wall_time="10:00:00"
k8s_priority="normal"
file_dir="."

# 你的ak/sk（可在paddlecloud web页面【个人中心】处获取）
ak=0f3cb66642345872a9f2b05c8cd9dfc9
sk=b5f8ba5a1918555e8176255b40b770e9

paddlecloud job --ak ${ak} --sk ${sk} \
        train --job-name ${job_name} \
        --job-conf config.ini \
        --group-name ${group_name} \
        --start-cmd "${start_cmd}" \
        --file-dir ${file_dir} \
        --job-version ${job_version}  \
        --k8s-gpu-cards ${k8s_gpu_cards} \
        --k8s-priority ${k8s_priority} \
        --wall-time ${wall_time} \
        --is-standalone 1