#!/bin/bash                                                                                                                         
###############################################################                                                                     
##                  注意-- 注意--注意                        ##                                                                     
##                 K8S 单机作业示例                          ##                                                                     
##                 请将下面的 ak/sk 替换成自己的 ak/sk              ##                                                              
###############################################################                                                                     
cur_time=`date  +"%Y%m%d%H%M"`
job_name=lamb-cifar-iid-${cur_time}


group_name="ccl-32g-0-yq01-k8s-gpu-v100-8"                   # 将作业提交到group_name指定的组，必填                                 
job_version="custom-framework"
#job_version="pytorch-1.4.0"                                                                                                        
start_cmd="sh job.sh"
k8s_gpu_cards=1
k8s_trainers=1
wall_time="1000:00:00"
k8s_priority="normal"
file_dir="."

image_addr=registry.baidu.com/lixu13/base_image:6.2
#image_addr=registry.baidu.com/lixu13/base_image:4.0                                                                                

# registry.baidu.com/lixu13/mcts_env:2.0                                                                                            

paddlecloud job train --job-name ${job_name} \
        --job-conf config.ini \
        --group-name ${group_name} \
        --start-cmd "${start_cmd}" \
        --file-dir ${file_dir} \
        --job-version ${job_version}  \
        --k8s-gpu-cards ${k8s_gpu_cards} \
        --k8s-priority ${k8s_priority} \
        --wall-time ${wall_time} \
        --is-standalone 1 \
        --image-addr ${image_addr} \