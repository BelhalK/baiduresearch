sh env.sh
sh run_sagd.sh "cuda:0" > logs/sagd.txt &
#sh run_SARMSprop.sh "cuda:0" > logs/sarmsprop.txt &
sh run_sagd_sparse.sh "cuda:1" > logs/sagd_sparse.txt &
#sh run_SARMSprop_sparse.sh "cuda:1" > logs/sarmsprop_sparse.txt &
