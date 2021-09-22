sh env.sh
# sh run_sgd.sh "cuda:0" > logs/sgd.txt &
sh run_adagrad.sh "cuda:1" > logs/adagrad.txt &
sh run_adabound.sh "cuda:2" > logs/adabound.txt &
sh run_adam.sh "cuda:3" > logs/adam.txt &
sh run_padam.sh "cuda:4" > logs/padam.txt &
sh run_rmsprop.sh "cuda:5" > logs/rmsprop.txt &
sh run_sagd.sh "cuda:6" > logs/sagd.txt &
sh run_SARMSprop.sh "cuda:7" > logs/sarmsprop.txt &
sh run_sagd_sparse.sh "cuda:0" > logs/sagd_sparse.txt &
sh run_SARMSprop_sparse.sh "cuda:1" > logs/sarmsprop_sparse.txt &
