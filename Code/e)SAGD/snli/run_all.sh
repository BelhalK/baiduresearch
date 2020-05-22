sh env.sh
sh run_adagrad.sh "cuda:1" > logs/adagrad.txt &
sh run_adabound.sh "cuda:2" > logs/adabound.txt &
sh run_adam.sh "cuda:3" > logs/adam.txt &
sh run_sagd.sh "cuda:4" > logs/sagd.txt &
sh run_sgd.sh "cuda:5" > logs/sgd.txt &
sh run_amsgrad.sh "cuda:6" > logs/amsgrad.txt &
