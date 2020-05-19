sh env.sh
# sh run_sgd.sh "cuda:0" > logs/sgd.txt &
sh run_adagrad.sh "cuda:0" > logs/adagrad.txt &
sh run_adabound.sh "cuda:1" > logs/adabound.txt &
#sh run_adam.sh "cuda:2" > logs/adam.txt &
sh run_padam.sh "cuda:2" > logs/padam.txt &
#sh run_rmsprop.sh "cuda:4" > logs/rmsprop.txt &