sh env.sh

sh run_adabound.sh vggnet 0 &
sh run_adagrad.sh vggnet 1 &
sh run_adam.sh vggnet 2 &
sh run_padam.sh vggnet 3 &
sh run_rmsprop.sh vggnet 4 &
sh run_sgd.sh vggnet 5 &

#sh run_sagd_sparse.sh vggnet 0 &
#sh run_sagd.sh vggnet 1 &
#sh run_SARMSprop_sparse.sh vggnet 2 &
#sh run_SARMSprop.sh vggnet 3 &
