for optim in "adabound" "adagrad" "adam" "padam" "rmsprop" "sagd_sparse" "sagd" "SARMSprop_sparse" "SARMSprop" 
#"sgd"
do
    (sh submit.sh $optim && sleep 3s) || break
done
