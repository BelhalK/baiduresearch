#! /bin/bash

# "SARMSprop" "SARMSprop_sparse"
# "adam"
# "sagd"
#"sgd" "adagrad" "padam" "adabound" "rmsprop" "sagd_sparse"
for optim in "SARMSprop" "SARMSprop_sparse"
do
 (sh submit.sh $optim && sleep 3s) || break
done
