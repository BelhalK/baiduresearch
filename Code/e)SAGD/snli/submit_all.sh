#! /bin/bash

# "SARMSprop" "SARMSprop_sparse"
# "adam"
# "sagd"
for optim in "sgd" "adagrad" "adabound" "amsgrad" "sagd"
do
 (sh submit.sh $optim && sleep 3s) || break
done
