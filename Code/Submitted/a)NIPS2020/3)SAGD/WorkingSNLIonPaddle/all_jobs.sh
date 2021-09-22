#! /bin/bash

# "SARMSprop" "SARMSprop_sparse"
# "adam"
# "sagd"
for optim in "adagrad" "adabound" "amsgrad" "adam" "sagd"
do
 (sh run_$optim.sh  && sleep 3s) || break
done
