sh env.sh
model=vggnet
cudaId=0
for decay in True False
do
  for lr in 0.01 0.05
  do
    for noi in 0.01
    do
      python/bin/python3 -u run_nn_cifar10.py --model $model --cuda-id $cudaId --lr $lr --noise-coe $noi --batch-size 128 --epochs 100 --repeat 3  --LR-decay $decay --decay-epoch 30 --optim sagd_sparse
    done
  done
done
