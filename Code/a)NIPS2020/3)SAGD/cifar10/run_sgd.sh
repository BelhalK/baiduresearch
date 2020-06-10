sh env.sh
model=vggnet
cudaId=0
for decay in  True False
do
  for lr in  0.1 0.01
  do
    for m in 0 0.1
    do
      python/bin/python3 -u run_nn_cifar10.py --model $model --cuda-id $cudaId --lr $lr --batch-size 128 --epochs 100 --repeat 3 --momentum $m --LR-decay $decay --decay-epoch 30 --optim sgd
    done
  done
done
