sh env.sh
model=vggnet
cudaId=0
for decay in True False
do
  for lr in 0.005 0.002 0.009
  do
    python/bin/python3 -u run_nn_cifar10.py --model $model --cuda-id $cudaId --lr $lr --batch-size 128 --epochs 100 --repeat 3  --LR-decay $decay --decay-epoch 30 --optim adabound
  done
done
