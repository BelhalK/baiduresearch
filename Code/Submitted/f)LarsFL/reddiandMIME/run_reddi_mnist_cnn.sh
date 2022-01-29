for lr in {0.1,0.05,0.01,0.005,0.001,0.0005,0.0001}
do

for glob_lr in {0.1,0.05,0.01,0.005,0.001,0.0005,0.0001}
do

for s in {0,1,2}
do

CUDA_VISIBLE_DEVICES=1 python main_reddi.py --dataset mnist --model cnn --num_channels 1 --optim reddi --local_ep 1 --local_bs 128 --epochs 50 --num_users 50 --glob_lr $glob_lr --lr $lr --LAMB 0 --customarg 0 --gpu 0 --seed $s


done
done
done

