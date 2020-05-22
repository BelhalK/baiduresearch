sh env.sh
for lr in 0.001 0.002
do
python3 -u train.py --lr $lr --batch_size 20 --seed 141 --epoch 500 --repeat 5 --optimizer amsgrad
done