sh env.sh
for lr in 0.1 0.5
do
/home/belhal/belhal/baiduenv/bin/python3 -u train.py --lr $lr --batch_size 20 --seed 141 --epoch 100 --repeat 2 --optimizer adagrad
done