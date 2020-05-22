sh env.sh
for lr in 0.1 0.5
do
python3 -u train.py --lr $lr --batch_size 20 --seed 141 --epoch 500 --repeat 5 --optimizer adagrad
done
