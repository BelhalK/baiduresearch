sh env.sh
for lr in 0.1 0.5
do
python/bin/python3 -u main.py --lr $lr --batch_size 20 --data data/penn \
--dropouti 0.4 --dropouth 0.25 --wdrop 0.5 --seed 141 --epoch 500 --save output/PTB.pt \
--optimizer adagrad --device "cuda:0" --disable_asgd
done
