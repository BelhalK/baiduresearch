#export CUDA_VISIBLE_DEVICES=2
sh env.sh
for lr in 30 20:
do
python/bin/python3 -u main.py --batch_size 20 --lr $lr --data data/penn \
--dropouti 0.4 --dropouth 0.25 --wdrop 0.5 --seed 141 --epoch 500 --save output/PTB.pt \
--optimizer sgd --device "cuda:0" --disable_asgd
done