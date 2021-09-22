#export CUDA_VISIBLE_DEVICES=2
python -u main.py --batch_size 20 --data data/penn \
--dropouti 0.4 --dropouth 0.25 --wdrop 0.5 --seed 141 --epoch 5 --save output/PTB.pt \
--optimizer sgd --device $1 --disable_asgd
