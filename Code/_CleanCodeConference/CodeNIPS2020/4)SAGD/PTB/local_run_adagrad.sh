python -u main.py --lr 0.01 --batch_size 20 --data data/penn \
--dropouti 0.4 --dropouth 0.25 --wdrop 0 --seed 141 --epoch 500 --save output/PTB.pt \
--optimizer adagrad --device $1
