import subprocess
import os

print(os.getcwd())

python3 run.py 

#HWA (sgd or adam)
for avg in [10, 50, 100]:
    subprocess.call(['python3 ./run.py --num_epochs 2 --viz_steps 200 \
        --num_monte_carlo 5 --optimizer hwa_sgd \
             --start_avg 10 --avg_period {}'.format(avg)], shell=True)
    subprocess.call(['python3 ./run.py --num_epochs 2 --viz_steps 200 \
        --num_monte_carlo 5 --optimizer hwa \
             --start_avg 10 --avg_period {}'.format(avg)], shell=True)

#sgd
subprocess.call(['python3 ./run.py --num_epochs 2 --viz_steps 200 \
        --num_monte_carlo 5 --optimizer sgd \
             --start_avg 10 --avg_period 10'], shell=True)

#Adam
subprocess.call(['python3 ./run.py --num_epochs 2 --viz_steps 200 \
        --num_monte_carlo 5 --optimizer adam \
             --start_avg 10 --avg_period 10'], shell=True)