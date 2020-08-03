import subprocess
import os

print(os.getcwd())


#sgd
subprocess.call(['python3 ./run.py --num_epochs 10 --viz_steps 200 \
        --num_monte_carlo 5 --optimizer sgd \
             --start_avg 10 --avg_period 10'], shell=True)


#HWA (sgd or adam)
for avg in [10, 50, 100]:
    subprocess.call(['python3 ./run.py --num_epochs 10 --viz_steps 200 \
        --num_monte_carlo 5 --optimizer hwa_sgd \
             --start_avg 10 --avg_period {}'.format(avg)], shell=True)