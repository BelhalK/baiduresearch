#python3 test.py --batchsize=128 --nbepochs=3 --nbruns=1 --ntrains=50000 --lr=0.01
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import warnings
import os


# Dependency imports
import argparse
import numpy as np


IMAGE_SHAPE = [32, 32, 3]


ap = argparse.ArgumentParser()
ap.add_argument("-b", "--batchsize", type=int, default=1,help="")
ap.add_argument("-e", "--nbepochs", type=int, default=1,help="")
ap.add_argument("-r", "--nbruns", type=int, default=1,help="")
ap.add_argument("-n", "--ntrains", type=int, default=2000,help="")
ap.add_argument("-l", "--lr", type=float , default=0.001,help="")
args = vars(ap.parse_args())

print(args["lr"])