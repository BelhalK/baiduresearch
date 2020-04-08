from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

# Dependency imports
from absl import flags
import matplotlib.pyplot as plt
import pylab
import numpy as np
import pickle 


warnings.simplefilter(action="ignore")

IMAGE_SHAPE = [32, 32, 3]

loss_dir = 'losses/'

##LOAD LOSSES
with open (loss_dir+'adam', 'rb') as fp:
    adam = pickle.load(fp)
with open (loss_dir+'adagrad', 'rb') as fp:
    adagrad = pickle.load(fp)
with open (loss_dir+'adadelta', 'rb') as fp:
    adadelta = pickle.load(fp)
with open (loss_dir+'rmsprop', 'rb') as fp:
    misso = pickle.load(fp)


def tsplotseveral(x, y,figsize,leg_fontsize, axis_fontsize, n=20, percentile_min=1, percentile_max=99, color='r', plot_mean=True, plot_median=False, line_color='k', **kwargs):
    line_colors=['y','b','g','r','black']
    #colors=['y','b','g','r','black']
    colors=['lightyellow','lightskyblue','palegreen','mistyrose','lightgrey']
    labels= ['MC-ADAM','MC-Adagrad','MC-Adadelta','MISSO','MC-SAG']
    i = 0
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    axes.set_facecolor('white')
    axes.grid(linestyle='-', linewidth='0.2', color='grey')
    axes.spines['bottom'].set_color('black')
    axes.spines['top'].set_color('black') 
    axes.spines['right'].set_color('black')
    axes.spines['left'].set_color('black')
    
    for element in y:
      perc1 = np.percentile(element, np.linspace(percentile_min, 50, num=n, endpoint=False), axis=0)
      perc2 = np.percentile(element, np.linspace(50, percentile_max, num=n+1)[1:], axis=0)

      # fill lower and upper percentile groups
      for p1, p2 in zip(perc1, perc2):
          plt.fill_between(x, p1, p2, color=colors[i], edgecolor=None,alpha=0.002)

      if plot_mean:
          plt.plot(x, np.mean(element, axis=0), color=line_colors[i],label=labels[i])


      if plot_median:
          plt.plot(x, np.median(element, axis=0), color=line_colors[i],label=labels[i])
      i += 1
    leg = plt.legend(fontsize=leg_fontsize,fancybox=True, loc=0,ncol=3)
    leg.get_frame().set_alpha(0.5)
    plt.xlabel('Epoch', fontsize=axis_fontsize)
    plt.ylabel('ELBO', fontsize=axis_fontsize)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(linestyle='dotted',linewidth=2)
    pylab.ticklabel_format(axis='y',style='sci',scilimits=(1,4))
    fig.tight_layout()
    #fig.show()
    fig.savefig("missoresnet.png",bbox_inches = 'tight')
    return i

iterations = len(adagrad[0])
itera = np.linspace(0,iterations,iterations)
figsize = (8, 4.5)
leg_fontsize= 16
axis_fontsize = 18
toplot = [adam, adagrad, adadelta, misso]
tsplotseveral(itera,toplot,figsize,leg_fontsize, axis_fontsize,
              n=100, percentile_min=2.5, percentile_max=97.5, plot_median=True, plot_mean=False, color='g', line_color='navy')
