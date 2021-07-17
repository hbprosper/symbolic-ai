#!/usr/bin/env python
#------------------------------------------------------------------------------
# Real time monitoring of loss curves during training
# Harrison B. Prosper
# July 2021
#------------------------------------------------------------------------------
import os, sys
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#------------------------------------------------------------------------------
DELAY = 10 # seconds - interval between plot updates
#------------------------------------------------------------------------------
# The loss file should be a simple text file with two columns of numbers:
#
#   train-losses validation-losses
#     
def get_losses(loss_file):
    losses       = [a.split() for a in open(loss_file).readlines()]
    train_losses = [float(z)  for z, _ in losses]
    valid_losses = [float(z)  for _, z in losses]
    epochs       = np.arange(len(train_losses))
    return epochs, train_losses, valid_losses
#------------------------------------------------------------------------------
# get name of loss file
#
argv = sys.argv[1:]
argc = len(argv)
if argc < 1:
    sys.exit('''
    Usage:
       ./monitor_losses.py loss-file-name
''')
loss_file = argv[0]
if not os.path.exists(loss_file):
    sys.exit("\n\t** file %s not found\n" % loss_file)
#------------------------------------------------------------------------------
# set up an empty figure
fig = plt.figure(figsize=(6, 4))

# add a subplot to it
nrows, ncols, index = 1,1,1
ax  = fig.add_subplot(nrows, ncols, index)
#------------------------------------------------------------------------------
def plot_losses():
    epochs, train_losses, valid_losses = get_losses(loss_file)
    epoch = len(epochs)
    
    ax.clear()
    ax.set_title('epoch: %5d | %s' % (epoch, time.ctime()))
    
    ax.plot(epochs, train_losses, c='red',  label='training')
    ax.plot(epochs, valid_losses, c='blue', label='validation')
    ax.set_xlabel('epoch', fontsize=16)
    ax.set_ylabel('$\overline{loss}$', fontsize=16) 
    ax.grid(True, which="both", linestyle='-')
    ax.legend()
    
    fig.tight_layout()

def animate(i):
    plot_losses()

plot_losses()

interval = 1000 * DELAY # milliseconds

ani = FuncAnimation(fig, animate, interval=interval, repeat=False)
 
plt.show()

