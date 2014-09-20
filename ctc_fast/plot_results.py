import sys
import argparse
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os
from os.path import join as pjoin

# PARAM
WINDOW_SIZE = 10
WINDOW_RATIO = 10.0

'''
Inspect and plot results from run
'''

def smooth_arr(arr, win_size):
    window = np.ones(int(win_size)) / float(win_size)
    return np.convolve(arr, window, 'valid')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot run results')
    parser.add_argument('run_dir', help='Directory containing run outputs')
    parser.add_argument('--out_file', help='Save to specified output image file')
    args = parser.parse_args()
    params_file = pjoin(args.run_dir, 'params.pk')

    with open(params_file, 'rb') as fh:
        [it, costt, expcost, _] = pickle.load(fh)

    # dvipng not installed which causes issue
    #rc('text', usetex=True)
    rc('font', family='serif')

    win = min(WINDOW_SIZE, len(costt) / WINDOW_RATIO)

    if len(costt) == 0 or win == 0:
        print args.run_dir, 'does not have enough data to plot'
        sys.exit(0)

    plt.figure()
    subplots = list()

    subplots.append(plt.subplot(2, 1, 1))
    cost = np.array(costt)
    p1, = plt.plot(cost, 'm-.')

    plt.title(os.path.basename(args.run_dir))

    subplots.append(plt.subplot(2, 1, 2))
    ecost = np.array(expcost)
    p2, = plt.plot(ecost, 'c--')

    plt.gcf().legend([p1, p2], ['cost', 'expcost'])


    # TODO Plot CER and WER

    for sp in subplots:
        sp.tick_params(axis='both', which='major', labelsize=9)
        sp.tick_params(axis='both', which='major', labelsize=9)

    if args.out_file:
        plt.savefig(args.out_file, bbox_inches='tight')
    else:
        plt.show()
