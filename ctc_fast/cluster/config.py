import os
import multiprocessing

NUM_CPUS = multiprocessing.cpu_count() - 1

CLUSTER_DIR = os.path.dirname(os.path.abspath(__file__))

CPU_FREE_TOL = 0.3
RAM_FREE_TOL = 0.1

SSH_CMD = 'ssh -q -x -o ConnectTimeout=10 -o ServerAliveInterval=3'
PYTHON_CMD = '/afs/cs.stanford.edu/u/zxie/virtualenvs/scl/bin/python'

CLUSTER_NODES = {
    'gorgon': range(40, 70),
    # deep3, deep5, deep13, deep15, deep23
    #'deep': list(range(1, 3)) + list(range(6, 13)) + list(range(16, 23))
    'deep': list(range(1, 6)) + list(range(7, 25))
}

SLEEP_SEC = 30

GPU_FREE_TOL = 0.01

FLAGGED_GPUS = ['gorgon18_0']
