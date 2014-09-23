import os
import time
import argparse
import string
from run_cfg import RUN_DIR, CTC_DIR
from os.path import join as pjoin
from run_utils import TimeString, file_alive, get_run_dirs, load_config
from cluster.utils import run_gpu_job, get_all_free_gpus, get_next_free_gpu_sequential
from cluster.config import PYTHON_CMD, SLEEP_SEC, FLAGGED_GPUS


def reboot_run(run_dir, used_gpus):
    cfg_file = pjoin(run_dir, 'cfg.json')

    # Read in cluster we should be using
    cfg = load_config(cfg_file)
    cluster = ''.join(c for c in cfg['host'] if not c.isdigit())

    run_args = '--cfg_file %s' % cfg_file
    cmd = 'cd %s; source ~/.bashrc; nohup %s runNNet.py %s' % (CTC_DIR, PYTHON_CMD, run_args)
    print cmd

    gpu_node = None
    while not gpu_node:
        all_free_gpus = get_all_free_gpus(cluster)
        print all_free_gpus
        gpu_node, gpu = get_next_free_gpu_sequential(all_free_gpus, used_gpus, FLAGGED_GPUS)
        if not gpu_node:
            print 'No free GPUs, waiting for a bit'
            time.sleep(SLEEP_SEC)

    # Log to file in for debugging
    log_file = pjoin(RUN_DIR, '%s.txt' % str(TimeString()))
    print 'Logging to %s' % log_file
    run_gpu_job(gpu_node, gpu, cmd, blocking=False,
            stdout=open(log_file, 'w'))

    used_gpus.add(gpu_node + '_' + str(gpu))

    time.sleep(SLEEP_SEC)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', help='Directory containing runs', default=RUN_DIR)
    args = parser.parse_args()

    # Get run directories

    run_dirs = get_run_dirs(args.run_dir)

    used_gpus = set()
    for d in run_dirs:
        alive = True
        log_file = pjoin(d, 'train.log')

        # Check if alive

        alive = file_alive(log_file, max_dur_sec=30*60)

        if not alive:
            reboot_run(d, used_gpus)
            time.sleep(SLEEP_SEC)
