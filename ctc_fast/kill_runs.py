import os
import shutil
from os.path import join as pjoin
import argparse
from run_cfg import RUN_DIR
from run_utils import get_run_dirs, file_alive, load_config
from cluster.utils import run_cpu_job

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', help='Directory containing runs', default=RUN_DIR)
    parser.add_argument('--clear_dirs', action='store_true', help='Clear run directory as well')
    args = parser.parse_args()

    run_dirs = get_run_dirs(args.run_dir)

    for d in run_dirs:
        alive = False
        params_file = pjoin(d, 'params.pk')
        cfg_file = pjoin(d, 'cfg.json')

        if not os.path.exists(cfg_file):
            # Definitely delete it
            shutil.rmtree(d)
            continue

        if os.path.exists(params_file):
            alive = file_alive(params_file)
        else:
            alive = file_alive(cfg_file)

        if not alive:
            run = os.path.basename(d)
            cfg = load_config(cfg_file)
            host = cfg['host']
            pid = cfg['pid']
            print 'Killing run %s, PID %s on %s' % (run, cfg['pid'], cfg['host'])
            run_cpu_job(host, 'kill -9 %s' % pid)

            if args.clear_dirs:
                print 'Clearing %s' % d
                shutil.rmtree(d)
