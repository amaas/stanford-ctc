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
    print 'Parsed args'

    run_dirs = get_run_dirs(args.run_dir)

    for d in run_dirs:
        alive = False
        log_file = pjoin(d, 'train.log')
        cfg_file = pjoin(d, 'cfg.json')

        if not os.path.exists(cfg_file):
            # Definitely delete it
            shutil.rmtree(d)
            continue

        alive = file_alive(log_file, max_dur_sec=30*60)

        if not alive:
            run = os.path.basename(d)
            print 'loading config'
            print cfg_file
            cfg = load_config(cfg_file)
            print 'loaded config'
            host = cfg['host']
            pid = cfg['pid']
            print 'Killing run %s, PID %s on %s' % (run, cfg['pid'], cfg['host'])
            run_cpu_job(host, 'kill -9 %s' % pid)

            if args.clear_dirs:
                print 'Clearing %s' % d
                shutil.rmtree(d)
