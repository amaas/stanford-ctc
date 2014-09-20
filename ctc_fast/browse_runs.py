import os
import json
import argparse
import joblib
import shutil
from time import strftime
from os.path import join as pjoin
from os.path import exists as fexists
from run_cfg import RUN_DIR, VIEWER_DIR, BROWSE_RUNS_KEYS
from run_utils import file_alive, last_modified, get_run_dirs
from cluster.utils import NUM_CPUS
from subprocess import check_call


def read_cfg(cfg_file, run_data):
    with open(cfg_file, 'r') as fin:
        cfg = json.load(fin)
    for key in cfg.keys():
        run_data[key] = cfg[key]


def process_run_dir(run_dir, figs=False):
    print run_dir
    run_data = dict()

    # Config file
    cfg_file = pjoin(run_dir, 'cfg.json')
    if not fexists(cfg_file):
        print 'No config file in %s' % run_dir
        return

    # Get epoch
    epoch_file = pjoin(run_dir, 'epoch')
    if os.path.exists(epoch_file):
        epoch = int(open(epoch_file, 'r').read().strip())
    else:
        epoch = -1
    run_data['epoch'] = epoch

    # Alive / not
    params_file = pjoin(run_dir, 'params.pk')
    if os.path.exists(params_file):
        run_data['alive'] = file_alive(params_file)
    else:
        run_data['alive'] = file_alive(cfg_file)

    # Complete / not
    run_data['complete'] = os.path.exists(pjoin(run_dir, 'sentinel'))

    if run_data['complete']:
        run_data['alive'] = "<span style='background:#ccc;'>False</span>"
    elif run_data['alive']:
        run_data['alive'] = "<span style='background:#6d6;color:#fff'>True</span>"
    else:
        run_data['alive'] = "<span style='background:#d66;color:#fff'>False</span>"

    run_data['run'] = os.path.basename(run_dir)

    read_cfg(cfg_file, run_data)

    # TODO Load CER and WER

    if figs and os.path.exists(pjoin(run_dir, 'params.pk')):
        plot_file = pjoin(run_dir, 'plot.png')
        cmd = 'python plot_results.py %s --out_file %s' % (run_dir, plot_file)

        # Check if params file has been modified after the plot image file
        if (not os.path.exists(plot_file)) or (last_modified(plot_file) < last_modified(params_file)):
            print '%s modified, generating plot' % params_file
            try:
                check_call(cmd, shell=True)
            except:
                pass

        if args.viewer_dir:
            plot_dir = pjoin(args.viewer_dir, 'plots')
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            if os.path.exists(pjoin(run_dir, 'plot.png')):
                shutil.copyfile(pjoin(run_dir, 'plot.png'),
                        pjoin(plot_dir, '%s.png' % run_data['run']))

    return run_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', help='Directory containing runs', default=RUN_DIR)
    parser.add_argument('--viewer_dir', help='Directory to write webpage', default=VIEWER_DIR)
    parser.add_argument('--figs', action='store_true', default=False, help='Generate figures')
    args = parser.parse_args()

    # Get run directories

    run_dirs = get_run_dirs(args.run_dir)

    data = joblib.Parallel(n_jobs=NUM_CPUS)(joblib.delayed(process_run_dir)(run_dir, figs=args.figs) for run_dir in run_dirs)
    if None in data:
        ind = data.index(None)
        print 'Got None from process_run_dir, %s, trying again later...' % run_dirs[ind]
    while None in data:
        data.remove(None)

    # Output data in easily readable table

    keys = BROWSE_RUNS_KEYS

    if args.viewer_dir:
        for f in ['runs.html', 'viewer.css', 'viewer.js', 'jquery.tablesorter.min.js']:
            src = pjoin('viewer', f)
            dst = pjoin(args.viewer_dir, f)
            print 'Copying %s to %s' % (src, dst)
            shutil.copyfile(src, dst)
        json_data = {}
        json_data['keys'] = keys
        json_data['runs'] = data
        json_data['time'] = strftime('%Y-%m-%d %H:%M:%S')
        json_data['figs'] = args.figs
        with open(pjoin(args.viewer_dir, 'data.json'), 'w') as fout:
            json.dump(json_data, fout)
