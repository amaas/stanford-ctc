import argparse
from os.path import join as pjoin
from run_cfg import CTC_DIR, TRAIN_DATA_DIR, RUN_DIR
from run_utils import TimeString


PARAM_SETTING_LIST = [
    #{'layerSize': 1824, 'inputDim': 21*15, 'run_desc': 'fbank_context'},
    #{'layerSize': 1824, 'inputDim': 31*15, 'run_desc': 'fbank_context'},
    #{'layerSize': 1824, 'inputDim': 41*15, 'run_desc': 'fbank_context'},
    #{'layerSize': 1824, 'inputDim': 11*13, 'rawDim': 21*13, 'run_desc': 'mfcc_context', 'dataDir': TRAIN_DATA_DIR['mfcc']},
    #{'layerSize': 1824, 'inputDim': 15*13, 'rawDim': 21*13, 'run_desc': 'mfcc_context', 'dataDir': TRAIN_DATA_DIR['mfcc']},
    #{'layerSize': 1824, 'inputDim': 21*13, 'rawDim': 21*13, 'run_desc': 'mfcc_context', 'dataDir': TRAIN_DATA_DIR['mfcc']},
    #{'layerSize': 1824, 'inputDim': 15*39, 'rawDim': 41*39, 'run_desc': 'mfcc_deltas', 'dataDir': TRAIN_DATA_DIR['mfcc_delta']},
    #{'layerSize': 3000, 'inputDim': 21*13, 'rawDim': 21*13, 'run_desc': 'mfcc_layer_scaling', 'dataDir': TRAIN_DATA_DIR['mfcc']},
    {'layerSize': 4000, 'inputDim': 21*13, 'rawDim': 21*13, 'run_desc': 'mfcc_layer_scaling', 'dataDir': TRAIN_DATA_DIR['mfcc']},
    #{'layerSize': 5000, 'inputDim': 21*13, 'rawDim': 21*13, 'run_desc': 'mfcc_layer_scaling', 'dataDir': TRAIN_DATA_DIR['mfcc']},
    #{'layerSize': 6000, 'inputDim': 21*13, 'rawDim': 21*13, 'run_desc': 'mfcc_layer_scaling', 'dataDir': TRAIN_DATA_DIR['mfcc']},

]


if __name__ == '__main__':
    import time
    from cluster.utils import run_gpu_job, get_all_free_gpus, get_next_free_gpu_sequential
    from cluster.config import PYTHON_CMD, SLEEP_SEC, FLAGGED_GPUS

    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster', default='gorgon', choices=['gorgon', 'deep'])
    parser.add_argument('--view', action='store_true', default=False, help='Print options only')
    args = parser.parse_args()

    used_gpus = set()
    for setting in PARAM_SETTING_LIST:
        run_args = ' '.join(['--%s %s' % (key, val) for key, val in setting.iteritems()])
        print run_args
        cmd = 'cd %s; source ~/.bashrc; nohup %s runNNet.py %s' % (CTC_DIR, PYTHON_CMD, run_args)
        print cmd

        if args.view:
            continue

        gpu_node = None

        while not gpu_node:
            all_free_gpus = get_all_free_gpus(args.cluster)
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
