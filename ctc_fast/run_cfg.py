import os
from os.path import join as pjoin

CTC_DIR = os.path.dirname(os.path.abspath(__file__))

# See also default argument values in runNNet.py

RUN_DIR = '/scail/data/group/deeplearning/u/zxie/ctc_models'
SWBD_EXP_DIR = '/deep/group/speech/awni/kaldi-stanford/kaldi-trunk/egs/swbd/s5b/exp/'
SWBD_EXP_DIR_DELTA = '/deep/group/speech/asamar/kaldi-stanford/kaldi-trunk/egs/swbd/s5b/exp/'

# Feature types can be mfcc or fbank for now
# Different configuration for each

TRAIN_DATA_DIR = {
    'fbank': pjoin(SWBD_EXP_DIR, 'train_ctc/'),
    'mfcc': pjoin(SWBD_EXP_DIR, 'train_mfcc_ctc/'),
    'mfcc_delta': pjoin(SWBD_EXP_DIR_DELTA, 'train_mfcc_ctc_delta/')
}
TRAIN_ALIS_DIR = pjoin(SWBD_EXP_DIR, 'train_ctc/')
DEV_DATA_DIR = {
    'fbank': pjoin(SWBD_EXP_DIR, 'dev_ctc/'),
    'mfcc': pjoin(SWBD_EXP_DIR, 'dev_mfcc_ctc/'),
}
DEV_ALIS_DIR = pjoin(SWBD_EXP_DIR, 'dev_ctc/')
TEST_DATA_DIR = {
    'fbank': pjoin(SWBD_EXP_DIR, 'eval2000_ctc/'),
    'mfcc': pjoin(SWBD_EXP_DIR, 'eval2000_mfcc_ctc/'),
}
TEST_ALIS_DIR = pjoin(SWBD_EXP_DIR, 'eval2000_ctc/')

FEAT_DIMS = {
    'fbank': 15,
    'mfcc': 13
}
RAW_CONTEXTS = {
    'fbank': 41,
    'mfcc': 21
}

VIEWER_DIR = '/afs/cs.stanford.edu/u/zxie/www/ctc'

BROWSE_RUNS_KEYS = [
    'run', 'host', 'pid', 'cer', 'wer', 'cost', 'numLayers',
    'layerSize', 'temporalLayer', 'momentum', 'step', 'anneal',
    'param_count', 'inputDim', 'rawDim', 'outputDim', 'reg',
    'epoch', 'num_files', 'complete', 'alive', 'git_rev', 'run_desc'
]
