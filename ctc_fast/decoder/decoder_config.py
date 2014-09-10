import os
from os.path import join as pjoin

EGS_DIR = '/scail/group/deeplearning/speech/zxie/kaldi-stanford/kaldi-trunk/egs'

DATASET = 'swbd'

SCAIL_DATA_DIR = '/scail/data/group/deeplearning/u/zxie'

# CTC Parameters

if DATASET == 'wsj':
    #DATA_DIR = pjoin(EGS_DIR, 'wsj/s6/exp/test_dev93_ctc/')
    DATA_DIR = pjoin(EGS_DIR, 'wsj/s6/exp/train_si284_ctc/')
    INPUT_DIM = 21*23
    RAW_DIM = 41*23
    OUTPUT_DIM = 32  # FIXME
    MAX_UTT_LEN = 2000
elif DATASET == 'swbd':
    DATA_DIR = pjoin(EGS_DIR, 'swbd/s5b/exp/eval2000_ctc/')
    INPUT_DIM = 41*15
    RAW_DIM = 41*15
    OUTPUT_DIM = 35
    MAX_UTT_LEN = 1550

# LM Parameters

LM_SOURCE = '/afs/cs.stanford.edu/u/zxie/py-arpa-lm/lm.py'

if DATASET == 'wsj':
    SPACE = '<SPACE>'
    SPECIALS_LIST = ['<SPACE>', '<NOISE>']
    CHARMAP_PATH = pjoin(EGS_DIR, 'wsj/s6/ctc-utils/')
elif DATASET == 'swbd':
    SPACE = '[space]'
    SPECIALS_LIST = ['[vocalized-noise]', '[laughter]', '[space]', '[noise]']
    CHARMAP_PATH = pjoin(EGS_DIR, 'swbd/s5b/ctc-utils/')

USE_TRIGRAM = False
if USE_TRIGRAM:
    LM_PREFIX = 'lm_tg'
else:
    # Default to bigram
    LM_PREFIX = 'lm_bg'
LM_ARPA_FILE = pjoin(CHARMAP_PATH, '%s.arpa' % LM_PREFIX)

# Model parameters

NUM_LAYERS = 5
LAYER_SIZE = 1824
ANNEAL = 1.2 if DATASET == 'wsj' else 1.3
TEMPORAL_LAYER = 3

#MODEL_DIR = '/afs/cs.stanford.edu/u/zxie/kaldi-stanford/stanford-nnet/ctc_fast/models'
MODEL_DIR = '/scail/group/deeplearning/speech/zxie/ctc_models'

def get_brnn_model_file():
    # TODO Figure out what "new_layers" means in wsj model
    model_file = pjoin(MODEL_DIR, '%s_%d_%d_bitemporal_%d_step_1e-5_mom_.95_anneal_%.1f.bin' % (DATASET, NUM_LAYERS, LAYER_SIZE, TEMPORAL_LAYER, ANNEAL))
    assert os.path.exists(model_file)
    return model_file
