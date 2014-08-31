import pickle
import numpy as np
import prefixTree
import dataLoader as dl
from decoder import decode_bg_lm
from brnnet import NNet as BRNNet
from decoder_config import get_brnn_model_file, INPUT_DIM, OUTPUT_DIM,\
        RAW_DIM, LAYER_SIZE, NUM_LAYERS, MAX_UTT_LEN, TEMPORAL_LAYER,\
        DATA_DIR


def load_data():
    loader = dl.DataLoader(DATA_DIR, RAW_DIM, INPUT_DIM)
    data_dict, alis, keys, _ = loader.loadDataFileDict(1)
    data, labels = data_dict[keys[3]], np.array(alis[keys[3]], dtype=np.int32)
    #data = data_dict[keys[3]]
    return data, labels


def load_nnet():
    rnn = BRNNet(INPUT_DIM, OUTPUT_DIM, LAYER_SIZE, NUM_LAYERS,
            MAX_UTT_LEN, train=False, temporalLayer=TEMPORAL_LAYER)
    rnn.initParams()
    fid = open(get_brnn_model_file(), 'rb')
    cfg = pickle.load(fid)
    sgd_costs = pickle.load(fid)
    rnn.fromFile(fid)
    return rnn


def int_to_char(int_seq, prefix_tree):
    char_map = dict((v, k) for k, v in prefix_tree.chars.iteritems())
    return [char_map[i] for i in int_seq]


if __name__ == '__main__':
    print 'Loading data'
    data, labels = load_data()
    print 'Loading neural net'
    # NOTE Net loads prefix tree and LM
    rnn = load_nnet()

    # Profiling setup
    import pstats, cProfile
    import pyximport
    pyximport.install()
    cProfile.runctx('rnn.costAndGrad(data, labels)', globals(), locals(), 'costAndGrad.lprof')
    s = pstats.Stats('costAndGrad.lprof')
    s.strip_dirs().sort_stats('time').print_stats()

    '''
    print 'Computing likelihoods'
    hyp, hypScore, refScore = rnn.costAndGrad(data, labels)

    print '-' * 80
    if labels:
        print 'labels:', int_to_char(labels, rnn.pt)
    prefix = int_to_char(hyp, rnn.pt)
    print 'top hyp:', prefix
    print 'score:', hypScore
    print 'ref score:', refScore
    '''
