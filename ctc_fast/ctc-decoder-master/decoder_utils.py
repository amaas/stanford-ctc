import numpy as np
import pickle
import dataLoader as dl
from brnnet import NNet as BRNNet
from decoder_config import get_brnn_model_file, INPUT_DIM, OUTPUT_DIM,\
        RAW_DIM, LAYER_SIZE, NUM_LAYERS, MAX_UTT_LEN, TEMPORAL_LAYER,\
        DATA_DIR, SPACE
import decoder
import clm_decoder
import ctc_fast as ctc
import prefixTree


def load_data(fnum=1):
    loader = dl.DataLoader(DATA_DIR, RAW_DIM, INPUT_DIM)
    data_dict, alis, keys, _ = loader.loadDataFileDict(fnum)
    return data_dict, alis, keys


def load_nnet():
    rnn = BRNNet(INPUT_DIM, OUTPUT_DIM, LAYER_SIZE, NUM_LAYERS,
            MAX_UTT_LEN, train=False, temporalLayer=TEMPORAL_LAYER)
    rnn.initParams()
    print get_brnn_model_file()
    fid = open(get_brnn_model_file(), 'rb')
    cfg = pickle.load(fid)
    sgd_costs = pickle.load(fid)
    rnn.fromFile(fid)
    return rnn


def int_to_char(int_seq, chars):
    char_map = dict((v, k) for k, v in chars.iteritems())
    return [char_map[i] for i in int_seq]


def collapse_seq(char_seq):
    return ''.join([' ' if c == SPACE else c for c in char_seq])


def decode(data, labels, rnn, alpha=1.0, beta=0.0, method='clm'):
    probs = rnn.costAndGrad(data, labels)
    probs = np.log(probs.astype(np.float64))

    refScore = None
    # TODO Couldn't find score_sentence
    #refScore = ctc.score_sentence(probs,labels)
    #refScore += alpha*self.lm.score_tg(" ".join(sentence)) + beta*len(sentence)

    # Various decoding options

    if method == 'pmax':
        # Pointwise argmax
        hyp = ctc.decode_best_path(probs)
    elif method == 'bg':
        # Bigram LM w/ prefix tree dictionary constraint
        # FIXME Prefix tree + lm loading should be moved out
        print 'Loading prefix tree (this can take a while)...'
        pt = prefixTree.loadPrefixTree()
        lm = pt.lm
        print 'Done loading prefix tree.'
        hyp, hypScore = decoder.decode_bg_lm(probs, pt, lm, beam=400,
                alpha=alpha, beta=beta)
    elif method == 'clm':
        # Character LM
        clm = None
        hyp, hypScore = clm_decoder.decode_clm(probs, clm, beam=40,
                alpha=alpha, beta=beta)
    elif method == 'fast':
        # TODO Fix bugs in fastdecode
        from fastdecode import decode_lm_wrapper
        decode_lm_wrapper(probs, 3, alpha, beta)
    else:
        assert False, 'No such decoding method %s' % method

    return hyp, hypScore, refScore
