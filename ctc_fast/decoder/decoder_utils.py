import time
import numpy as np
import pickle
import dataLoader as dl
from nnets.brnnet import NNet as BRNNet
from decoder_config import get_brnn_model_file, INPUT_DIM, OUTPUT_DIM,\
        RAW_DIM, LAYER_SIZE, NUM_LAYERS, MAX_UTT_LEN, TEMPORAL_LAYER,\
        DATA_DIR, SPACE, CHARMAP_PATH
import decoder
import ctc_fast as ctc
# char lm
import kenlm


def load_chars():
    with open(CHARMAP_PATH+'chars.txt') as fid:
        chars = dict(tuple(l.strip().split()) for l in fid.readlines())
    for k, v in chars.iteritems():
        chars[k] = int(v)
    print chars
    return chars


def load_words():
    with open(CHARMAP_PATH+'wordlist') as fid:
        words = [l.strip() for l in fid.readlines()]
    print 'Loaded %d words' % len(words)
    return words


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


def decode(data, labels, rnn, alpha=1.0, beta=0.0, beam=100, method='clm'):
    probs = rnn.costAndGrad(data, labels)
    probs = np.log(probs.astype(np.float64) + 1e-30)

    hypScore = None
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
        import prefixTree
        pt = prefixTree.loadPrefixTree()
        lm = pt.lm
        print 'Done loading prefix tree.'
        tic = time.time()
        hyp, hypScore = decoder.decode_bg_lm(probs, pt, lm, beam=beam,
                alpha=alpha, beta=beta)
        toc = time.time()
        print 'decoding time (wall): %f' % (toc - tic)
    elif method == 'clm':
        import clm_decoder
        # Character LM
        # NOTE need to restructure decoders into classes
        clm = kenlm.LanguageModel('/scail/group/deeplearning/speech/amaas/kaldi-stanford/kaldi-trunk/egs/wsj/s6/data/local/lm/text_char.2g.arpa')
        hyp, hypScore = clm_decoder.decode_clm(probs, clm, beam=beam,
                alpha=alpha, beta=beta)
    elif method == 'clm2':
        import clm_decoder2
        # Character LM
        # NOTE need to restructure decoders into classes
        clm = kenlm.LanguageModel('/scail/group/deeplearning/speech/amaas/kaldi-stanford/kaldi-trunk/egs/wsj/s6/data/local/lm/text_char.2g.arpa')
        hyp, hypScore = clm_decoder2.decode_clm(probs, clm, beam=beam,
                alpha=alpha, beta=beta)
    elif method == 'fast':
        # TODO Fix bugs in fastdecode
        from fastdecode import decode_lm_wrapper
        hyp, hypScore = decode_lm_wrapper(probs, beam, alpha, beta)
    else:
        assert False, 'No such decoding method: %s' % method

    return hyp, hypScore, refScore
