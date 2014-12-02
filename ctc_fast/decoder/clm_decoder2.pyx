# cython: profile=False

from libc cimport math
import numpy as np
cimport numpy as np
np.seterr(all='raise')
import collections
from decoder_utils import int_to_char, load_char_map
from decoder_config import LM_ORDER
from rnn import one_hot
from ctc_loader import SOURCE_CONTEXT, blank_loglikes, NUM_CHARS
from runDecode import MODEL_TYPE


cdef double combine(double a,double b,double c=float('-inf')):
    cdef double psum = math.exp(a) + math.exp(b) + math.exp(c)
    if psum == 0.0:
        return float('-inf')
    else:
        return math.log(psum)


cdef double lm_score_final_char(lm, char_map, prefix, query_char, order=LM_ORDER):
    # Have to reverse prefix for srilm
    s = int_to_char(prefix[-1:-1-order:-1], char_map)
    if len(s) < order - 1:
        s = s + ['<s>']
    # NOTE srilm gives log10
    return 10**lm.logprob_strings(char_map[query_char], s)


def ngram_score_chars(lm, char_map, prefix, char_inds, N, order=LM_ORDER):
    cprobs = np.ones(N) * 1e-5  # FIXME
    cdef unsigned int j
    for j in xrange(1, N):
        if char_map[j] not in char_inds:
            continue
        cprobs[char_inds[char_map[j]]] = lm_score_final_char(lm, char_map, prefix, j)
    cprobs = cprobs / np.linalg.norm(cprobs)
    cprobs = np.log(cprobs)
    return cprobs


def lm_score_chars(lm, char_map, char_inds, prefix, order=LM_ORDER, prev_h0=None):
    if MODEL_TYPE == 'rnn':
        s = int_to_char(prefix[-1:], char_map)
        if len(s) < 1:
            s = ['<s>']
    else:
        s = int_to_char(prefix[-order+1:], char_map)
        if len(s) < order - 1:
            s = ['<null>'] * (order - len(s) - 2) + ['<s>'] + s

    data = np.array([char_inds[c] for c in s], dtype=np.int8).reshape((-1, 1))
    data = one_hot(data, len(char_map))

    if MODEL_TYPE == 'rnn':
        _, probs = lm.cost_and_grad(data, None, prev_h0=prev_h0)
        probs = probs[:, -1, :]
    else:
        data = data.reshape((-1, data.shape[2]))
        _, probs = lm.cost_and_grad(data, None)

    probs = probs.ravel()
    if MODEL_TYPE == 'rnn':
        return np.log(probs), lm.last_h
    else:
        return np.log(probs)


def rnn_lm_score_chars(lm, char_map, char_inds, prefix, hidden_state_cache):
    if len(prefix) > 0 and prefix[:-1] in hidden_state_cache:
        cprobs, curr_h0 = lm_score_chars(lm, char_map, char_inds, prefix, prev_h0=hidden_state_cache[prefix[:-1]])
    else:
        cprobs, curr_h0 = lm_score_chars(lm, char_map, char_inds, prefix, prev_h0=None)
    hidden_state_cache[prefix] = curr_h0
    return cprobs


def decode_clm(double[::1,:] probs not None, lm,
                unsigned int beam=40, double alpha=1.0, double beta=0.0, char_inds=None):

    cdef unsigned int N = probs.shape[0]
    cdef unsigned int T = probs.shape[1]
    cdef unsigned int t, i, k
    cdef float v0, v1, v2, v3

    char_map = load_char_map()
    char_inds['[space]'] = char_inds[' ']

    keyFn = lambda x: combine(x[1][0],x[1][1]) + beta * x[1][2]
    initFn = lambda : [float('-inf'),float('-inf'),0]

    # [prefix, [p_nb, p_b, |W|]]
    Hcurr = [[(),[float('-inf'),0.0,0]]]
    Hold = collections.defaultdict(initFn)

    # For RNN, save computations
    if MODEL_TYPE == 'rnn':
        hidden_state_cache = dict()

    # loop over time
    for t in xrange(T):
        Hcurr = dict(Hcurr)
        Hnext = collections.defaultdict(initFn)

        for prefix, (v0, v1, numC) in Hcurr.iteritems():

            # CHUNK 1
            valsP = Hnext[prefix]
            # Handle blank
            valsP[1] = combine(v0+probs[0,t],v1+probs[0,t])
            valsP[2] = numC
            if len(prefix) > 0:
                # Handle collapsing
                valsP[0] = combine(v0+probs[prefix[-1],t],valsP[0])

            if MODEL_TYPE == 'ngram':
                cprobs = ngram_score_chars(lm, char_map, prefix, char_inds, N)
            elif MODEL_TYPE == 'rnn':
                cprobs = rnn_lm_score_chars(lm, char_map, char_inds, prefix, hidden_state_cache)
            else:
                cprobs = lm_score_chars(lm, char_map, char_inds, prefix)

            for i in xrange(1, N):
                if char_map[i] not in char_inds:
                    continue

                nprefix = tuple(list(prefix) + [i])
                valsN = Hnext[nprefix]

                lm_prob = alpha * cprobs[char_inds[char_map[i]]]

                valsN[2] = numC + 1

                if len(prefix)==0 or (len(prefix) > 0 and i != prefix[-1]):
                    # Adding a character, apply language model
                    valsN[0] = combine(v0+probs[i,t]+lm_prob,v1+probs[i,t]+lm_prob,valsN[0])
                else:
                    # Repeats must have blank between
                    valsN[0] = combine(v1+probs[i,t]+lm_prob,valsN[0])

                # CHUNK 2
                # If it is in Hcurr then it'll get updated above...
                # NOTE Same as CHUNK 1 above
                if nprefix not in Hcurr:
                    v2,v3,_ = Hold[nprefix]
                    # Handle blank
                    valsN[1] = combine(v2+probs[0,t],v3+probs[0,t],valsN[1])
                    # Handle collapsing
                    valsN[0] = combine(v2+probs[i,t],valsN[0])

        Hold = Hnext

        if t == T - 1:
            # Apply the end of sentence </s> LM probability as well
            for prefix, (v0, v1, numC) in Hnext.iteritems():
                if MODEL_TYPE == 'ngram':
                    cprobs = ngram_score_chars(lm, char_map, prefix, char_inds, N)
                elif MODEL_TYPE == 'rnn':
                    cprobs = rnn_lm_score_chars(lm, char_map, char_inds, prefix, hidden_state_cache)
                else:
                    cprobs = lm_score_chars(lm, char_map, char_inds, prefix)
                lm_prob = cprobs[char_inds['</s>']]
                Hnext[prefix][0] = combine(v0 + lm_prob, v1 + lm_prob, Hnext[prefix][0])

        Hcurr = sorted(Hnext.iteritems(), key=keyFn, reverse=True)[:beam]

        # Clear prefixes from cache
        if MODEL_TYPE == 'rnn':
            curr_ps = set([p[:-1] for p in dict(Hcurr).keys()])
            ps = hidden_state_cache.keys()
            for p in ps:
                if p not in curr_ps:
                    del hidden_state_cache[p]

    return list(Hcurr[0][0]),keyFn(Hcurr[0])
