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


cdef double combine(double a,double b,double c=float('-inf')):
    cdef double psum = math.exp(a) + math.exp(b) + math.exp(c)
    if psum == 0.0:
        return float('-inf')
    else:
        return math.log(psum)

'''
def lm_score_final_char(lm, chars, prefix, query_char):
    """
    uses lm to score entire prefix
    returns only the log prob of final char
    """
    # convert prefix and query to actual text
    # TODO why is prefix a tuple?
    prefix = list(prefix)
    #print prefix
    prefix_str = ' '.join(int_to_char(prefix+[query_char], chars))
    #print prefix_str
    prefix_scores = lm.full_scores(prefix_str)
    words = ['<s>'] + prefix_str.split() + ['</s>']
    #TODO verify lm is not returning score for </s>
    prob_list = [x[0] for x in prefix_scores]
    #for i, (prob, length) in enumerate(prefix_scores):
    #    print words[i], length, prob
    return prob_list[-1]
'''

cdef double lm_score_final_char(lm, char_map, prefix, query_char, order=LM_ORDER):
    # Have to reverse prefix for srilm
    s = int_to_char(prefix[-1:-1-order:-1], char_map)
    if len(s) < order - 1:
        s = s + ['<s>']
    # TODO Fix log10
    return lm.logprob_strings(char_map[query_char], s)


def lm_score_chars(lm, char_map, char_inds, prefix, ctc_probs=None, order=LM_ORDER):
    if ctc_probs is not None:
        ctc_probs = np.array(ctc_probs).reshape((NUM_CHARS, -1))
        if ctc_probs.shape[1] < SOURCE_CONTEXT:
            left = blank_loglikes(SOURCE_CONTEXT - ctc_probs.shape[1])
            ctc_probs = np.hstack((ctc_probs, left))
    s = int_to_char(prefix[-order+1:], char_map)
    if len(s) < order - 1:
        s = ['<null>'] * (order - len(s) - 2) + ['<s>'] + s
    #print s
    data = np.array([char_inds[c] for c in s], dtype=np.int8).reshape((-1, 1))
    #data = np.expand_dims(data, 1)
    data = one_hot(data, len(char_map))
    data = data.reshape((-1, data.shape[2]))
    if ctc_probs is not None:
        # FIXME
        ctc_probs = np.log((1-np.exp(np.array(ctc_probs).reshape((-1, 1)))) + 1e-10)
        data = (data, ctc_probs)
    _, probs = lm.cost_and_grad(data, None)
    #probs = probs[:, -1, :]
    #data = data.reshape((data.size, 1))
    #inds = dict((v, k) for (k, v) in char_inds.iteritems())
    #for k in range(probs.shape[0]):
        #print inds[k], probs[k]
    #assert False
    probs = probs.ravel()
    #print probs
    return np.log(probs)


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
                #cprobs = lm_score_chars(lm, char_map, char_inds, prefix, probs[:, t])
                cprobs = lm_score_chars(lm, char_map, char_inds, prefix)
                lm_prob = cprobs[char_inds['</s>']]
                Hnext[prefix][0] = combine(v0 + lm_prob, v1 + lm_prob, Hnext[prefix][0])

        Hcurr = sorted(Hnext.iteritems(), key=keyFn, reverse=True)[:beam]
    return list(Hcurr[0][0]),keyFn(Hcurr[0])
