# cython: profile=False

from libc cimport math
import numpy as np
cimport numpy as np
np.seterr(all='raise')
import collections
from decoder_utils import int_to_char, load_char_map

cdef double combine(double a,double b,double c=float('-inf')):
    cdef double psum = math.exp(a) + math.exp(b) + math.exp(c)
    if psum == 0.0:
        return float('-inf')
    else:
        return math.log(psum)

# TODO Need id -> char mapping
cdef double lm_placeholder(c, seq):
    return 0.0

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

# FIXME Determine order using lm object
def lm_score_final_char(lm, char_map, prefix, query_char, order=5):
    # Have to reverse prefix for srilm
    s = int_to_char(prefix[-1:-1-order:-1], char_map)
    #print s
    return lm.logprob_strings(char_map[query_char], s)


def decode_clm(double[::1,:] probs not None, lm,
                unsigned int beam=40, double alpha=1.0, double beta=0.0):

    cdef unsigned int N = probs.shape[0]
    cdef unsigned int T = probs.shape[1]
    cdef unsigned int t, i
    cdef float v0, v1, v2, v3

    char_map = load_char_map()

    keyFn = lambda x: combine(x[1][0],x[1][1]) + beta * x[1][2]
    initFn = lambda : [float('-inf'),float('-inf'),0]

    # [prefix, [p_nb, p_b, |W|]]
    Hcurr = [[(),[float('-inf'),0.0,0]]]
    Hold = collections.defaultdict(initFn)

    # loop over time
    for t in xrange(T):
        Hcurr = dict(Hcurr)
        Hnext = collections.defaultdict(initFn)

        for prefix,(v0,v1,numC) in Hcurr.iteritems():

            # CHUNK 1
            valsP = Hnext[prefix]
            # Handle blank
            valsP[1] = combine(v0+probs[0,t],v1+probs[0,t])
            valsP[2] = numC
            if len(prefix) > 0:
                # Handle collapsing
                valsP[0] = combine(v0+probs[prefix[-1],t],valsP[0])

            for i in xrange(1,N):
                nprefix = tuple(list(prefix) + [i])
                valsN = Hnext[nprefix]

                # query the LM for final char score
                lm_prob = alpha * lm_score_final_char(lm, char_map, prefix, i)
                #lm_prob = alpha*lm_placeholder(i,prefix)

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
        Hcurr = sorted(Hnext.iteritems(), key=keyFn, reverse=True)[:beam]

    return list(Hcurr[0][0]),keyFn(Hcurr[0])
