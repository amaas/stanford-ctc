# cython: profile=False, boundscheck=False, wraparound=False

import sys
from libc cimport math
import numpy as np
cimport numpy as np
np.seterr(all='raise')
import collections
from decoder_utils import int_to_char, load_chars

class Hyp:
    def __init__(self, pb, pnb, nc):
        self.p_b = pb
        self.p_nb = pnb
        self.n_c = nc

def init_hyp():
    hyp = Hyp(float('-inf'), float('-inf'), 0)
    return hyp

# Add 2 probabilities in log space together and take log
# Used for p_b + p_nb
cdef double exp_sum_log(double a, double b):
    cdef double psum = math.exp(a) + math.exp(b)
    if psum == 0.0:
        return float('-inf')
    return math.log(psum)

# TODO Need id -> char mapping
cdef double lm_placeholder(c, seq):
    return 0.0

def lm_score_final_char(lm, chars, prefix):
    """
    uses lm to score entire prefix
    returns only the log prob of final char
    """
    prefix_str = '<s> ' + ' '.join([chars[x] for x in prefix])
    print prefix_str,
    prefix_scores = lm.full_scores(prefix_str)
    print prefix_scores[-1]

def decode_clm(double[::1,:] probs not None, lm, unsigned int beam=40, double alpha=1.0, double beta=0.0):
    cdef unsigned int N = probs.shape[0]
    cdef unsigned int T = probs.shape[1]
    cdef unsigned int t, k, l, y_e
    cdef double collapse_prob, p_tot

    # Need beta to be in scope
    pref_prob = lambda x: exp_sum_log(x[1].p_nb, x[1].p_b) + beta * x[1].n_c

    chars = load_chars()

    # For values not on the beam
    B_old = collections.defaultdict(init_hyp)

    # Loop over time
    for t in xrange(T):
        if t == 0:
            B_hat = dict()
            # Initial empty prefix
            B_hat[()] = Hyp(0.0, float('-inf'), 0)
        else:
            # Beam cutoff
            sorted_items = sorted(B.iteritems(), key=pref_prob, reverse=True)
            print t
            print '-' * 80
            for kv in sorted_items[:beam]:
                print int_to_char(kv[0], chars)
            B_old = B
            B_hat = dict(sorted_items[:beam])
            if t >= 100:
                sys.exit(0)
        B = collections.defaultdict(init_hyp)

        # Loop over prefixes
        for prefix, hyp in B_hat.iteritems():
            l = hyp.n_c
            p_tot = exp_sum_log(hyp.p_b, hyp.p_nb)

            new_hyp = B[prefix]
            new_hyp.n_c = hyp.n_c
            # Handle collapsing
            if l > 0:
                new_hyp.p_nb = hyp.p_nb + probs[prefix[l-1], t]
                prev_pref = prefix[:l-1]
                if prev_pref in B_hat:
                    prev_hyp = B_hat[prev_pref]
                else:
                    prev_hyp = B_old[prev_pref]

                y_e = prefix[l-1]
                # P(y[-1], y[:-1], t) in Graves paper
                # query the LM for final char score
                lm_val = lm_score_final_char(lm, prefix)
                # lm_placeholder(y_e, prev_pref)
                collapse_prob = probs[y_e, t] + alpha * lm_val    
                if l > 1 and y_e == prefix[l-2]:
                    collapse_prob += prev_hyp.p_b
                else:
                    collapse_prob += exp_sum_log(prev_hyp.p_b, prev_hyp.p_nb)

                new_hyp.p_nb = exp_sum_log(new_hyp.p_nb, collapse_prob)

            # Handle blank extension
            new_hyp.p_b = p_tot + probs[0, t]

            # Handle other extensions
            # Loop over characters excluding blank
            for k in xrange(1, N):
                ext_prefix = tuple(list(prefix) + [k])
                ext_hyp = Hyp(float('-inf'), 0.0, hyp.n_c + 1)

                # P(k, y, t) in Graves paper
                ext_hyp.p_nb = probs[k, t] + alpha * lm_placeholder(k, prefix)
                if l > 0 and k == prefix[l-1]:
                    ext_hyp.p_nb += hyp.p_b
                else:
                    ext_hyp.p_nb += p_tot

                B[ext_prefix] = ext_hyp

    B_final = sorted(B.iteritems(), key=pref_prob, reverse=True)
    return list(B_final[0][0]), pref_prob(B_final[0])
