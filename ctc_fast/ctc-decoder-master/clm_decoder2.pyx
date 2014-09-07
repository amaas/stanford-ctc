# cython: profile=False

from libc cimport math
import numpy as np
cimport numpy as np
np.seterr(all='raise')
import collections


cdef double combine(double a,double b,double c=float('-inf')):
    cdef double psum = math.exp(a) + math.exp(b) + math.exp(c)
    if psum == 0.0:
        return float('-inf')
    else:
        return math.log(psum)

# TODO Need id -> char mapping
cdef double lm_placeholder(c, seq):
    return 0.0

def decode_clm(double[::1,:] probs not None, lm,
                unsigned int beam=40, double alpha=1.0, double beta=0.0):

    cdef unsigned int N = probs.shape[0]
    cdef unsigned int T = probs.shape[1]
    cdef unsigned int t, i
    cdef float v0, v1, v2, v3

    keyFn = lambda x: combine(x[1][0],x[1][1]) + beta * x[1][2]
    initFn = lambda : [float('-inf'),float('-inf'),0]

    # [prefix, [p_nb, p_b, node, |W|]]
    Hcurr = [[(),[float('-inf'),0.0,0]]]
    Hold = collections.defaultdict(initFn)

    # loop over time
    for t in xrange(T):
        Hcurr = dict(Hcurr)
        Hnext = collections.defaultdict(initFn)

        for prefix,(v0,v1,numC) in Hcurr.iteritems():

            valsP = Hnext[prefix]
            valsP[1] = combine(v0+probs[0,t],v1+probs[0,t],valsP[1])
            valsP[2] = numC
            if len(prefix) > 0:
                valsP[0] = combine(v0+probs[prefix[-1],t],valsP[0])

            for i in xrange(1,N):
                nprefix = tuple(list(prefix) + [i])
                valsN = Hnext[nprefix]

                lm_prob = alpha*lm_placeholder(i,prefix)
                valsN[2] = numC + 1

                if len(prefix)==0 or (len(prefix) > 0 and i != prefix[-1]):
                    valsN[0] = combine(v0+probs[i,t]+lm_prob,v1+probs[i,t]+lm_prob,valsN[0])
                else:
                    valsN[0] = combine(v1+probs[i,t]+lm_prob,valsN[0])

                if nprefix not in Hcurr:
                    v2,v3,_ = Hold[nprefix]
                    valsN[1] = combine(v2+probs[0,t],v3+probs[0,t],valsN[1])
                    valsN[0] = combine(v2+probs[i,t],valsN[0])


        Hold = Hnext
        Hcurr = sorted(Hnext.iteritems(), key=keyFn, reverse=True)[:beam]

    return list(Hcurr[0][0]),keyFn(Hcurr[0])
