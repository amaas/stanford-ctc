# cython: profile=False

from libc cimport math
import cython
import numpy as np
cimport numpy as np
np.seterr(all='raise')
import collections
cimport cython

ctypedef np.float64_t f_t

cdef double combine(double a,double b,double c=float('-inf')):
    cdef double psum = math.exp(a) + math.exp(b) + math.exp(c)
    if psum == 0.0:
        return float('-inf')
    else:
        return math.log(psum)

def decode_bg_clm(double[::1,:] probs not None, lm, clm, int_char_map, unsigned int beam=40, double alpha=1.0, double beta=0.0):
    # TODO NOTE Doesn't use a prefix tree right now to get a better sense
    # of what the character LM is doing. May also not be necessary once
    # we start using a 6 or 7-gram and may complicate vectorizing

    cdef unsigned int N = probs.shape[0]
    cdef unsigned int T = probs.shape[1]
    cdef unsigned int t, i
    cdef float v0, v1, v2, v3

    keyFn = lambda x: combine(x[1][0],x[1][1]) + beta * x[1][3]
    initFn = lambda : [float('-inf'), float('-inf'), None, 0]

    # [prefix, [p_nb, p_b, lm.start, |C|]]
    Hcurr = [[(), [float('-inf'), 0.0, lm.start, 0]]]
    Hold = collections.defaultdict(initFn)

    # Loop over time
    for t in xrange(T):
        Hcurr = dict(Hcurr)
        Hnext = collections.defaultdict(initFn)

        for prefix, (v0, v1, prevId, numC) in Hcurr.iteritems():

            valsP = Hnext[prefix]
            valsP[1] = combine(v0+probs[0,t], v1+probs[0,t], valsP[1])
            valsP[2] = prevId
            valsP[3] = numC

            for i in range(1, N):
                nprefix = list(prefix)
                nprefix.append(i)
                nprefix = tuple(nprefix)
                valsN = Hnext[nprefix]

                # Query KenLM
                # NOTE Currently need to add spaces between characters
                print prevId
                #print int_char_map[prevId]
                #print i
                #print int_char_map[i]
                #print '---'
                bg = alpha * clm.score(int_char_map[prevId] + ' ' + int_char_map[i])
                valsN[3] = numC + 1

                if len(prefix)==0 or (len(prefix) > 0 and i != prefix[-1]):
                    valsN[0] = combine(v0 + probs[i,t] + bg, v1 + probs[i,t] + bg, valsN[0])
                else:
                    valsN[0] = combine(v1 + probs[i,t] + bg,valsN[0])

                if nprefix not in Hcurr:
                    v2, v3, _, _ = Hold[nprefix]
                    valsN[1] = combine(v2 + probs[0,t], v3 + probs[0,t], valsN[1])
                    valsN[0] = combine(v2 + probs[i,t], valsN[0])

                if len(prefix) > 0 and i == prefix[-1]:
                    valsP[0] = combine(v0 + probs[i,t], valsP[0])

        Hold = Hnext
        Hcurr = sorted(Hnext.iteritems(), key=keyFn, reverse=True)[:beam]

    return list(Hcurr[0][0]), keyFn(Hcurr[0])


#@cython.boundscheck(False)
#@cython.wraparound(False)
def decode_bg_lm(double[::1,:] probs not None, prefixTree, lm,
                unsigned int beam=40, double alpha=1.0, double beta=0.0):

    cdef unsigned int N = probs.shape[0]
    cdef unsigned int T = probs.shape[1]
    cdef unsigned int t, i
    cdef float v0, v1, v2, v3
    cdef unsigned int space = prefixTree.space

    keyFn = lambda x: combine(x[1][0],x[1][1]) + beta * x[1][4]
    initFn = lambda : [float('-inf'),float('-inf'),None,None,0]

    # [prefix, [p_nb, p_b, node, lm.start, |W|]]
    Hcurr = [[(),[float('-inf'),0.0,prefixTree.root,lm.start,0]]]
    Hold = collections.defaultdict(initFn)

    # loop over time
    for t in xrange(T):
        Hcurr = dict(Hcurr)
        Hnext = collections.defaultdict(initFn)

        for prefix,(v0,v1,node,prevId,numW) in Hcurr.iteritems():

            valsP = Hnext[prefix]
            valsP[1] = combine(v0+probs[0,t],v1+probs[0,t],valsP[1])
            valsP[2] = node
            valsP[3] = prevId
            valsP[4] = numW

            for i in xrange(1,N):
                skip = False
                emitW = False
                if i == space:
                    if node.isWord:
                        emitW = True
                        newNode = prefixTree.root
                    else:
                        skip = True
                else:
                    if node.children is not None and (node.children[i].isPrefix
                            or node.children[i].isWord):
                        newNode = node.children[i]
                    else:
                        skip = True

                if not skip:
                    nprefix = list(prefix)
                    nprefix.append(i)
                    nprefix = tuple(nprefix)
                    valsN = Hnext[nprefix]
                    valsN[2] = newNode

                    if emitW:
                        bg = alpha*lm.bg_prob(prevId,node.id)
                        valsN[3] = node.id
                        valsN[4] = numW + 1
                    else:
                        bg = 0.0
                        valsN[3] = prevId
                        valsN[4] = numW

                    if len(prefix)==0 or (len(prefix) > 0 and i != prefix[-1]):
                        valsN[0] = combine(v0+probs[i,t]+bg,v1+probs[i,t]+bg,valsN[0])
                    else:
                        valsN[0] = combine(v1+probs[i,t]+bg,valsN[0])

                    if nprefix not in Hcurr:
                        v2,v3,_,_,_ = Hold[nprefix]
                        valsN[1] = combine(v2+probs[0,t],v3+probs[0,t],valsN[1])
                        valsN[0] = combine(v2+probs[i,t],valsN[0])

                if len(prefix) > 0 and i == prefix[-1]:
                    valsP[0] = combine(v0+probs[i,t],valsP[0])

        Hold = Hnext
        Hcurr = sorted(Hnext.iteritems(), key=keyFn, reverse=True)[:beam]

    return list(Hcurr[0][0]),keyFn(Hcurr[0])

