
import cPickle as pickle
import cython
import numpy as np
np.seterr(divide='raise',invalid='raise')

cimport numpy as np
DTYPE = np.float64
ctypedef np.float64_t f_t
ctypedef np.int32_t i_t

# Turn off bounds checking
@cython.boundscheck(False)

def ctc_loss(np.ndarray[f_t, ndim=2] params, 
        np.ndarray[i_t, ndim=1] seq, int blank=0):

    assert params.dtype == DTYPE
    assert seq.dtype == np.int32

    cdef int seqLen = seq.shape[0] # Length of label sequence (# phones)
    cdef int numphones = params.shape[0] # Number of labels
    cdef int L = 2*seqLen + 1 # Length of label sequence with blanks
    cdef int T = params.shape[1] # Length of utterance (time)

    cdef np.ndarray[f_t, ndim=2] alphas = np.zeros((L,T), dtype=DTYPE)
    cdef np.ndarray[f_t, ndim=2] betas = np.zeros((L,T), dtype=DTYPE)
    cdef np.ndarray[f_t, ndim=2] grad = np.zeros((numphones,T), dtype=DTYPE)
    cdef np.ndarray[f_t, ndim=2] ab = np.empty((L,T), dtype=DTYPE)
    cdef np.ndarray[f_t, ndim=1] absum = np.empty(T, dtype=DTYPE)

    cdef int t, s, start, end, l
    cdef f_t c, llForward, llBackward, llDiff

    # TODO this try catch isn't great
    try:
        # Initialize alphas and forward pass 
        alphas[0,0] = params[blank,0]
        alphas[1,0] = params[seq[0],0]
        c = np.sum(alphas[:,0])
        alphas[:,0] = alphas[:,0] / c
        llForward = np.log(c)
        for t in xrange(1,T):
            start = max(0,L-2*(T-t)) 
            end = min(2*t+2,L)
            for s in xrange(start,L):
                l = (s-1)/2
                # blank
                if s%2 == 0:
                    if s==0:
                        alphas[s,t] = alphas[s,t-1] * params[blank,t]
                    else:
                        alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[blank,t]
                # same label twice
                elif s == 1 or seq[l] == seq[l-1]:
                    alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[seq[l],t]
                else:
                    alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1] + alphas[s-2,t-1]) \
                                  * params[seq[l],t]
                
            # normalize at current time (prevent underflow)
            c = np.sum(alphas[start:end,t])
            alphas[start:end,t] = alphas[start:end,t] / c
            llForward += np.log(c)

        # Initialize betas and backwards pass
        betas[L-1,T-1] = params[blank,T-1]
        betas[L-2,T-1] = params[seq[seqLen-1],T-1]
        c = np.sum(betas[:,T-1])
        betas[:,T-1] = betas[:,T-1] / c
        llBackward = np.log(c)
        for t in xrange(T-2,-1,-1):
            start = max(0,L-2*(T-t)) 
            end = min(2*t+2,L)
            for s in xrange(end-1,-1,-1):
                l = (s-1)/2
                # blank
                if s%2 == 0:
                    if s == L-1:
                        betas[s,t] = betas[s,t+1] * params[blank,t]
                    else:
                        betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[blank,t]
                # same label twice
                elif s == L-2 or seq[l] == seq[l+1]:
                    betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[seq[l],t]
                else:
                    betas[s,t] = (betas[s,t+1] + betas[s+1,t+1] + betas[s+2,t+1]) \
                                 * params[seq[l],t]

            c = np.sum(betas[start:end,t])
            betas[start:end,t] = betas[start:end,t] / c
            llBackward += np.log(c)


        # Compute gradient with respect to unnormalized input parameters
        ab = alphas*betas
        for s in xrange(L):
            # blank
            if s%2 == 0:
                grad[blank,:] += ab[s,:]
                ab[s,:] = ab[s,:]/params[blank,:]
            else:
                grad[seq[(s-1)/2],:] += ab[s,:]
                ab[s,:] = ab[s,:]/(params[seq[(s-1)/2],:]) 
        absum = np.sum(ab,axis=0)
        grad = params - np.nan_to_num(grad / (params * absum))

        llDiff = np.abs(llForward-llBackward)

    except FloatingPointError as e:
        print e.message
        print "Diff in forward/backward LL : %f"%llDiff
        print "Zeros found : (%d/%d)"%(np.sum(absum==0),absum.shape[0])
        with open('badutts.bin','w') as fid:
            pickle.dump(seq,fid)
            pickle.dump(alphas,fid)
            pickle.dump(betas,fid)
            pickle.dump(params,fid)
        return -llForward,grad,True 


    return -llForward,grad,False



