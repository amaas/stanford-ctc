
from libc cimport math
import cython
import numpy as np
cimport numpy as np
np.seterr(divide='raise',invalid='raise')

ctypedef np.float64_t f_t

# Turn off bounds checking, negative indexing
@cython.boundscheck(False)
@cython.wraparound(False)
def ctc_loss(double[::1,:] params not None, 
        int[::1] seq not None):
    """
    CTC loss function.
    params - n x m matrix of n-D probability distributions over m frames. Must
    be in Fortran order in memory.
    seq - sequence of phone id's for given example.
    Returns objective and gradient.
    """

    cdef unsigned int seqLen = seq.shape[0] # Length of label sequence (# phones)
    cdef unsigned int numphones = params.shape[0] # Number of labels
    # now assume seq comes with blanks already. no special treatment
    cdef unsigned int L = seqLen # Length of label sequence with blanks
    # cdef unsigned int L = 2*seqLen + 1 # Length of label sequence with blanks
    cdef unsigned int T = params.shape[1] # Length of utterance (time)

    cdef double[::1,:] alphas = np.zeros((L,T), dtype=np.double, order='F')
    cdef double[::1,:] betas = np.zeros((L,T), dtype=np.double, order='F')
    cdef double[::1,:] ab = np.empty((L,T), dtype=np.double, order='F')
    cdef np.ndarray[f_t, ndim=2] grad = np.zeros((numphones,T), 
                            dtype=np.double, order='F')
    cdef double[::1,:] grad_v = grad
    cdef double[::1] absum = np.empty(T, dtype=np.double)

    cdef unsigned int start, end
    cdef unsigned int t, s, l
    cdef double c, llForward, llBackward, llDiff, tmp

    try:
        # Initialize alphas and forward pass 
	# there is now only one valid start state, seq[0]
        alphas[0,0] = 1.0 
        llForward = math.log(params[seq[0],0])
        for t in xrange(1,T):
            for s in xrange(0,L):
                if s==0:                        
                    alphas[s,t] = alphas[s,t-1] * params[s,t]
                # NOTE with blank forcing we dont allow skipping so no s-2 case
                else:
                    alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[seq[s],t]
                
            # normalize at current time (prevent underflow)
            c = 0.0
            for s in xrange(0,L):
                c += alphas[s,t]
            for s in xrange(0,L):
                alphas[s,t] = alphas[s,t] / c
            llForward += math.log(c)

        # Initialize betas and backwards pass
        # normalized beta is just prob 1
        betas[L-1,T-1] = 1.0
        llBackward = math.log(params[seq[L-1],T-1])
        for t in xrange(T-1,0,-1):
            # XXX why subtract 1 here?
            t = t-1
            for s in xrange(L,0,-1):
                s = s-1
                if s == L-1:
                    betas[s,t] = betas[s,t+1] * params[seq[s],t]
                else:
                    betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[seq[s],t]

            c = 0.0
            for s in xrange(0,L):
                c += betas[s,t]
            for s in xrange(0,L):
                betas[s,t] = betas[s,t] / c
            llBackward += math.log(c)

        # Compute gradient with respect to unnormalized input parameters
        for t in xrange(T):
            for s in xrange(L):
                ab[s,t] = alphas[s,t]*betas[s,t]
        for s in xrange(L):
            for t in xrange(T):
                grad_v[seq[s],t] += ab[s,t]
                if ab[s,t] != 0:
                    ab[s,t] = ab[s,t]/(params[seq[s],t]) 

        for t in xrange(T):
            absum[t] = 0
            for s in xrange(L):
                absum[t] += ab[s,t]

        # grad = params - grad / (params * absum)
        for t in xrange(T):
            for s in xrange(numphones):
                tmp = (params[s,t]*absum[t])
                if tmp > 0:
                    grad_v[s,t] = params[s,t] - grad_v[s,t] / tmp 
                else:
                    grad_v[s,t] = params[s,t]

    except (FloatingPointError,ZeroDivisionError) as e:
        print e.message
        return -llForward,grad,True


    return -llForward,grad,False

def decode_best_path(double[::1,:] probs not None, unsigned int blank=0):
    """
    Computes best path given sequence of probability distributions per frame.
    Simply chooses most likely label at each timestep then collapses result to
    remove blanks and repeats.
    Optionally computes edit distance between reference transcription
    and best path if reference provided.
    """

    # Compute best path
    cdef unsigned int T = probs.shape[1]
    cdef long [::1] best_path = np.argmax(probs,axis=0)

    # Collapse phone string
    cdef unsigned int i, b
    hyp = []
    for i in xrange(T):
        b = best_path[i]
        # ignore blanks
        if b == blank:
            continue
        # ignore repeats
        elif i != 0 and  b == best_path[i-1]:
            continue
        else:
            hyp.append(b)
    return hyp

