
import numpy as np

def ctc_loss(params, seq, blank=39, is_prob = True):
    """
    CTC loss function.
    params - n x m matrix of n-D probability distributions over m frames.
    seq - sequence of phone id's for given example.
    is_prob - whether params have already passed through a softmax
    Returns objective and gradient.
    """
    # TODO stateify this into object so don't have to redefine this on every
    # call/ create new mem for grads etc
    #print 'blank sym index', blank

    seqLen = seq.shape[0] # Length of label sequence (# phones)
    numphones = params.shape[0] # Number of labels
    L = 2*seqLen + 1 # Length of label sequence with blanks
    T = params.shape[1] # Length of utterance (time)
    

    alphas = np.zeros((L,T))
    betas = np.zeros((L,T))

    # Normalize params into probabilities
    # TODO move this, assume NN outputs probs
    if not is_prob:
        params = params - np.max(params,axis=0)
        params = np.exp(params)
        params = params / np.sum(params,axis=0)

    #print params
    #print seq
    # Initialize alphas and forward pass 
    alphas[0,0] = params[blank,0]
    alphas[1,0] = params[seq[0],0]
    c = np.sum(alphas[:,0])
    alphas[:,0] = alphas[:,0] / c
    #print alphas[:,0]
    llForward = np.log(c)
    for t in xrange(1,T):
	start = max(0,L-2*(T-t)) 
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
	c = np.sum(alphas[:,t])
        #print alphas[:,t]
	alphas[:,t] = alphas[:,t] / c
	llForward += np.log(c)
        #print alphas[:,t]
        #print llForward, c

    # Initialize betas and backwards pass
    #print 'starting backward pass'
    betas[-1,-1] = params[blank,-1]
    betas[-2,-1] = params[seq[-1],-1]
    c = np.sum(betas[:,-1])
    betas[:,-1] = betas[:,-1] / c
    llBackward = np.log(c)
    for t in xrange(T-2,-1,-1):
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

	c = np.sum(betas[:,t])
	betas[:,t] = betas[:,t] / c
	llBackward += np.log(c)
        #print llBackward

    # Compute gradient with respect to unnormalized input parameters
    grad = np.zeros(params.shape)
    abs = alphas*betas
    for s in xrange(L):
	# blank
	if s%2 == 0:
	    grad[blank,:] += abs[s,:]
	    abs[s,:] = abs[s,:]/params[blank,:]
	else:
	    grad[seq[(s-1)/2],:] += abs[s,:]
	    abs[s,:] = abs[s,:]/params[seq[(s-1)/2],:]

    grad = params - grad / (params * np.sum(abs,axis=0))

    return -llForward,grad


def grad_check(epsilon=1e-4):
    np.random.seed(33)

    numPhones = 40
    seqLen = 10
    uttLen = 30

    seq = np.floor(np.random.rand(seqLen,1)*numPhones)
    seq = seq.astype(np.int32)

    params = np.random.randn(numPhones,uttLen) 
    _,grad = ctc_loss(params,seq, is_prob=False)
    numgrad = np.empty(grad.shape)

    for i in xrange(params.shape[0]):
	print i
	for j in xrange(params.shape[1]):
	    params[i,j] += epsilon
	    costP,_ = ctc_loss(params,seq, is_prob=False)
	    params[i,j] -= 2*epsilon
	    costL,_ = ctc_loss(params,seq,is_prob=False)
	    params[i,j] += epsilon
	    numgrad[i,j] = (costP - costL) / (2*epsilon)

    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print "Relative norm of difference : %f"%diff
    return diff,numgrad,grad
    
if __name__=='__main__':
    
    diff,numgrad,grad = grad_check()
    
   # np.random.seed(33)

   # numPhones = 40 
   # seqLen = 10
   # uttLen = 30

   # seq = np.floor(np.random.rand(seqLen,1)*numPhones)
   # seq = seq.astype(np.int32)

    # test params 1, gaussian 
   # params = np.random.randn(numPhones,uttLen) 

   # cost,grad = ctc_loss(params,seq)

   # print "Negative loglikelihood is %f"%cost


