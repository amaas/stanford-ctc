
import gnumpy as gp
import numpy as np
import ctc_fast as ctc
#import ctc as ctc

def relu_hard(x, computeGrad = False):
    if (not computeGrad): 
        f = (1/2.)*(x+gp.sign(x)*x)
        return f

    g = gp.sign(x)
    return g

def relu(x, computeGrad = False):
    negslope = .01
    a = (1+negslope)/2.; b = (1-negslope)/2.
    if (not computeGrad): 
        f = a*x + b*gp.sign(x)*x
        return f

    g = a + b*gp.sign(x)
    return g

def sigmoid(x, computeGrad = False):
    if (not computeGrad): 
        f = gp.logistic(x)
        return f

    g = x * (1.-x)
    return g

class NNet:

    def __init__(self,inputDim,outputDim,layerSizes,train=True,
		 activation='relu'):
        self.outputDim = outputDim
        self.inputDim = inputDim
        self.layerSizes = layerSizes
        self.stack = None
        self.train = train
        self.funcdict = {
            "relu_hard" : relu_hard,
            "relu"      : relu,
            "sigmoid"   : sigmoid,
        }
        self.activation = self.funcdict[activation]
	self.hist = {}

    def initParams(self):
	"""
	Initialize parameters using 6/sqrt(fanin+fanout)
	"""
        sizes = [self.inputDim]+self.layerSizes+[self.outputDim]
        scales = [gp.sqrt(6)/gp.sqrt(n+m) for n,m in zip(sizes[:-1],sizes[1:])]
        self.stack = [[gp.rand(m,n)*2*s-s,gp.zeros((m,1))] \
                            for n,m,s in zip(sizes[:-1],sizes[1:],scales)]
        self.hActs = [gp.empty((s,1)) for s in sizes]

        if self.train:
            self.deltas = [gp.empty((s,1)) for s in sizes[1:]]
            self.grad = [[gp.empty(w.shape),gp.empty(b.shape)] for w,b in self.stack]


    def updateParams(self,scale,update,log=False):
	if log:
	    for w,u in zip(self.stack,update):
		wrms = gp.sqrt(gp.mean(w[0]**2))
		urms = gp.sqrt(gp.mean((scale*u[0])**2))
		print "weight rms=%f -- update rms=%f"%(wrms,urms)

        self.stack = [[ws[0]+scale*wsDelta[0],ws[1]+scale*wsDelta[1]] 
                        for ws,wsDelta in zip(self.stack,update)]

    def vecToStack(self,vec):
        start = 0
        sizes = [self.inputDim]+self.layerSizes+[self.outputDim]
        for n,m,i in zip(sizes[:-1],sizes[1:],range(len(sizes)-1)):
            self.stack[i] = [np.array(np.reshape(np.array(vec[start:start+m*n]),(m,n))),\
                np.array(np.reshape(np.array(vec[start+m*n:start+m*(n+1)]),(m,1)))]
            start += m*(n+1)
    
    def vectorize(self, x):
        return [v for layer in x for wb in layer for w_or_b in wb for v in w_or_b]

    def paramVec(self):
        return self.vectorize(self.stack)

    def paramsToHost(self):
	self.stack = [[w.as_numpy_array(), b.as_numpy_array()] 
                        for w,b in self.stack]

    def costAndGradSFO(self,stack,datums):
        """
	Wrapper function used for SFO optimizer.
        """
	N = len(datums)
	cost = 0.
	grad = [[gp.zeros(w.shape),gp.zeros(b.shape)] 
			  for w,b in self.stack]

        # Push stack to device
	self.stack = [[gp.garray(w),gp.garray(b)] 
                        for w,b in stack]

	for datum in datums:
            data = gp.garray(self.data_dict[datum])
            labels =  np.array(self.alis[datum], dtype=np.int32)
	    costSingle,gradSingle,skip = self.costAndGrad(data,labels)
            if skip:
                print "LOGGING SKIP" #TODO what to do here?
                N -= 1
                continue
	    grad = [[gs[0]+g[0],gs[1]+g[1]]
			      for gs,g in zip(gradSingle,grad)]
	    cost += costSingle
            
            # Have to force GC the gpu... gnumpy lameness
	    gp.free_reuse_cache()

        # Pull gradient from device
        grad = [[((1./N)*gw).as_numpy_array(), ((1./N)*gb).as_numpy_array()] 
                      for gw,gb in grad]
        cost *= 1./N

        return cost,grad

    def costAndGradVec(self,params,data,labels):
        """
        Vectorized version of CTC cost 
        """
        self.vecToStack(params)
        cost,grad = self.costAndGrad(data,labels)
        if (grad != None):
            vecgrad = self.vectorize(grad)
        return cost,vecgrad

    def costAndGrad(self,data,labels=None,key=None):
        """
        Forward prop entire utterance
        Call CTC cost function
        Compute gradient

        data is a 2-D matrix where each column is a single time frame
        Number of input frames changes across iterations
        
        labels is a vector of symbol ids, length unknown and does not
        depend on the number of time frames
        """

        ## forward prop
        # this is the same as minibatch forward prop 
        # since we pre-compute context window features for each time
        self.hActs[0] = data
        i = 1
        for w,b in self.stack:
            self.hActs[i] = w.dot(self.hActs[i-1])+b
            if i <= len(self.layerSizes):
                self.hActs[i] = self.activation(self.hActs[i])
            i += 1

        probs = self.hActs[-1]-gp.max(self.hActs[-1],axis=0)
	probs = gp.as_numpy_array(probs)
        probs = np.exp(probs)
        probs = probs/np.sum(probs,axis=0)
#	probs[probs<1e-12] = 1e-12 # TODO have to clamp?

        ## pass probs and label string to ctc loss
        # TODO how much does passing to different function cost us? 
	if not self.train:
	    return ctc.decode_best_path(probs, ref=labels, blank=0)
	    #return ctc.decode_bp_bigrams(probs, blank=0, B=None)

        cost, self.deltas[-1], skip = ctc.ctc_loss(probs, labels, blank=0)

	# Bad utterance ?
	if skip:
	    return cost,self.grad,skip

	# Store probabilities and error signal for a given key
	#if key is not None and key in self.hist:
	#    self.hist[key].append((probs,self.deltas[-1]))

	self.deltas[-1] = gp.garray(self.deltas[-1])

        # back prop
        i = len(self.layerSizes)-1
        for w,b in reversed(self.stack[1:]):
            grad = self.activation(self.hActs[i+1], True)
            self.deltas[i] = w.T.dot(self.deltas[i+1])*grad
            i -= 1

        # compute gradients
        # NOTE we do not divide by utterance length. 
        #    Will need to scale up weight norm penalty accordingly
        for i in range(len(self.grad)):
            self.grad[i][0] = self.deltas[i].dot(self.hActs[i].T)
            self.grad[i][1] = gp.sum(self.deltas[i],axis=1).reshape(-1,1)

        return cost,self.grad,skip

    def toFile(self,fid):
	"""
	Saves only the network parameters to the given fd.
	"""
	import cPickle as pickle
	pickle.dump(self.stack,fid)

    def fromFile(self,fid):
	import cPickle as pickle
	self.stack = pickle.load(fid)
        self.stack = [[gp.garray(w),gp.garray(b)] for w,b in self.stack]

def gradcheck(epsilon=1e-4):

    import dataLoader as dl
    import random 

    loader = dl.DataLoader('/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/timit/s5/exp/nn_train/',41*23,41*23)
    nn = NNet(41*23,41*23,[1024])
    nn.initParams()

    data_dict,alis,keys,sizes = loader.loadDataFileDict(1)

    k = random.sample(keys,1)[0]

    data = gp.garray(data_dict[k])
    labels = np.array(alis[k],dtype=np.int32)

    cost,grad,_ = nn.costAndGrad(data,labels)
    print data.shape
    print labels.shape

    while True:
        m,n = nn.stack[1][0].shape
        msample,nsample = random.randint(0,m-1),random.randint(0,n-1)
        nn.stack[1][0][msample,nsample] += epsilon

        cost2,grad,_ = nn.costAndGrad(data,labels)
    
        nn.stack[1][0][msample,nsample] -= epsilon

        finite_diff = (cost2 - cost) / epsilon
        print "Analytic %.6f -- Finite %.6f"%(grad[1][0][msample,nsample],finite_diff)
            
        # Clear gp mem
        gp.free_reuse_cache()


if __name__=='__main__':
    gradcheck()

    inputDim = 5
    numPhones = 6
    outputDim = numPhones + 1
    seq_len_out = 4
    seq_len_in = 10
    # can't output more symbols than input times
    assert seq_len_in > seq_len_out
    layerSizes = [10, 10]

    # make sure seq labels do not have '0' which is our blank index
    label_seq = np.floor(np.random.rand(seq_len_out,1)*numPhones)
    label_seq = 1 + label_seq.astype(np.int32)
    data = np.random.randn(inputDim,seq_len_in)

    # make nnet
    nn = NNet(inputDim, outputDim, layerSizes, seq_len_in, train=True)
    nn.initParams()

    # run
    cost,grad = nn.costAndGrad(data,label_seq)
    print cost
