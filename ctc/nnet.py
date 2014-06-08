
# import gnumpy as gp
import numpy as np
import ctc

def relu_hard(x, computeGrad = False):
    if (not computeGrad): 
        f = (1/2.)*(x+np.sign(x)*x)
        return f

    g = np.sign(x)
    return g

def relu(x, computeGrad = False):
    negslope = .01
    a = (1+negslope)/2.; b = (1-negslope)/2.
    if (not computeGrad): 
        f = a*x + b*np.sign(x)*x
        return f

    g = a + b*np.sign(x)
    return g

def sigmoid(x, computeGrad = False):
    if (not computeGrad): 
        f = np.logistic(x)
        return f

    g = x * (1.-x)
    return g

class NNet:

    def __init__(self,inputDim,outputDim,layerSizes,mbSize=256,train=True,
		 activation='relu_hard'):
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
        self.mbSize = mbSize
	self.hist = {}

    def initParams(self):
	"""
	Initialize parameters using 6/sqrt(fanin+fanout)
	"""
        sizes = [self.inputDim]+self.layerSizes+[self.outputDim]
        scales = [np.sqrt(6)/np.sqrt(n+m) for n,m in zip(sizes[:-1],sizes[1:])]
        self.stack = [[np.random.rand(m,n)*2*s-s,np.zeros((m,1))] \
                            for n,m,s in zip(sizes[:-1],sizes[1:],scales)]
        self.hActs = [np.empty((s,self.mbSize)) for s in sizes]

        if self.train:
            self.deltas = [np.empty((s,self.mbSize)) for s in sizes[1:]]
            self.grad = [[np.empty(w.shape),np.empty(b.shape)] for w,b in self.stack]

    def ceCostAndGrad(self,data,labels):
        # NOTE prefixed with CE (cross entropy) so that CTC is default cost
        # forward prop
        self.hActs[0] = data
        i = 1
        for w,b in self.stack:
            self.hActs[i] = w.dot(self.hActs[i-1])+b
            if i <= len(self.layerSizes):
                self.hActs[i] = self.activation(self.hActs[i])
            i += 1

        probs = self.hActs[-1]-np.max(self.hActs[-1],axis=0)
        probs = np.exp(probs)
        probs = probs/np.sum(probs,axis=0)

        labelMat = np.zeros(probs.shape)
        labelMat[labels,range(self.mbSize)] = 1
        labelMat = np.array(labelMat)
        cost = -(1./self.mbSize)*np.nansum(np.as_numpy_array(labelMat*np.log(probs)))

        if not self.train:
            return cost,None

        # back prop
        self.deltas[-1] = probs-labelMat
        i = len(self.layerSizes)-1
        for w,b in reversed(self.stack[1:]):
            grad = self.activation(self.hActs[i+1], True)
            self.deltas[i] = w.T.dot(self.deltas[i+1])*grad
            i -= 1

        # compute gradients
        for i in range(len(self.grad)):
            self.grad[i][0] = (1./self.mbSize)*self.deltas[i].dot(self.hActs[i].T)
            self.grad[i][1] = (1./self.mbSize)*np.sum(self.deltas[i],axis=1).reshape(-1,1)
        return cost,self.grad

    def updateParams(self,scale,update):
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

    def costAndGradVec(self,params,data,labels):
        """
        Vectorized version of CTC cost 
        """
        self.vecToStack(params)
        cost,grad = self.costAndGrad(data,labels)
        if (grad != None):
            vecgrad = self.vectorize(grad)
        return cost,vecgrad

    def costAndGrad(self,data,labels,key=None):
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
            self.hActs[i] = np.dot(w, self.hActs[i-1])+b
            if i <= len(self.layerSizes):
                self.hActs[i] = self.activation(self.hActs[i])
            i += 1

        probs = self.hActs[-1]-np.max(self.hActs[-1],axis=0)
        probs = np.exp(probs)
        probs = probs/np.sum(probs,axis=0)

        ## pass probs and label string to ctc loss
        # TODO how much does passing to different function cost us? 
	if not self.train:
	    return ctc.decode_best_path(probs, ref=labels, blank=0)

        cost, self.deltas[-1] = ctc.ctc_loss(probs, labels, blank=0, is_prob=True)

	# Store probabilities and error signal for a given key
	if key is not None and key in self.hist:
	    self.hist[key].append((probs,self.deltas[-1]))

        if not self.train:
            return cost,None

        # back prop
        i = len(self.layerSizes)-1
        for w,b in reversed(self.stack[1:]):
            grad = self.activation(self.hActs[i+1], True)
            self.deltas[i] = np.dot(w.T, self.deltas[i+1])*grad
            i -= 1

        # compute gradients
        # NOTE we do not divide by utterance length. 
        #    Will need to scale up weight norm penalty accordingly
        for i in range(len(self.grad)):
            self.grad[i][0] = np.dot(self.deltas[i], self.hActs[i].T)
            self.grad[i][1] = np.sum(self.deltas[i],axis=1).reshape(-1,1)

        return cost,self.grad

    def toFile(self,fid):
	"""
	Saves only the network parameters to the given fd.
	"""
	import cPickle as pickle
	pickle.dump(self.stack,fid)

    def fromFile(self,fid):
	import cPickle as pickle
	self.stack = pickle.load(fid)

if __name__=='__main__':
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
