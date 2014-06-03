# TODO merge this with normal nnet
# import gnumpy as gp
import numpy as np
import ctc
# debug tool
from IPython import embed
#from ipsh import *
#DEBUG = TRUE

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

class RNNet:

    def __init__(self,inputDim,outputDim,layerSizes,train=True,
		 activation='relu_hard', temporalLayer = -1):
        """
        temporalLayer indicates which layer is recurrent. <= 0 indicates no recurrernce
        """

        self.outputDim = outputDim
        self.inputDim = inputDim
        self.layerSizes = layerSizes
        self.temporalLayer = temporalLayer
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
        scales = [np.sqrt(6)/np.sqrt(n+m) for n,m in zip(sizes[:-1],sizes[1:])]
        self.stack = [[np.random.rand(m,n)*2*s-s,np.zeros((m,1))] \
                            for n,m,s in zip(sizes[:-1],sizes[1:],scales)]
        if self.temporalLayer > 0:
            rs = sizes[self.temporalLayer]
            s = np.sqrt(6)/ rs
            # temporal layer stored at end of stack
            self.stack.append((np.random.rand(rs,rs) * 2 * s - s, np.zeros(1)))
        
        if self.train:
            #TODO why store all deltas?
            #self.deltas = [np.empty((s,self.mbSize)) for s in sizes[1:]]
            #NOTE if a temporal layer is used it's already added to stack so will have a grad
            self.grad = [[np.empty(w.shape),np.empty(b.shape)] for w,b in self.stack]
 

    def updateParams(self,scale, update):
        """
        This is to do parameter updates in place. Performs the same whether RNN or not
        """
        self.stack = [[ws[0]+scale*wsDelta[0],ws[1]+scale*wsDelta[1]] 
                        for ws,wsDelta in zip(self.stack,update)]
            
    def vecToStack(self,vec):
        start = 0
        sizes = [self.inputDim]+self.layerSizes+[self.outputDim]
        for n,m,i in zip(sizes[:-1],sizes[1:],range(len(sizes)-1)):
            # TODO why is this unpacking into a list for w,b instead of tuple?
            self.stack[i] = [np.array(np.reshape(np.array(vec[start:start+m*n]),(m,n))),\
                np.array(np.reshape(np.array(vec[start+m*n:start+m*(n+1)]),(m,1)))]
            start += m*(n+1)
        if self.temporalLayer > 0:
            rs = self.layerSizes[self.temporalLayer-1]
            self.stack[-1] = [np.array(np.reshape(np.array(vec[start:start+rs*rs]),(rs,rs))),\
                                   np.array(vec[start+rs*rs:end])]
    def vectorize(self, x):
        """
        Converts a stack object into a single parameter vector
        x is a stack
        returns a single numpy array
        XXX or does this return a list of lists?
        """
        return [v for layer in x for wb in layer for w_or_b in wb for v in w_or_b]

    def paramVec(self):
        return self.vectorize(self.stack)

    def costAndGradVec(self,params,data,labels):
        """
        Vectorized version of CTC cost
        data is a single utterance. Each column is a time index [0...T]
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
        T = data.shape[1]
        sizes = [self.inputDim]+self.layerSizes+[self.outputDim]
        stackMax = len(self.stack)-1
        if self.temporalLayer > 0:
            stackMax -= 1

        self.hActs = [np.empty((s,T)) for s in sizes]
        self.hActs[0] = data
        for t in range(T):
            i = 1
            for l in range(stackMax+1):
                w,b = self.stack[l]

                self.hActs[i][:,t] = np.dot(w, self.hActs[i-1][:,t])
                if (self.temporalLayer-1) == l and t>0:
                    w_t,b_t = self.stack[-1]
                    self.hActs[i][:,t] += np.dot(w_t, self.hActs[i][:,t-1])
                # have to index b to make np broadcast work. it's just a vector
                self.hActs[i][:,t] += b[:,0]

                # hidden layer activation function
                #print i
                if i <= stackMax:
                    #print i, self.hActs[i].shape, 'applied nonlin'
                    self.hActs[i][:,t] = self.activation(self.hActs[i][:,t])
                i += 1

        # convert final layer to probs after all time iteration complete
        probs = self.hActs[-1]-np.max(self.hActs[-1],axis=0)
        probs = np.exp(probs)
        probs = probs/np.sum(probs,axis=0)

        ## pass probs and label string to ctc loss
        # TODO how much does passing to different function cost us? 
        cost, delta_output = ctc.ctc_loss(probs, labels, blank=0, is_prob=True)

	# Store probabilities and error signal for a given key
	if key is not None and key in self.hist:
	    self.hist[key].append((probs,delta_output))

        if not self.train:
            return cost,None

        
        ## back prop through time
        # zero gradients
        self.grad = [[np.zeros(w.shape),np.zeros(b.shape)] for w,b in self.stack]
        if self.temporalLayer > 0:
            delta_t = np.zeros(self.layerSizes[self.temporalLayer-1])
        for t in reversed(range(T)):
            # get delta from loss function
            delta = delta_output[:,t].T

            # compute gradient for output layer
            #print self.hActs[-2].shape, delta.shape, self.stack[stackMax][0].shape
            #print delta.reshape(-1,1).shape, self.hActs[-2][:,t].reshape(-1,1).shape
            # TODO can we get rid of some of these annoying reshape -1 1?
            self.grad[stackMax][0] +=  np.dot(delta.reshape(-1,1), self.hActs[-2][:,t].reshape(-1,1).T)
            self.grad[stackMax][1] +=  delta.reshape(-1, 1)

            # push delta through output layer
            #print self.stack[stackMax][0].T.shape
            delta = np.dot(self.stack[stackMax][0].T, delta)
            

            # iterate over lower layers
            i = len(self.layerSizes)-1
            while i >= 0:
                # add the temporal delta if this is the recurrent layer
                #if (self.temporalLayer-1) == i:
                #    delta += delta_t
                # push delta through activation function for this layer
                #print i, stackMax, delta.shape, self.hActs[i+1][:,t].shape
                delta = delta * self.activation(self.hActs[i+1][:,t], True)
                #embed()
                # compute the gradient
                #print i, delta.shape, self.hActs[i][:,t].T.reshape(1,-1).shape, self.grad[i][0].shape
                self.grad[i][0] += np.dot(delta.reshape(-1,1), self.hActs[i][:,t].T.reshape(1,-1))
                self.grad[i][1] += delta.reshape(-1,1)

                # add the temporal delta if this is the recurrent layer
                # if (self.temporalLayer-1) == i and t > 0:
                #     self.grad[-1][0] += np.dot(delta.reshape(-1,1), self.hActs[i+1][:,t-1].T.reshape(1,-1))
                #     # push delta through temporal connections
                #     delta_t = np.dot(self.stack[-1][0].T, delta)

                # push the delta downward
                w,b = self.stack[i]
                delta = np.dot(w.T, delta)
                i -= 1
        #print self.grad
        return cost,self.grad


if __name__=='__main__':
    inputDim = 3
    numPhones = 6
    outputDim = numPhones + 1
    seq_len_out = 2
    seq_len_in = 3
    # can't output more symbols than input times
    assert seq_len_in >= seq_len_out
    layerSizes = [10, 5]

    # make sure seq labels do not have '0' which is our blank index
    label_seq = np.floor(np.random.rand(seq_len_out,1)*numPhones)
    label_seq = 1 + label_seq.astype(np.int32)
    data = np.random.randn(inputDim,seq_len_in)

    # make nnet
    nn = RNNet(inputDim, outputDim, layerSizes, train=True, temporalLayer=-1)
    nn.initParams()

    # run
    cost,grad = nn.costAndGrad(data,label_seq)
    print cost
    #print grad
