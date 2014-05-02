
import gnumpy as gp
import numpy as np

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

    def __init__(self,inputDim,outputDim,layerSizes,mbSize=256,train=True,
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
        self.mbSize = mbSize

    def initParams(self):
	"""
	Initialize parameters using 6/sqrt(fanin+fanout)
	"""
        sizes = [self.inputDim]+self.layerSizes+[self.outputDim]
        scales = [gp.sqrt(6)/gp.sqrt(n+m) for n,m in zip(sizes[:-1],sizes[1:])]
        self.stack = [[gp.rand(m,n)*2*s-s,gp.zeros((m,1))] \
                            for n,m,s in zip(sizes[:-1],sizes[1:],scales)]
        self.hActs = [gp.empty((s,self.mbSize)) for s in sizes]

        if self.train:
            self.deltas = [gp.empty((s,self.mbSize)) for s in sizes[1:]]
            self.grad = [[gp.empty(w.shape),gp.empty(b.shape)] for w,b in self.stack]

    def costAndGrad(self,data,labels):
        
        # forward prop
        self.hActs[0] = data
        i = 1
        for w,b in self.stack:
            self.hActs[i] = w.dot(self.hActs[i-1])+b
            if i <= len(self.layerSizes):
                self.hActs[i] = self.activation(self.hActs[i])
            i += 1

        probs = self.hActs[-1]-gp.max(self.hActs[-1],axis=0)
        probs = gp.exp(probs)
        probs = probs/gp.sum(probs,axis=0)

        labelMat = np.zeros(probs.shape)
        labelMat[labels,range(self.mbSize)] = 1
        labelMat = gp.garray(labelMat)
        cost = -(1./self.mbSize)*np.nansum(gp.as_numpy_array(labelMat*gp.log(probs)))

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
            self.grad[i][1] = (1./self.mbSize)*gp.sum(self.deltas[i],axis=1).reshape(-1,1)
        return cost,self.grad

    def updateParams(self,scale,update):
        self.stack = [[ws[0]+scale*wsDelta[0],ws[1]+scale*wsDelta[1]] 
                        for ws,wsDelta in zip(self.stack,update)]

    def vecToStack(self,vec):
        start = 0
        sizes = [self.inputDim]+self.layerSizes+[self.outputDim]
        for n,m,i in zip(sizes[:-1],sizes[1:],range(len(sizes)-1)):
            self.stack[i] = [gp.garray(np.reshape(np.array(vec[start:start+m*n]),(m,n))),\
                gp.garray(np.reshape(np.array(vec[start+m*n:start+m*(n+1)]),(m,1)))]
            start += m*(n+1)
    
    def vectorize(self, x):
        return [v for layer in x for wb in layer for w_or_b in wb for v in w_or_b]

    def paramVec(self):
        return self.vectorize(self.stack)

    def costAndGradVec(self,params,data,labels):
        """
        Vectorized version of costAndGrad
        """
        self.vecToStack(params)
        cost,grad = self.costAndGrad(data,labels)
        if (grad != None):
            vecgrad = self.vectorize(grad)
        return cost,vecgrad

if __name__=='__main__':
    inputDim = 5
    outputDim = 10
    layerSizes = [100, 100, 300]
    mbSize = 5

    # fake data
    data = gp.rand(inputDim,mbSize)
    import random
    labels = [random.randint(0,9)]*mbSize

    # make nnet
    nn = NNet(inputDim,outputDim,layerSizes,mbSize,train=True)
    nn.initParams()

    # run
    cost,grad = nn.costAndGrad(data,labels)
    print cost
