import ctc_fast as ctc
import numpy as np
import pdb

class RNNet:

    def __init__(self,inputDim,outputDim,layerSize,numLayers,maxBatch,
            train=True,temporalLayer=-1):

        self.outputDim = outputDim
        self.inputDim = inputDim
        self.layerSize = layerSize
        self.numLayers = numLayers
        self.layerSizes = [layerSize]*numLayers
        self.maxBatch = maxBatch
        self.train = train

        if temporalLayer <= 0 or temporalLayer >= numLayers:
            self.temporalLayer = -1
        else:
            self.temporalLayer = temporalLayer

    def initParams(self):
	"""
	Initialize parameters using 6/sqrt(fanin+fanout)
	"""
        sizes = [self.inputDim]+self.layerSizes+[self.outputDim]
        scales = [np.sqrt(6)/np.sqrt(n+m) for n,m in zip(sizes[:-1],sizes[1:])]
        self.stack = [[np.random.rand(m,n)*2*s-s,np.zeros((m,1))] \
                            for n,m,s in zip(sizes[:-1],sizes[1:],scales)]
        self.hActs = [np.empty((s,self.maxBatch)) for s in sizes]
        self.deltaTemps = np.empty((self.layerSize,self.maxBatch))

        if self.train:
            # Now assuming that all layers are the same size
            self.grad = [[np.empty(w.shape),np.empty(b.shape)] for w,b in self.stack]
 
        if self.temporalLayer > 0:
            # dummy bias used for temporal layer

            dummy = np.zeros((1,1))

            scale = np.sqrt(6)/np.sqrt(self.layerSize*2)
            wtf = 2*scale*np.random.rand(self.layerSize,self.layerSize) - scale
            self.stack.append([wtf,dummy])
            wtb = 2*scale*np.random.rand(self.layerSize,self.layerSize) - scale
            self.stack.append([wtb,dummy])

            dwtf = np.empty(wtf.shape)
            self.grad.append([dwtf,dummy])
            dwtb = np.empty(wtb.shape)
            self.grad.append([dwtb,dummy])

    def costAndGrad(self,data,labels):
        
        T = data.shape[1]
        
        # forward prop
        self.hActs[0] = data

        if self.temporalLayer > 0:
            stack = self.stack[:-2]
            wtf,_ = self.stack[-2]
            wtb,_ = self.stack[-1]
            grad = self.grad[:-2]
            dwtf,_ = self.grad[-2]
            dwtb,_ = self.grad[-1]
        else:
            stack = self.stack
            grad = self.grad
 
        i = 1
        for w,b in stack:
            self.hActs[i] = np.dot(w,self.hActs[i-1])
            self.hActs[i] += b

            # forward prop through time
            if i == self.temporalLayer:
                preActs = np.array(self.hActs[i])
                actsForward = np.empty(preActs.shape)
                actsForward[:,0] = preActs[:,0]
                actsForward[preActs[:,0]<=0,0] = 0.0 
                actsBackward = np.empty(preActs.shape)
                actsBackward[:,-1] = preActs[:,-1]
                actsBackward[preActs[:,-1]<=0,-1] = 0.0 
                for t in xrange(1,T):
                    actsForward[:,t] = np.dot(wtf,actsForward[:,t-1]) + preActs[:,t]
                    actsBackward[:,-t-1] = np.dot(wtb,actsBackward[:,-t]) + preActs[:,-t-1]
                    actsForward[actsForward[:,t]<=0,t] = 0.0
                    actsBackward[actsBackward[:,-t-1]<=0,-t-1] = 0.0
                self.hActs[i][:] = actsForward + actsBackward

            if i <= self.numLayers and i != self.temporalLayer:
                # hard relu
                self.hActs[i][self.hActs[i]<0.0] = 0.0
            i += 1

        # Subtract max activation
        probs = self.hActs[-1] - self.hActs[-1].max(axis=0)[None,:]

        # Softmax
        probs = np.exp(probs)
        probs /= probs.sum(axis=0)[None,:]
        cost, deltasC, skip = ctc.ctc_loss(np.asfortranarray(probs),labels,blank=0)

        if skip:
            return cost,self.grad,skip

        # back prop
        i = self.numLayers 
        self.deltasOut = None
        self.deltasIn = None
        deltasIn,deltasOut = deltasC,self.deltasOut
        for w,b in reversed(stack):
            # compute gradient
            grad[i][0] = np.dot(deltasIn,self.hActs[i].T)
            grad[i][1] = deltasIn.sum(axis=1)[:,None]

            # compute next layer deltas
            if i > 0:
                deltasOut = np.dot(w.T,deltasIn)

            # backprop through time
            if i == self.temporalLayer:
                tmpGradF = np.sign(actsForward)
                tmpGradB = np.sign(actsBackward)
                deltasForward = np.array(deltasOut)
                deltasForward[:,-1] *= tmpGradF[:,-1]
                deltasBackward = np.array(deltasOut)
                deltasBackward[:,0] *= tmpGradB[:,0]
                for t in xrange(1,T):
                    deltasForward[:,-t-1] = tmpGradF[:,-t-1]*(deltasForward[:,-t-1]+np.dot(wtf.T,deltasForward[:,-t]))
                    deltasBackward[:,t] = tmpGradB[:,t]*(deltasBackward[:,t]+np.dot(wtb.T,deltasBackward[:,t-1]))

                # Compute temporal gradient
                dwtb[:] = np.dot(deltasBackward[:,:-1],actsBackward[:,1:].T)
                dwtf[:] = np.dot(deltasForward[:,1:],actsForward[:,:-1].T)
                deltasOut = deltasForward + deltasBackward

            if i > 0 and i != self.temporalLayer:
                tmpGrad = np.sign(self.hActs[i])
                deltasOut *= tmpGrad

            if i == self.numLayers:
                deltasIn = self.deltasIn

            deltasIn,deltasOut = deltasOut,deltasIn
            i -= 1

        return cost,self.grad,skip

    def check_grad(self,data,labels,epsilon=1e-6):
        cost,grad,_ = self.costAndGrad(data,labels)

        for param,delta in zip(self.stack,grad):
            print "new params"
            w,b = param 
            dw,db = delta
            for i in xrange(w.shape[0]):
                for j in xrange(w.shape[1]):
                    w[i,j] += epsilon
                    costP,_,_ = self.costAndGrad(data,labels)
                    numGrad = (costP - cost) / epsilon
                    w[i,j] -= epsilon
                    if np.abs(dw[i,j] - numGrad) > 1e-4:
                        #raise ValueError
                        print "ERROR"
                    print "Analytic %f, Numeric %f, Diff %.9f."%(dw[i,j],numGrad,dw[i,j]-numGrad)
            """
            b[i] += epsilon
            costP,_,_ = self.costAndGrad(data,labels)
            numGrad = (costP - cost) / epsilon
            b[i] -= epsilon
            if np.abs(db[i] - numGrad) > 1e-4:
                print "ERROR B"
            print "Analytic %f, Numeric %f"%(dw[i,j],numGrad)
            """


if __name__=='__main__':
    np.random.seed(33)
    layerSize = 30
    numLayers = 3

    inputDim = 20
    maxUttLen = 10
    outputDim = 6

    data = np.random.randn(inputDim,maxUttLen)
    labels = np.arange(3).astype(np.int32)
    rnncp = RNNet(inputDim,outputDim,layerSize,numLayers,maxUttLen,temporalLayer=2)
    rnncp.initParams()
    cost,gradcp,_ = rnncp.costAndGrad(data,labels)
    print "COST %.9f"%cost

    rnncp.check_grad(data,labels)

