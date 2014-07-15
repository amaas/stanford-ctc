import cudamat as cm
import ctc_fast as ctc
import numpy as np
import pdb

class NNet:

    def __init__(self,inputDim,outputDim,layerSize,numLayers,maxBatch,
            train=True,temporalLayer=-1):
        # Initialize cublas
        cm.cublas_init()

        self.outputDim = outputDim
        self.inputDim = inputDim
        self.layerSize = layerSize
        self.numLayers = numLayers
        self.layerSizes = [layerSize]*numLayers
        self.maxBatch = maxBatch
        self.train = train

        if not self.train:
            np.seterr(all='ignore')

        if temporalLayer <= 0 or temporalLayer >= numLayers:
            self.temporalLayer = -1
        else:
            self.temporalLayer = temporalLayer

        self.maxAct = 20.0

    def initParams(self):
	"""
	Initialize parameters using 6/sqrt(fanin+fanout)
	"""
        sizes = [self.inputDim]+self.layerSizes+[self.outputDim]
        scales = [np.sqrt(6)/np.sqrt(n+m) for n,m in zip(sizes[:-1],sizes[1:])]
        self.stack = [[np.random.rand(m,n)*2*s-s,np.zeros((m,1))] \
                            for n,m,s in zip(sizes[:-1],sizes[1:],scales)]
        self.hActs_M = [cm.empty((s,self.maxBatch)) for s in sizes]

        if self.train:
            # Now assuming that all layers are the same size
            self.grad = [[cm.empty(w.shape),cm.empty(b.shape)] for w,b in self.stack]
            self.deltasC_M = cm.empty((self.outputDim,self.maxBatch))
            self.deltasOut_M = cm.empty((sizes[1],self.maxBatch)) 
            self.deltasIn_M = cm.empty((sizes[1],self.maxBatch)) 
            self.tmpGrad_M = cm.empty((self.layerSize,self.maxBatch))
 
        # Allocate memory once here and reuse
        # Store probs
        self.probs_M = cm.empty((self.outputDim,self.maxBatch))
        # Store col max
        self.rowVec_M = cm.empty((1,self.maxBatch))
       
        self.stack = [[cm.CUDAMatrix(w),cm.CUDAMatrix(b)]
                    for w,b in self.stack]

        if self.temporalLayer > 0:
            # dummy bias used for temporal layer
            dummy = cm.empty((1,1))
            dummy.assign(0.0)

            scale = np.sqrt(6)/np.sqrt(self.layerSize*2)
            wt = 2*scale*np.random.rand(self.layerSize,self.layerSize)-scale
            wt = cm.CUDAMatrix(wt)
            self.stack.append([wt,dummy])

            if self.train:
                dwt = cm.empty((self.layerSize,self.layerSize))
                self.grad.append([dwt,dummy])

                self.deltaTemp_M = cm.empty((self.layerSize,self.maxBatch))


    def setViews(self,batchSize):
        """
        Sets view of gpu memory to be correct size for utterance.
        """
        assert batchSize <= self.maxBatch, "Batch size exceeds max batch"
        self.hActs = [H.get_col_slice(0,batchSize) for H in self.hActs_M]
        self.probs = self.probs_M.get_col_slice(0,batchSize)
        self.rowVec = self.rowVec_M.get_col_slice(0,batchSize)
        if self.train:
            self.deltasC = self.deltasC_M.get_col_slice(0,batchSize)
            self.deltasOut = self.deltasOut_M.get_col_slice(0,batchSize)
            self.deltasIn = self.deltasIn_M.get_col_slice(0,batchSize)
            self.tmpGrad = self.tmpGrad_M.get_col_slice(0,batchSize)
            if self.temporalLayer > 0:
                self.deltaTemp = self.deltaTemp_M.get_col_slice(0,batchSize)

    def costAndGrad(self,data,labels=None):
        
        T = data.shape[1]
        self.setViews(T)

        if self.temporalLayer > 0:
            stack = self.stack[:-1]
            wt,_ = self.stack[-1]
            if self.train:
                grad = self.grad[:-1]
                dwt,_ = self.grad[-1]
        else:
            stack = self.stack
            if self.train:
                grad = self.grad
        
        # forward prop 
        self.hActs[0].assign(cm.CUDAMatrix(data))

        i = 1
        for w,b in stack:
            cm.dot(w,self.hActs[i-1],self.hActs[i])
            self.hActs[i].add_col_vec(b)

            # forward prop through time
            if i == self.temporalLayer:
                for t in xrange(1,T):
                    self.hActs[i].minmax(0.0,self.maxAct,col=t-1)
                    cm.mvdot_col_slice(wt,self.hActs[i],t-1,self.hActs[i],t,beta=1.0)
                self.hActs[i].minmax(0.0,self.maxAct,col=T-1)

            if i <= self.numLayers and i != self.temporalLayer:
                # hard relu
                self.hActs[i].maximum(0.0)
            i += 1

        # Subtract max activation
        self.hActs[-1].max(axis=0,target=self.rowVec)
        self.hActs[-1].add_row_mult(self.rowVec,-1.0,target=self.probs)

        # Softmax
        cm.exp(self.probs)
        self.probs.sum(axis=0,target=self.rowVec)
        cm.pow(self.rowVec,-1.0,target=self.rowVec)
        self.probs.mult_by_row(self.rowVec)

        self.probs.copy_to_host()
	if not self.train:
	    return ctc.decode_best_path(self.probs.numpy_array.astype(np.float64))

        cost, deltas, skip = ctc.ctc_loss(self.probs.numpy_array.astype(np.float64),
                labels,blank=0)

        if skip:
            return cost,self.grad,skip

        self.deltasC.assign(cm.CUDAMatrix(deltas))

        # back prop
        i = self.numLayers 
        deltasIn,deltasOut = self.deltasC,self.deltasOut
        for w,b in reversed(stack):
            # compute gradient
            cm.dot(deltasIn,self.hActs[i].T,target=grad[i][0])
            deltasIn.sum(axis=1,target=grad[i][1])

            # compute next layer deltas
            if i > 0:
                cm.dot(w.T,deltasIn,target=deltasOut)

            # backprop through time
            if i == self.temporalLayer:
                self.hActs[i].within(0.0,self.maxAct,target=self.tmpGrad)
                self.deltaTemp.assign(0.0)
                for t in xrange(T-1,0,-1):
                    # Add in temporal delta
                    cm.mvdot_col_slice(wt.T,self.deltaTemp,t,deltasOut,t,beta=1.0)

                    # Push through activation fn
                    deltasOut.mult_slice(t,self.tmpGrad,t) 
                    self.deltaTemp.set_single_col(t-1,deltasOut,t)

 
                # Accumulate temporal gradient
                cm.dot(self.deltaTemp,self.hActs[i].T,
                        target=dwt)

                cm.mvdot_col_slice(wt.T,self.deltaTemp,0,deltasOut,0,beta=1.0)
                deltasOut.mult_slice(0,self.tmpGrad,0)

            if i > 0 and i != self.temporalLayer:
                self.hActs[i].sign(target=self.tmpGrad)
                deltasOut.mult(self.tmpGrad)

            if i == self.numLayers:
                deltasIn = self.deltasIn

            deltasIn,deltasOut = deltasOut,deltasIn
            i -= 1

        return cost,self.grad,skip

    def updateParams(self,scale,update):
        for params,paramsDel in zip(self.stack,update):
            w,b = params
            dw,db = paramsDel
            w.add_mult(dw, alpha=scale)
            b.add_mult(db, alpha=scale)

    def toFile(self,fid):
	"""
	Saves only the network parameters to the given fd.
	"""
	import cPickle as pickle
        stack = []
        for w,b in self.stack:
            w.copy_to_host()
            b.copy_to_host()
            stack.append([w.numpy_array,b.numpy_array])
	pickle.dump(stack,fid)

    def fromFile(self,fid):
	import cPickle as pickle
        stack = pickle.load(fid)
        for (w,b),(wi,bi) in zip(self.stack,stack):
            w.copy_to_host()
            b.copy_to_host()
            w.numpy_array[:] = wi[:]
            b.numpy_array[:] = bi[:]
            w.copy_to_device()
            b.copy_to_device()

    def check_grad(self,data,labels,epsilon=1e-3):
        cost,grad,_ = self.costAndGrad(data,labels)
        # TODO randomize grad check selection
        for param,delta in zip(self.stack,grad):
            w,b = param
            dw,db = delta
            dw.copy_to_host()
            w.copy_to_host()
            for i in xrange(w.shape[0]):
                for j in xrange(w.shape[1]):
                    w.numpy_array[i,j] += epsilon
                    w.copy_to_device()
                    costP,_,_ = self.costAndGrad(data,labels)
                    numGrad = (costP - cost) / epsilon
                    w.numpy_array[i,j] -= epsilon
                    print "Analytic %f, Numeric %f"%(dw.numpy_array[i,j],numGrad)

if __name__=='__main__':
    import dataLoader as dl
    np.random.seed(33)
    layerSize = 40
    numLayers = 3

    dataDir = "/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/swbd/s5b/exp/train_ctc/"
    inputDim = 41*15
    rawDim = 41*15
    outputDim = 35
    maxUttLen = 1500

    loader = dl.DataLoader(dataDir,rawDim,inputDim)
    data_dict,alis,keys,_ = loader.loadDataFileDict(1)
    data,labels = data_dict[keys[3]],np.array(alis[keys[3]],dtype=np.int32)

    rnn = NNet(inputDim,outputDim,layerSize,numLayers,maxUttLen,temporalLayer=2)
    rnn.initParams()
    cost,grad,_ = rnn.costAndGrad(data,labels)
    print "COST %.9f"%cost

    rnn.check_grad(data,labels)

