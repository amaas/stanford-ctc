import numpy as np
import cudamat as cm
import pickle
import random


class SGD:

    def __init__(self,model,maxBatch,alpha=1e-2,momentum=0.9):
        self.model = model
        self.maxBatch = maxBatch
        self.it = 0
        self.momentum = momentum # momentum
        self.alpha = alpha # learning rate
        self.velocity = [[cm.CUDAMatrix(np.zeros(w.shape)),cm.CUDAMatrix(np.zeros(b.shape))] 
                          for w,b in self.model.stack]
	self.costt = []
	self.expcost = []

    def run(self,data_dict,alis,keys,sizes):
        """
        Runs stochastic gradient descent with nesterov acceleration.
        Model is objective.  
        """
        
        # momentum setup
        momIncrease = 10
        mom = 0.5

        # randomly select minibatch
       	random.shuffle(keys)

        for k in keys:
            self.it += 1

            if self.it > momIncrease:
                mom = self.momentum

            mb_data = data_dict[k]
            if mb_data.shape[1] > self.maxBatch:
                print "SKIPPING utt exceeds batch length. (Utterance length %d)"%mb_data.shape[1]
                continue

            # w = w+mom*velocity (evaluate gradient at future point)
            self.model.updateParams(mom,self.velocity)
            mb_labels = np.array(alis[k],dtype=np.int32)

            cost,grad,skip = self.model.costAndGrad(mb_data,mb_labels)

	    # undo update
            # w = w-mom*velocity
            self.model.updateParams(-mom,self.velocity)

	    if skip:
		print "SKIPPING: Key=%s, Cost=%f, SeqLen=%d, NumFrames=%d."%(k,
			    cost,mb_labels.shape[0],mb_data.shape[1])
		continue

            if np.isfinite(cost):
                # compute exponentially weighted cost
                if self.it > 1:
                    self.expcost.append(.01*cost + .99*self.expcost[-1])
                else:
                    self.expcost.append(cost)
                self.costt.append(cost)

            # velocity = mom*velocity - alpha*grad
            for vs,gs in zip(self.velocity,grad):
                vw,vb = vs 
                dw,db = gs
                vw.mult(mom)
                vb.mult(mom)
                vw.add_mult(dw,alpha=-self.alpha)
                vb.add_mult(db,alpha=-self.alpha)

            update = self.velocity
            scale = 1.0

	    # update params
	    self.model.updateParams(scale,update)

            if self.it%1 == 0:
                print "Iter %d : Cost=%.4f, ExpCost=%.4f, SeqLen=%d, NumFrames=%d."%(self.it,
                        cost,self.expcost[-1],mb_labels.shape[0],mb_data.shape[1])
        
