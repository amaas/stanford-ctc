import numpy as np
import cudamat as cm
import pickle
import random
import pdb

class SGD:

    def __init__(self,model,maxBatch,alpha=1e-2,optimizer='nesterov',
                 momentum=0.9):
        self.model = model
        self.maxBatch = maxBatch
        self.it = 0
        self.momentum = momentum # momentum
        self.alpha = alpha # learning rate
        self.optimizer = optimizer
        self.maxGNorm = 1500 # gradient clip norm value

        if self.optimizer == 'nesterov':
            self.velocity = [[cm.CUDAMatrix(np.zeros(w.shape)),
                              cm.CUDAMatrix(np.zeros(b.shape))] 
                              for w,b in self.model.stack]
        elif self.optimizer == 'adagrad':
            epsilon = 0.0
            self.gradt = [[cm.CUDAMatrix(epsilon+np.zeros(w.shape)),
                           cm.CUDAMatrix(epsilon+np.zeros(b.shape))] 
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
                print ("SKIPPING utt exceeds batch length "
                       "(Utterance length %d).")%mb_data.shape[1]
                continue

            mb_labels = np.array(alis[k],dtype=np.int32)

            if mb_data.shape[1] < mb_labels.shape[0]:
                print ("SKIPPING utt frames less than label length "
                       "(Utterance length %d, Num Labels %d)."
                       "")%(mb_data.shape[1],mb_labels.shape[0])
                continue


            if self.optimizer == 'nesterov':
                # w = w+mom*velocity (evaluate gradient at future point)
                self.model.updateParams(mom,self.velocity)

            cost,grad,skip = self.model.costAndGrad(mb_data,mb_labels)

            if self.optimizer == 'nesterov':
                # undo update
                # w = w-mom*velocity
                self.model.updateParams(-mom,self.velocity)

            # Compute norm of all parameters as one vector
            gnorm = 0.0
            for dw,db in grad:
                gnorm += dw.euclid_norm()**2
                gnorm += db.euclid_norm()**2
            gnorm = np.sqrt(gnorm)

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
            if self.optimizer == 'nesterov':
                alph = self.alpha
                if gnorm > self.maxGNorm:
                    alph *= (self.maxGNorm/gnorm)
                for vs,gs in zip(self.velocity,grad):
                    vw,vb = vs 
                    dw,db = gs
                    vw.mult(mom)
                    vb.mult(mom)
                    vw.add_mult(dw,alpha=-alph)
                    vb.add_mult(db,alpha=-alph)
                update = self.velocity
                scale = 1.0

            elif self.optimizer == 'adagrad':
                delta = 1e-10
                for gts,gs in zip(self.gradt,grad):
                    dwt,dbt = gts 
                    dw,db = gs
                    gamma = 1. - 1./(1e-2*self.it+1)
                    cm.add_pow(dwt,dw,2,alpha=gamma,target=dwt)
                    cm.add_pow(dbt,db,2,alpha=gamma,target=dbt)
                    dwt.add(delta,target=dwt)
                    dbt.add(delta,target=dbt)
                    cm.mult_pow(dw,dwt,-0.5,target=dw)
                    cm.mult_pow(db,dbt,-0.5,target=db)
                    dwt.add(-delta,target=dwt)
                    dbt.add(-delta,target=dbt)
                update = grad
                scale = -self.alpha

	    # update params
	    self.model.updateParams(scale,update)

            if self.it%1 == 0:
                print ("Iter %d : Cost=%.4f, ExpCost=%.4f, GradNorm=%.4f, "
                       "SeqLen=%d, NumFrames=%d.")%(self.it,cost,
                       self.expcost[-1],gnorm,mb_labels.shape[0],
                       mb_data.shape[1])
        
