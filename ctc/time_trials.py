import ctc_fast as cf
import ctc as cl
import numpy as np
import time

def run(fn,params,seq,trials=1):
    for _ in range(trials):
        cost,_,_ = fn(params, seq)
        print "Negative loglikelihood is %.9f"%(cost)

if __name__=='__main__':
    
    np.random.seed(33)
    numPhones = 40 
    seqLen = 125
    uttLen = 1200

    seq = np.floor(np.random.rand(seqLen)*numPhones)
    seq = seq.astype(np.int32)

    params = np.random.randn(numPhones,uttLen) 
    params[seq,np.arange(seqLen)] = 3 
    params[0,seqLen:] = 3
    params = np.exp(params)
    params = params/np.sum(params,axis=0)


    start_cf = time.time()
    run(cf.ctc_loss,params,seq,trials=1)
    end_cf = time.time()

    start_cl = time.time()
    run(cl.ctc_loss,params,seq,trials=1)
    end_cl = time.time()

    print "Cython time = %f  Python time = %f"%(end_cf-start_cf,end_cl-start_cl)

