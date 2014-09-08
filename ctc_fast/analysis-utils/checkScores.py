import cPickle as pickle
import numpy as np
with open('scores.bin','r') as fid:
    hyps = np.array(pickle.load(fid))
    refs = np.array(pickle.load(fid))

# Want the score to be more positive, i.e. the number of 
# refs > hyps is of interest.
print "Scores: Avg refs %f, Avg hyps %f"%(np.mean(refs),np.mean(hyps))
print "Num refs better %d/%d"%(np.sum((refs-hyps)>0),refs.shape[0])
