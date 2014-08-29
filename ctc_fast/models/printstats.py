import cPickle as pickle
import numpy as np
import sys

model = sys.argv[-1]
with open(model) as fid:
    a = pickle.load(fid)
    b = np.array(pickle.load(fid))

stride = 195000
#stride = 37400
print b.shape[0]
for i in range(0,b.shape[0],stride):
    print np.mean(b[i:i+stride])

