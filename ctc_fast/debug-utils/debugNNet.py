import numpy as np
import cPickle as pickle
import cudamat as cm

import sgd
import rnnet as rnnet
import dataLoader as dl
import pdb

inFile = "models/swbd_layers_5_2048_temporal_3_step_1e-5_mom_.95_anneal_1.3.bin"

np.random.seed(33)
import random
random.seed(33)

# Load model if specified
with open(inFile,'r') as fid:
    opts = pickle.load(fid)
    loader = dl.DataLoader(opts.dataDir,opts.rawDim,opts.inputDim)

    nn = rnnet.NNet(opts.inputDim,opts.outputDim,opts.layerSize,opts.numLayers,
                opts.maxUttLen,temporalLayer=opts.temporalLayer)
    nn.initParams()
    SGD = sgd.SGD(nn,opts.maxUttLen,alpha=opts.step,momentum=opts.momentum)
    SGD.expcost = pickle.load(fid)
    SGD.it = 100
    nn.fromFile(fid)
    velocity = pickle.load(fid)
    for (w,b),(wv,bv) in zip(velocity,SGD.velocity):
        wv.copy_to_host()
        bv.copy_to_host()
        wv.numpy_array[:] = w[:]
        bv.numpy_array[:] = b[:]
        wv.copy_to_device()
        bv.copy_to_device()


# Training
pdb.set_trace()
for i in np.random.permutation(opts.numFiles)+1:
    print "File %d"%i
    if i != 96:
        nk = 500
        if i == 384:
            nk = 139
        random.shuffle(np.arange(nk).tolist())
    else:
        data_dict,alis,keys,sizes = loader.loadDataFileDict(i)
        SGD.run(data_dict,alis,keys,sizes)
