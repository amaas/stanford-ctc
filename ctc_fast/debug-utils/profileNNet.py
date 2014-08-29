import optparse
import numpy as np
import cPickle as pickle

import sgd
import rnnet as nnet
import dataLoader as dl
import random

np.random.seed(33)
random.seed(33)

layerSize = 2048
numLayers = 5
momentum = 0.95
epochs = 2
step = 1e-6
anneal = 1.1

dataDir = "/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/swbd/s5b/exp/train_ctc/"
inputDim = 41*15
rawDim = 41*15
outputDim = 35
maxUttLen = 1500
temporalLayer = 3


loader = dl.DataLoader(dataDir,rawDim,inputDim)
nn = nnet.NNet(inputDim,outputDim,layerSize,numLayers,maxUttLen,temporalLayer=temporalLayer)
nn.initParams()

SGD = sgd.SGD(nn,maxUttLen,alpha=step,momentum=momentum)

data_dict,alis,keys,sizes = loader.loadDataFileDict(1)

# Training
for e in range(epochs):
    print "Epoch %d"%e
    SGD.run(data_dict,alis,keys,sizes)
    SGD.alpha /= anneal

