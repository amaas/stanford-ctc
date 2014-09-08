import numpy as np
import brnnet as rnn
import rnnetcpu as rnncp

import dataLoader as dl

layerSize = 100
numLayers = 3

dataDir = "/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/swbd/s5b/exp/train_ctc/"
inputDim = 41*15
rawDim = 41*15
outputDim = 35
maxUttLen = 1500

loader = dl.DataLoader(dataDir,rawDim,inputDim)
data_dict,alis,keys,_ = loader.loadDataFileDict(1)
data,labels = data_dict[keys[3]],np.array(alis[keys[3]],dtype=np.int32)

np.random.seed(33)
rnn = rnn.NNet(inputDim,outputDim,layerSize,numLayers,maxUttLen,temporalLayer=2)
rnn.initParams()
cost,grad,_ = rnn.costAndGrad(data,labels)
print "COST %.9f"%cost

np.random.seed(33)
rnncp = rnncp.RNNet(inputDim,outputDim,layerSize,numLayers,maxUttLen,temporalLayer=2)
rnncp.initParams()
cost,gradcp,_ = rnncp.costAndGrad(data,labels)
print "COST %.9f"%cost

def diff(ga,gb):
    ga.copy_to_host()
    print np.sum((ga.numpy_array-gb)**2)

for gas,gbs in zip(grad,gradcp):
    ga,ba = gas
    gb,bb = gbs
    diff(ga,gb)
    diff(ba,bb)
