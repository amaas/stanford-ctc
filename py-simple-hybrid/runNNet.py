import sgd
import nnet
import dataLoader as dl
import numpy as np

def run():
    # Config #
    dataDir = "/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/swbd/s5b/exp/nn_train_fmllr_rand/"
    rawSize = 41*40
    inputSize = rawSize
    outputSize = 8986
    layerSizes = [1024]*2
    minibatch = 256
    epochs = 1
    numFiles = 300
    stepSize = 1e-3
    nn = nnet.NNet(inputSize,outputSize,layerSizes,minibatch)
    nn.initParams()
    
    loader = dl.DataLoader(dataDir,rawSize,inputSize)

    SGD = sgd.SGD(nn,alpha=stepSize,minibatch=minibatch)

    for _ in range(epochs):
        for i in np.random.permutation(numFiles)+1:
            data,labels,_,_=loader.loadDataFile(i)
            SGD.run(data,labels)


if __name__=='__main__':
    run()


