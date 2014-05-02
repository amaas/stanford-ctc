import optparse
import numpy as np
import gnumpy as gp
import cPickle as pickle

import sgd
import nnet
import dataLoader as dl

gp.board_id_to_use = 0

def run(args=None):
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("--minibatch",dest="minibatch",type="int",default=256)
    parser.add_option("--layers",dest="layers",type="string",
	    default="100,100",help="layer1size,layer2size,...,layernsize")
    parser.add_option("--optimizer",dest="optimizer",type="string",
	    default="momentum")
    parser.add_option("--momentum",dest="momentum",type="float",
	    default=0.9)
    parser.add_option("--epochs",dest="epochs",type="int",default=1)
    parser.add_option("--step",dest="step",type="float",default=1e-2)
    parser.add_option("--outFile",dest="outFile",type="string",
	    default="models/test.bin")
    parser.add_option("--anneal",dest="anneal",type="float",default=1)

    (opts,args)=parser.parse_args(args)
    opts.layers = [int(l) for l in opts.layers.split(',')]

    # Other Config #
    dataDir = "/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/fisher_english/s5/exp/nn_train_1/"
    rawSize = 41*40
    inputSize = rawSize
    outputSize = 7793  
    numFiles = 300

    nn = nnet.NNet(inputSize,outputSize,opts.layers,opts.minibatch)
    nn.initParams()
    
    loader = dl.DataLoader(dataDir,rawSize,inputSize)

    SGD = sgd.SGD(nn,alpha=opts.step,minibatch=opts.minibatch,
		  optimizer=opts.optimizer,momentum=opts.momentum)

    for _ in range(opts.epochs):
        for i in np.random.permutation(numFiles)+1:
            data,labels,_,_=loader.loadDataFile(i)
            SGD.run(data,labels)
	    with open(opts.outFile,'w') as fid:
		pickle.dump(opts,fid)
		pickle.dump(SGD.costt,fid)


if __name__=='__main__':
    run()


