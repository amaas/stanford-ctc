import optparse
import numpy as np
#import gnumpy as gp
import cPickle as pickle

import sgd
import nnet
import timitLoader as dl

#gp.board_id_to_use = 0

def run(args=None):
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    # Architecture
    parser.add_option("--layers",dest="layers",type="string",
	    default="100,100",help="layer1size,layer2size,...,layernsize")

    # Optimization
    parser.add_option("--optimizer",dest="optimizer",type="string",
	    default="momentum")
    parser.add_option("--momentum",dest="momentum",type="float",
	    default=0.9)
    parser.add_option("--epochs",dest="epochs",type="int",default=1)
    parser.add_option("--step",dest="step",type="float",default=1e-4)
    parser.add_option("--anneal",dest="anneal",type="float",default=1)

    # Data
    parser.add_option("--dataDir",dest="dataDir",type="string",
	    default="/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/timit/s5/exp/nn_train/")
    parser.add_option("--numFiles",dest="numFiles",type="int",default=19)
    parser.add_option("--inputDim",dest="inputDim",type="int",default=41*23)
    parser.add_option("--rawDim",dest="rawDim",type="int",default=41*23)
    parser.add_option("--outputDim",dest="outputDim",type="int",default=40)

    parser.add_option("--outFile",dest="outFile",type="string",
	    default="models/test.bin")

    (opts,args)=parser.parse_args(args)
    opts.layers = [int(l) for l in opts.layers.split(',')]

    nn = nnet.NNet(opts.inputDim,opts.outputDim,opts.layers)
    nn.initParams()

    loader = dl.TimitLoader(opts.dataDir,opts.rawDim,opts.inputDim)

    SGD = sgd.SGD(nn,alpha=opts.step,optimizer=opts.optimizer,
		  momentum=opts.momentum)

    # Setup some random keys for tracing
    with open('randKeys.txt','r') as fid:
	traceK = pickle.load(fid)
    for k in traceK:
	nn.hist[k] = []

    for _ in range(opts.epochs):
        for i in np.random.permutation(opts.numFiles)+1:
            data_dict,alis,keys,sizes = loader.loadDataFileDict(i)

            SGD.run_seq(data_dict,alis,keys,sizes)

	SGD.alpha = SGD.alpha / opts.anneal
	with open(opts.outFile,'w') as fid:
	    pickle.dump(opts,fid)
	    pickle.dump(SGD.costt,fid)
	    pickle.dump(nn.hist,fid)


if __name__=='__main__':
    run()


