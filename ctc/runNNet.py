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

    parser.add_option("--test",action="store_true",dest="test",default=False)

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
    parser.add_option("--anneal",dest="anneal",type="float",default=1,
	    help="Sets (learning rate := learning rate / anneal) after each epoch.")

    # Data
    parser.add_option("--dataDir",dest="dataDir",type="string",
	    default="/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/timit/s5/exp/nn_train/")
    parser.add_option("--numFiles",dest="numFiles",type="int",default=19)
    parser.add_option("--inputDim",dest="inputDim",type="int",default=41*23)
    parser.add_option("--rawDim",dest="rawDim",type="int",default=41*23)
    parser.add_option("--outputDim",dest="outputDim",type="int",default=62)

    parser.add_option("--outFile",dest="outFile",type="string",
	    default="models/test.bin")
    parser.add_option("--inFile",dest="inFile",type="string",
	    default=None)

    (opts,args)=parser.parse_args(args)
    opts.layers = [int(l) for l in opts.layers.split(',')]


    # Testing
    if opts.test:
	test(opts)
	return
	
    loader = dl.TimitLoader(opts.dataDir,opts.rawDim,opts.inputDim)
    nn = nnet.NNet(opts.inputDim,opts.outputDim,opts.layers)
    nn.initParams()
    SGD = sgd.SGD(nn,alpha=opts.step,optimizer=opts.optimizer,
		  momentum=opts.momentum)

    # Setup some random keys for tracing
    with open('randKeys.bin','r') as fid:
	traceK = pickle.load(fid)
    for k in traceK:
	nn.hist[k] = []

    # Training
    for _ in range(opts.epochs):
	for i in np.random.permutation(opts.numFiles)+1:
	    data_dict,alis,keys,sizes = loader.loadDataFileDict(i)

	    SGD.run_seq(data_dict,alis,keys,sizes)

	SGD.alpha = SGD.alpha / opts.anneal
	with open(opts.outFile,'w') as fid:
	    pickle.dump(opts,fid)
	    pickle.dump(SGD.costt,fid)
	    pickle.dump(nn.hist,fid)
	    nn.toFile(fid)

def test(opts):
    kaldi_base = "/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/timit/s5/"
    with open(kaldi_base+'data/lang/phones/sets.txt','r') as fid:
	sets_txt = [l.strip().split()[0] for l in fid.readlines()]
    with open(kaldi_base+'data/lang/phones/sets.int','r') as fid:
	sets_int = [int(l.strip().split()[0]) for l in fid.readlines()]
    phones = dict(zip(sets_int,sets_txt))

    print "Testing file %s"%opts.inFile
    with open(opts.inFile,'r') as fid:
	old_opts = pickle.load(fid)
	_ = pickle.load(fid)
	_ = pickle.load(fid)
	loader = dl.TimitLoader(opts.dataDir,old_opts.rawDim,old_opts.inputDim)
	nn = nnet.NNet(old_opts.inputDim,old_opts.outputDim,old_opts.layers,train=False)
	nn.initParams()
	nn.fromFile(fid)

    totdist = numphones = 0

    fid = open('hyp.txt','w')
    for i in range(1,opts.numFiles+1):
	data_dict,alis,keys,sizes = loader.loadDataFileDict(i)
	for k in keys:
	    hyp,dist = nn.costAndGrad(data_dict[k],np.array(alis[k],dtype=np.int32))
	    fid.write(k+' '+' '.join(phones[h] for h in hyp)+'\n')
	    totdist += dist
	    numphones += len(alis[k])

    fid.close()
    print "WER : %f"%(100*totdist/float(numphones))

if __name__=='__main__':
    run()


