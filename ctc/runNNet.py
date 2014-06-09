import optparse
import numpy as np
import gnumpy as gp
import cPickle as pickle

import sgd
import nnet
import dataLoader as dl

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
	    default="/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/swbd/s5b/exp/train_ctc/")
    parser.add_option("--numFiles",dest="numFiles",type="int",default=384)
    parser.add_option("--inputDim",dest="inputDim",type="int",default=41*15)
    parser.add_option("--rawDim",dest="rawDim",type="int",default=41*15)
    parser.add_option("--outputDim",dest="outputDim",type="int",default=34)

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
	
    loader = dl.DataLoader(opts.dataDir,opts.rawDim,opts.inputDim)
    nn = nnet.NNet(opts.inputDim,opts.outputDim,opts.layers)
    nn.initParams()

    # Load model if exists
    if opts.inFile is not None:
        with open(opts.inFile,'r') as fid:
            _ = pickle.load(fid)
            _ = pickle.load(fid)
            _ = pickle.load(fid)
            nn.fromFile(fid)

    SGD = sgd.SGD(nn,alpha=opts.step,optimizer=opts.optimizer,
		  momentum=opts.momentum)

    # Setup some random keys for tracing
    with open('randKeys.bin','r') as fid:
	traceK = pickle.load(fid)
    for k in traceK:
	nn.hist[k] = []

    # Training
    fn = 0 # File number
    for _ in range(opts.epochs):
	for i in np.random.permutation(opts.numFiles)+1:
	    data_dict,alis,keys,sizes = loader.loadDataFileDict(i)

	    SGD.run_seq(data_dict,alis,keys,sizes)

        ## TODO changed to save/anneal every 100 files for swbd
        #fn += 1
        #if fn%100 == 0:
        SGD.alpha = SGD.alpha / opts.anneal
        with open(opts.outFile,'w') as fid:
            pickle.dump(opts,fid)
            pickle.dump(SGD.costt,fid)
            pickle.dump(nn.hist,fid)
            nn.toFile(fid)

def test(opts):
    import editDistance as ed

    print "Testing model %s"%opts.inFile

    phone_map = get_phone_map_swbd()

    with open(opts.inFile,'r') as fid:
	old_opts = pickle.load(fid)
	_ = pickle.load(fid)
	_ = pickle.load(fid)
	loader = dl.DataLoader(opts.dataDir,old_opts.rawDim,old_opts.inputDim)
	nn = nnet.NNet(old_opts.inputDim,old_opts.outputDim,old_opts.layers,train=False)
	nn.initParams()
	nn.fromFile(fid)

    totdist = numphones = 0

    fid = open('hyp.txt','w')
    for i in range(1,opts.numFiles+1):
	data_dict,alis,keys,sizes = loader.loadDataFileDict(i)
	for k in keys:
	    gp.free_reuse_cache()
	    hyp,_ = nn.costAndGrad(data_dict[k])
	    hyp = [phone_map[h] for h in hyp]
	    ref = [phone_map[int(r)] for r in alis[k]]
            print "Hyp %d -- Ref %d"%(len(hyp),len(ref))
	    dist,ins,dels,subs,corr = ed.edit_distance(ref,hyp)
	    print "Distance %d/%d"%(dist,len(ref))
	    fid.write(k+' '+' '.join(hyp)+'\n')
	    totdist += dist
	    numphones += len(alis[k])

    fid.close()
    print "PER : %f"%(100*totdist/float(numphones))

def get_phone_map_swbd():
    kaldi_base = "/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/swbd/s5b/"
    with open(kaldi_base+'ctc-utils/chars.txt','r') as fid:
	labels = [l.strip().split() for l in fid.readlines()]
        labels = dict((int(k),v) for v,k in labels)
    return labels



def get_phone_map_timit():
    kaldi_base = "/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/timit/s5/"
    with open(kaldi_base+'data/lang/phones/sets61.txt','r') as fid:
	sets_txt = [l.strip().split()[0] for l in fid.readlines()]
    with open(kaldi_base+'data/lang/phones/sets61.int','r') as fid:
	sets_int = [int(l.strip().split()[0]) for l in fid.readlines()]
    int_to_p61 = dict(zip(sets_int,sets_txt))

    p61_p39 = {}
    with open(kaldi_base+'conf/phones.60-48-39.map','r') as fid:
	k_v = [l.strip().split() for l in fid.readlines()]
	for p in k_v:
	    if len(p)==3:
		p61_p39[p[0]]=p[-1]
	    else:
		p61_p39[p[0]]='' # e.g. q maps to nothing
    phone_map = dict((k,p61_p39[int_to_p61[k]]) for k in int_to_p61.keys())
    return phone_map


if __name__=='__main__':
    run()


