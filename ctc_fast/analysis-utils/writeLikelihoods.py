import optparse
import os, struct
import cPickle as pickle
import dataLoader as dl
import brnnet as rnnet
import numpy as np

def writeUttHeader(fid,key,uttSize,numClasses):
    """
    Writes header for each utterance in Kaldi Style, assumes 
    data written in float32 and C order. 
    """
    fid.write(key+' ') # write key
    # kaldi specific dat
    dat = struct.pack('b',0)
    fid.write(dat)
    fid.write('BFM ')
    floatsize = struct.pack('b',4)
    fid.write(floatsize)
    # num rows (utterance size)
    dat = struct.pack('i',uttSize)
    fid.write(dat)
    fid.write(floatsize)
    # num cols (num classes)
    dat = struct.pack('i',numClasses)
    fid.write(dat)

def writeLogLikes(loader,nn,fn, outDir, writePickle=False):

    data_dict,alis,keys,sizes = loader.loadDataFileDict(fn)

    fid = open(outDir+'/loglikelihoods%d.ark'%fn,'w')

    lik_dict = dict()

    print "Running file %d"%fn
    for i,k in enumerate(keys): 
        assert data_dict[k].shape[1] < nn.maxBatch,\
            "Need larger max utt length."
        writeUttHeader(fid,k,sizes[i],nn.outputDim)
        probs = nn.costAndGrad(data_dict[k])
        assert probs.dtype==np.float32,"Probs array malformed."
        assert probs.shape[0]==nn.outputDim,"Probs dimensions mismatch."
        probs = np.log(probs)
        probs.T.tofile(fid)
        lik_dict[k] = probs
    fid.close()

    if writePickle:
        print "Writing pickle for file %d"%fn
        with open(outDir+'/loglikelihoods_%d.pk'%fn,'wb') as f:
            pickle.dump(lik_dict,f)

    print "Done with file %d"%fn
    return lik_dict

def run(args=None):
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    # Data
    parser.add_option("--dataDir",dest="dataDir",type="string",
	    default="/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/swbd/s5b/exp/dev_ctc/")
    parser.add_option("--numFiles",dest="numFiles",type="int",default=10)
    parser.add_option("--maxUttLen",dest="maxUttLen",type="int",default=2000)

    parser.add_option("--inFile",dest="inFile",type="string",
	    default=None,help="Saved model")
    parser.add_option("--outFile",dest="outFile",type="string",
	    default=None,help="Location to store log likes")
    parser.add_option("--writePickle",action="store_true", dest="writePickle",
	    default=None,help="flag to write pickle files of likelihoods")
    
    (opts,args)=parser.parse_args(args)

    with open(opts.inFile,'r') as fid:
	old_opts = pickle.load(fid)
	_ = pickle.load(fid)
	loader = dl.DataLoader(opts.dataDir,old_opts.rawDim,old_opts.inputDim,load_ali=False)
	nn = rnnet.NNet(old_opts.inputDim,old_opts.outputDim,
                old_opts.layerSize,old_opts.numLayers,opts.maxUttLen,
                temporalLayer=old_opts.temporalLayer,train=False)
	nn.initParams()
	nn.fromFile(fid)

    for i in range(1,opts.numFiles+1):
        writeLogLikes(loader,nn,i, opts.outFile,opts.writePickle)

if __name__=='__main__':
    run()
