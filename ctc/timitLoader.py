import numpy as np
import os

class TimitLoader:
    def __init__(self,filedir,rawsize,imgsize):
        self.filedir = filedir
        self.rawsize = rawsize
        self.imgsize = imgsize

    def loadDataFile(self,filenum):
    
        keyfile = self.filedir+'keys%d.txt'%filenum
        alisfile = self.filedir+'alis%d.txt'%filenum
        datafile = self.filedir+'feats%d.bin'%filenum

        keys = None
        sizes = None
        if os.path.exists(keyfile):
            keyf = open(keyfile,'r')
            uttdat = [u.split() for u in keyf.readlines()]
            keyf.close()
            sizes = [int(u[1]) for u in uttdat]
            sizes = np.array(sizes)
            keys = [u[0] for u in uttdat]
            
        left = (self.rawsize-self.imgsize)/2
        right = left+self.imgsize
        
        data = np.fromfile(datafile,np.float32).reshape(-1,self.rawsize)
        data = data[:np.sum(sizes),left:right]
        
	alis = []
	with open(alisfile,'r') as fid:
	    for l in fid.readlines():
		l = l.split()
		alis.append((l[0],l[1:]))
	alis = dict(alis)
        
        return data.T,alis,keys,sizes


# Usage example / eyeball test below
if __name__=='__main__':
    dataDir = "/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/timit/s5/exp/nn_train/"
    rawSize = 41*23
    inputSize = rawSize
    dl = TimitLoader(dataDir,rawSize,inputSize)
    filenum = 1
    
    data,alis,keys,sizes =dl.loadDataFile(filenum)
    print "Data shape (featDim x frames): (%d,%d) "%data.shape
    print "Number of transcripts: %d"%len(alis.keys())
    print "Number of keys: %d"%len(keys)
    print "Number of frames: %d"%sum(sizes)
