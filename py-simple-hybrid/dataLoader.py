import numpy as np
import os

class DataLoader:
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
        
        alis = np.loadtxt(alisfile, np.int32, usecols=[0])
        alis = alis[:np.sum(sizes)]
        
        return data.T,alis,keys,sizes


if __name__=='__main__':
    dataDir = "/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/fisher_english/s5/exp/nn_train_1/"
    rawSize = 41*40
    inputSize = rawSize
    dl = DataLoader(dataDir,rawSize,inputSize)
    filenum = 1
    
    data,alis,_,_=dl.loadDataFile(filenum)
    print data.shape
    print alis.shape
