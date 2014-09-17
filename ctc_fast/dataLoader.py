import numpy as np
from itertools import izip
import os
import multiprocessing as mp

class DataLoader:
    def __init__(self,filedir_feat,rawsize,imgsize,filedir_ali=None, load_ali=True, load_data=True):
        """
        filedir_feat: directory for feature and key files
        filedir_ali: directory for alignment files. Assumed same as filedir if not given
        """
        self.filedir_feat = filedir_feat
        self.rawsize = rawsize
        self.imgsize = imgsize
        if filedir_ali is None:
            self.filedir_ali = filedir_feat
        else:
            self.filedir_ali = filedir_ali
        self.load_ali = load_ali
        self.load_data = load_data

    def getDataAsynch(self):
        assert self.p is not None, "Error in order of asynch calls."
        data_dict,alis,keys,sizes = self.p_conn.recv()
        self.p.join()
        return data_dict,alis,keys,sizes

    def loadDataFileAsynch(self,filenum):
        # spawn process to load datafile
        self.p_conn, c_conn = mp.Pipe()
        self.p = mp.Process(target=self.loadAndPipeFile,args=(filenum,c_conn))
        self.p.start()

    def loadAndPipeFile(self,filenum,conn):
        conn.send(self.loadDataFileDict(filenum))
        conn.close()

    def loadDataFile(self,filenum):
    
        keyfile = self.filedir_feat+'keys%d.txt'%filenum
        alisfile = self.filedir_ali+'alis%d.txt'%filenum
        datafile = self.filedir_feat+'feats%d.bin'%filenum

        keys = None
        sizes = None
        data = None
	alis = []
        if self.load_ali:
            with open(alisfile,'r') as fid:
                for l in fid.readlines():
                    l = l.split()
                    alis.append((l[0],l[1:]))
            alis = dict(alis)

        if self.load_data:

            if os.path.exists(keyfile):
                with open(keyfile,'r') as keyf:
                    uttdat = [u.split() for u in keyf.readlines()]
                sizes = [np.int32(u[1]) for u in uttdat]
                sizes = np.array(sizes)
                keys = [u[0] for u in uttdat]
            
            left = (self.rawsize-self.imgsize)/2
            right = left+self.imgsize
            
            data = np.fromfile(datafile,np.float32).reshape(-1,self.rawsize)
            data = data[:np.sum(sizes),left:right]                
            return data.T,alis,keys,sizes
        # return for case when no data loaded
        # use keys from alignments instead
        keys = alis.keys()
        return data,alis,keys,sizes

    def loadDataFileDict(self,filenum):
        """
        Loads a data file but stores input frames in a dictionary keyed by utterance
        Each input dictionary entry is a 2-D matrix of length equal to that utterance
        Other variables returned are the same as the original loader
        returns None for data when not set to load data
        """
        data_mat, alis, keys, sizes = self.loadDataFile(filenum)
        if self.load_data:
            data_dict = {}
            startInd = 0
            for k,s in izip(keys,sizes):
                endInd = startInd + s
                data_dict[k] = np.copy(data_mat[:,startInd:endInd])
                startInd = endInd

            # startInd = all frames means we loaded all data
            assert startInd, data_mat.shape[1]

            return data_dict, alis, keys, sizes
        return None, alis, keys, sizes
            
# Usage example / eyeball test below
if __name__=='__main__':
    dataDir = "/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/swbd/s5b/exp/train_ctc/"
    rawSize = 41*15
    inputSize = rawSize
    dl = DataLoader(dataDir,rawSize,inputSize)
    filenum = 1
    
    data,alis,keys,sizes = dl.loadDataFile(filenum)
    print "Data shape (featDim x frames): (%d,%d) "%data.shape
    print "Number of transcripts: %d"%len(alis.keys())
    print "Number of keys: %d"%len(keys)
    print "Number of frames: %d"%sum(sizes)
