import os
import optparse
from os.path import join as pjoin
import numpy as np
import cPickle as pickle
import cudamat as cm

from writeLikelihoods import writeLogLikes

import sgd
import nnets.brnnet as rnnet
import dataLoader as dl

from decoder_config import SCAIL_DATA_DIR, DATASET

def run(args=None):
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option(
        "--test", action="store_true", dest="test", default=False)

    # GPU Device
    parser.add_option("--deviceId", dest="deviceId", type="int", default=0)

    # Architecture
    parser.add_option(
        "--layerSize", dest="layerSize", type="int", default=2048)
    parser.add_option("--numLayers", dest="numLayers", type="int", default=5)
    parser.add_option(
        "--temporalLayer", dest="temporalLayer", type="int", default=-1)

    # Optimization
    parser.add_option("--momentum", dest="momentum", type="float",
                      default=0.9)
    parser.add_option("--epochs", dest="epochs", type="int", default=3)
    parser.add_option("--step", dest="step", type="float", default=1e-4)
    parser.add_option("--anneal", dest="anneal", type="float", default=1,
                      help="Sets (learning rate := learning rate / anneal) after each epoch.")

    # Data
    parser.add_option("--dataDir", dest="dataDir", type="string",
                      default="/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/swbd/s5b/exp/train_ctc/")
    parser.add_option("--numFiles", dest="numFiles", type="int", default=384)
    parser.add_option(
        "--inputDim", dest="inputDim", type="int", default=41 * 15)
    parser.add_option("--rawDim", dest="rawDim", type="int", default=41 * 15)
    parser.add_option("--outputDim", dest="outputDim", type="int", default=35)
    parser.add_option(
        "--maxUttLen", dest="maxUttLen", type="int", default=1500)

    # Save/Load
    parser.add_option("--outFile", dest="outFile", type="string",
                      default="models/test.bin")
    parser.add_option("--inFile", dest="inFile", type="string",
                      default=None)

    (opts, args) = parser.parse_args(args)

    # seed for debugging, turn off when stable
    np.random.seed(33)
    import random
    random.seed(33)

    cm.cuda_set_device(opts.deviceId)

    # Testing
    if opts.test:
        test(opts)
        return

    loader = dl.DataLoader(opts.dataDir, opts.rawDim, opts.inputDim)

    nn = rnnet.NNet(opts.inputDim, opts.outputDim, opts.layerSize, opts.numLayers,
                    opts.maxUttLen, temporalLayer=opts.temporalLayer)
    nn.initParams()

    # Load model if specified
    if opts.inFile is not None:
        with open(opts.inFile, 'r') as fid:
            _ = pickle.load(fid)
            _ = pickle.load(fid)
            nn.fromFile(fid)

    SGD = sgd.SGD(nn, opts.maxUttLen, alpha=opts.step, momentum=opts.momentum)

    # Training
    import time
    for _ in range(opts.epochs):
        perm = np.random.permutation(opts.numFiles) + 1
        loader.loadDataFileAsynch(perm[0])
        for i in xrange(perm.shape[0]):
            start = time.time()
            data_dict, alis, keys, sizes = loader.getDataAsynch()
            # Prefetch
            if i + 1 < perm.shape[0]:
                loader.loadDataFileAsynch(perm[i + 1])
            SGD.run(data_dict, alis, keys, sizes)
            end = time.time()
            print "File time %f" % (end - start)

        # Save after each epoch
        with open(opts.outFile, 'w') as fid:
            pickle.dump(opts, fid)
            pickle.dump(SGD.costt, fid)
            nn.toFile(fid)

        SGD.alpha = SGD.alpha / opts.anneal


def test(opts):
    print "Testing model %s" % opts.inFile

    with open(opts.inFile, 'r') as fid:
        old_opts = pickle.load(fid)
        pickle.load(fid)
        loader = dl.DataLoader(
            opts.dataDir, old_opts.rawDim, old_opts.inputDim)
        nn = rnnet.NNet(old_opts.inputDim, old_opts.outputDim, old_opts.layerSize, old_opts.numLayers,
                        opts.maxUttLen, temporalLayer=old_opts.temporalLayer, train=False)
        nn.initParams()
        nn.fromFile(fid)

    # FIXME Different output directory specific to test set
    out_dir = pjoin(SCAIL_DATA_DIR, 'ctc_loglikes_%s' % DATASET)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i in range(1, opts.numFiles + 1):
        writeLogLikes(loader, nn, i, out_dir, writePickle=True)


if __name__ == '__main__':
    run()
