import os
import time
import logging
import optparse
from os.path import join as pjoin
import numpy as np
import cPickle as pickle
import cudamat as cm

from writeLikelihoods import writeLogLikes

import sgd
import nnets.brnnet as rnnet
import dataLoader as dl

from decoder.decoder_config import SCAIL_DATA_DIR, DATASET, DATA_SUBSET, MAX_UTT_LEN

from run_utils import dump_config, load_config, CfgStruct, get_git_revision,\
        get_hostname, TimeString, touch_file
from run_cfg import RUN_DIR, TRAIN_DATA_DIR, TRAIN_ALIS_DIR


def run(args=None):
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option('--cfg_file', dest='cfg_file', default=None,
            help='File with settings from previously trained net')

    parser.add_option(
        "--test", action="store_true", dest="test", default=False)

    # Architecture
    parser.add_option(
        "--layerSize", dest="layerSize", type="int", default=1824)
    parser.add_option("--numLayers", dest="numLayers", type="int", default=5)
    parser.add_option(
        "--temporalLayer", dest="temporalLayer", type="int", default=3)

    # Optimization
    parser.add_option("--momentum", dest="momentum", type="float",
                      default=0.95)
    parser.add_option("--epochs", dest="epochs", type="int", default=20)
    parser.add_option("--step", dest="step", type="float", default=1e-5)
    parser.add_option("--anneal", dest="anneal", type="float", default=1.3,
                      help="Sets (learning rate := learning rate / anneal) after each epoch.")

    # Data
    parser.add_option("--dataDir", dest="dataDir", type="string",
                      default=TRAIN_DATA_DIR['fbank'])
    parser.add_option('--alisDir', dest='alisDir', type='string', default=TRAIN_ALIS_DIR)
    parser.add_option("--numFiles", dest="numFiles", type="int", default=384)
    parser.add_option(
        "--inputDim", dest="inputDim", type="int", default=41 * 15)
    parser.add_option("--rawDim", dest="rawDim", type="int", default=41 * 15)
    parser.add_option("--outputDim", dest="outputDim", type="int", default=35)
    parser.add_option(
        "--maxUttLen", dest="maxUttLen", type="int", default=MAX_UTT_LEN)

    # Save/Load
    parser.add_option('--save_every', dest='save_every', type='int',
            default=10, help='During training, save parameters every x number of files')

    parser.add_option('--run_desc', dest='run_desc', type='string', default='', help='Description of experiment run')

    (opts, args) = parser.parse_args(args)

    if opts.cfg_file:
        cfg = load_config(opts.cfg_file)
    else:
        cfg = vars(opts)

    # These config values should be updated every time
    cfg['host'] = get_hostname()
    cfg['git_rev'] = get_git_revision()
    cfg['pid'] = os.getpid()

    # Create experiment output directory

    if not opts.cfg_file:
        time_string = str(TimeString())
        output_dir = pjoin(RUN_DIR, time_string)
        cfg['output_dir'] = output_dir
        if not os.path.exists(output_dir):
            print 'Creating %s' % output_dir
            os.makedirs(output_dir)
        opts.cfg_file = pjoin(output_dir, 'cfg.json')
    else:
        output_dir = cfg['output_dir']

    cfg['in_file'] = pjoin(output_dir, 'params.pk')
    cfg['out_file'] = pjoin(output_dir, 'params.pk')

    # Logging

    logging.basicConfig(filename=pjoin(output_dir, 'train.log'), level=logging.DEBUG)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.info('Running on %s' % cfg['host'])

    # seed for debugging, turn off when stable
    np.random.seed(33)
    import random
    random.seed(33)

    if 'CUDA_DEVICE' in os.environ:
        cm.cuda_set_device(int(os.environ['CUDA_DEVICE']))
    else:
        cm.cuda_set_device(0)  # Default

    # Testing
    if opts.test:
        test(opts)
        return

    opts = CfgStruct(**cfg)

    alisDir = opts.alisDir if opts.alisDir else opts.dataDir
    loader = dl.DataLoader(opts.dataDir, opts.rawDim, opts.inputDim, alisDir)

    nn = rnnet.NNet(opts.inputDim, opts.outputDim, opts.layerSize, opts.numLayers,
                    opts.maxUttLen, temporalLayer=opts.temporalLayer)
    nn.initParams()

    SGD = sgd.SGD(nn, opts.maxUttLen, alpha=opts.step, momentum=opts.momentum)

    # Dump config
    cfg['param_count'] = nn.paramCount()
    dump_config(cfg, opts.cfg_file)

    # Load model if specified
    if os.path.exists(opts.in_file):
        with open(opts.in_file, 'r') as fid:
            SGD.fromFile(fid)
            nn.fromFile(fid)

    # Training
    epoch_file = pjoin(output_dir, 'epoch')
    if os.path.exists(epoch_file):
        start_epoch = int(open(epoch_file, 'r').read()) + 1
    else:
        start_epoch = 0

    num_files_file = pjoin(output_dir, 'num_files')

    for k in range(start_epoch, opts.epochs):
        perm = np.random.permutation(opts.numFiles) + 1
        loader.loadDataFileAsynch(perm[0])

        file_start = 0
        if k == start_epoch:
            if os.path.exists(num_files_file):
                file_start = int(open(num_files_file, 'r').read().strip())
                logger.info('Starting from file %d' % file_start)

        for i in xrange(file_start, perm.shape[0]):
            start = time.time()
            data_dict, alis, keys, sizes = loader.getDataAsynch()
            # Prefetch
            if i + 1 < perm.shape[0]:
                loader.loadDataFileAsynch(perm[i + 1])
            SGD.run(data_dict, alis, keys, sizes)
            end = time.time()
            logger.info('File time %f' % (end - start))

            # Save parameters
            if (i+1) % opts.save_every == 0:
                logger.info('Saving parameters')
                with open(opts.out_file, 'w') as fid:
                    SGD.toFile(fid)
                    nn.toFile(fid)
                    open(num_files_file, 'w').write('%d' % (i+1))
                logger.info('Done saving parameters')

        # Save epoch completed
        open(pjoin(output_dir, 'epoch'), 'w').write(k)

        SGD.alpha = SGD.alpha / opts.anneal

    # Run now complete, touch sentinel file
    touch_file(pjoin(output_dir, 'sentinel'))


def test(opts):
    old_opts = CfgStruct(**load_config(opts.cfg_file))

    logging.basicConfig(filename=pjoin(opts.output_dir, 'test.log'), level=logging.DEBUG)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.info('Running on %s' % get_hostname())

    with open(old_opts.in_file, 'r') as fid:
        pickle.load(fid)  # SGD data
        print 'rawDim:', old_opts.rawDim, 'inputDim:', old_opts.inputDim,\
            'layerSize:', old_opts.layerSize, 'numLayers:', old_opts.numLayers,\
            'maxUttLen:', old_opts.maxUttLen
        print 'temporalLayer:', old_opts.temporalLayer, 'outputDim:', old_opts.outputDim

        alisDir = opts.alisDir if opts.alisDir else opts.dataDir
        loader = dl.DataLoader(
            opts.dataDir, old_opts.rawDim, old_opts.inputDim, alisDir)
        nn = rnnet.NNet(old_opts.inputDim, old_opts.outputDim,
                old_opts.layerSize, old_opts.numLayers, old_opts.maxUttLen,
                temporalLayer=old_opts.temporalLayer, train=False)
        nn.initParams()
        nn.fromFile(fid)

    # FIXME Different output directory specific to test set
    out_dir = pjoin(SCAIL_DATA_DIR, 'ctc_loglikes_%s_%s' % (DATASET, DATA_SUBSET))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i in range(1, opts.numFiles + 1):
        writeLogLikes(loader, nn, i, out_dir, writePickle=True)

    # TODO Hook up with runDecode to do CER and WER computation
    # TODO Keep around hyp and ref files


if __name__ == '__main__':
    run()
