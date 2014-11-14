import os
import numpy as np
from os.path import join as pjoin
import dataLoader as dl
import cPickle as pickle
from collections import defaultdict
from joblib import Parallel, delayed
from decoder.decoder_config import SPACE, SCAIL_DATA_DIR,\
        INPUT_DIM, RAW_DIM, DATASET, DATA_SUBSET
from cluster.config import NUM_CPUS, CLUSTER_DIR, PYTHON_CMD
from cluster.utils import get_free_nodes, run_cpu_job
from decoder.decoder_utils import decode
if DATA_SUBSET == 'eval2000':
    from decoder.decoder_config import SWBD_SUBSET
from model_utils import get_model_class_and_params
from run_utils import load_config
from optimizer import OptimizerHyperparams
from errorAnalysis import replace_contractions

'''
Take probs outputted by runNNet.py forward props and run decoding
Afterwards use computeStats.py to get error metrics
'''
# TODO Should branch off of nnet experiment directories

def get_char_map(dataDir):
    kaldi_base = '/'.join(dataDir.split('/')[:-3]) + '/'
    with open(kaldi_base + 'ctc-utils/chars.txt', 'r') as fid:
        labels = [l.strip().split() for l in fid.readlines()]
        labels = dict((int(k), v) for v, k in labels)
    return labels


def decode_utterance(k, probs, labels, phone_map, lm=None):
    labels = np.array(labels, dtype=np.int32)
    # Build sentence for lm
    sentence = []
    ref = []
    word = ""
    for a in labels:
        token = phone_map[a]
        ref.append(token)
        if token != SPACE:
            word += token
        else:
            sentence.append(word)
            word = ""
    #ref = [phone_map[int(r)] for r in labels]

    probs = probs.astype(np.float64)

    hyp, hypscore, truescore = decode(probs,
            alpha=3.0, beta=1.5, beam=200, method='clm2', clm=lm)

    return (hyp, ref, hypscore, truescore)


def runSeq(opts):
    fid = open(opts.out_file, 'w')
    phone_map = get_char_map(opts.dataDir)

    alisDir = opts.alisDir if opts.alisDir else opts.dataDir
    loader = dl.DataLoader(opts.dataDir, opts.rawDim, opts.inputDim, alisDir)
    likelihoodsDir = pjoin(SCAIL_DATA_DIR, 'ctc_loglikes_%s_%s' % (DATASET, DATA_SUBSET))

    hyps = list()
    refs = list()
    hypscores = list()
    refscores = list()
    numphones = list()
    subsets = list()

    cfg_file = '/deep/u/zxie/dnn/3/cfg.json'
    cfg = load_config(cfg_file)
    model_class, model_hps = get_model_class_and_params('dnn')
    opt_hps = OptimizerHyperparams()
    model_hps.set_from_dict(cfg)
    opt_hps.set_from_dict(cfg)

    clm = model_class(None, model_hps, opt_hps, train=False, opt='nag')
    with open('/deep/u/zxie/dnn/3/params.pk', 'rb') as fin:
        clm.from_file(fin)
    #clm = None

    for i in range(opts.start_file, opts.start_file + opts.numFiles):
        data_dict, alis, keys, sizes = loader.loadDataFileDict(i)

        # For Switchboard filter
        if DATA_SUBSET == 'eval2000':
            if SWBD_SUBSET == 'swbd':
                keys = [k for k in keys if k.startswith('sw')]
            elif SWBD_SUBSET == 'callhome':
                keys = [k for k in keys if k.startswith('en')]

        ll_file = pjoin(likelihoodsDir, 'loglikelihoods_%d.pk' % i)
        ll_fid = open(ll_file, 'rb')
        probs_dict = pickle.load(ll_fid)

        # Parallelize decoding over utterances
        print 'Decoding utterances in parallel, n_jobs=%d, file=%d' % (NUM_CPUS, i)
        decoded_utts = Parallel(n_jobs=NUM_CPUS)(delayed(decode_utterance)(k, probs_dict[k], alis[k], phone_map, lm=clm) for k in keys)

        for k, (hyp, ref, hypscore, refscore) in zip(keys, decoded_utts):
            if refscore is None:
                refscore = 0.0
            if hypscore is None:
                hypscore = 0.0
            hyp = [phone_map[h] for h in hyp]
            hyp = replace_contractions(hyp)
            fid.write(k + ' ' + ' '.join(hyp) + '\n')

            hyps.append(hyp)
            refs.append(ref)
            hypscores.append(hypscore)
            refscores.append(refscore)
            numphones.append(len(alis[k]))
            subsets.append('callhm' if k.startswith('en') else 'swbd')

    fid.close()

    # Pickle some values for computeStats.py
    pkid = open(opts.out_file.replace('.txt', '.pk'), 'wb')
    pickle.dump(hyps, pkid)
    pickle.dump(refs, pkid)
    pickle.dump(hypscores, pkid)
    pickle.dump(refscores, pkid)
    pickle.dump(numphones, pkid)
    pickle.dump(subsets, pkid)
    pkid.close()


def runNode(node, node_files_dict, opts):
    alisDir = opts.alisDir if opts.alisDir else opts.dataDir

    # Create decoding command for each file
    cmds = list()
    fns = node_files_dict[node]
    if len(fns) == 0:
        return None
    for fn in fns:
        cmd = '%s ../runDecode.py --dataDir %s --alisDir %s --numFiles 1 --start_file %d --out_file %s.%d' % (PYTHON_CMD, opts.dataDir, alisDir, fn, opts.out_file, fn)
        print cmd
        cmds.append(cmd)

    # Join the commands together and run
    full_cmd = 'cd %s/../%s-utils; source ~/.bashrc; ' % (CLUSTER_DIR, DATASET)
    full_cmd += '; '.join(cmds)
    print full_cmd
    log_file = '/tmp/%s_decode%s.log' % (DATASET, ','.join([str(x) for x in node_files_dict[node]]))
    run_cpu_job(node, full_cmd, stdout=open(log_file, 'w'), blocking=True)
    return None


def runParallel(opts):
    # FIXME Currently assumes you run this from ./${DATASET}-utils
    # TODO Get the CER and other stats as a post-processing step

    # Get free machines and split the files evenly between them
    free_nodes = get_free_nodes('deep')  # PARAM
    node_files_dict = defaultdict(list)
    for i in xrange(opts.start_file, opts.start_file + opts.numFiles):
        node_files_dict[free_nodes[i % len(free_nodes)]].append(i)

    # Now for each node run decoding of the files assigned to it
    Parallel(n_jobs=len(free_nodes))(delayed(runNode)(node, node_files_dict, opts) for node in node_files_dict)

    # Now merge the results together into single file like
    # we would get by running sequentially
    concat_list = list()
    hyps = list()
    refs = list()
    hypscores = list()
    refscores = list()
    numphones = list()
    subsets = list()
    for i in xrange(opts.start_file, opts.start_file + opts.numFiles):
        fi = '%s.%d' % (opts.out_file, i)
        fi_pk = fi.replace('.txt', '.pk')
        assert os.path.exists(fi), '%s does not exist' % fi
        assert os.path.exists(fi_pk), '%s does not exist' % fi_pk
        with open(fi, 'r') as f:
            concat_list.append(f.read().strip())
        with open(fi_pk, 'rb') as f:
            h = pickle.load(f)
            r = pickle.load(f)
            hs = pickle.load(f)
            rs = pickle.load(f)
            np = pickle.load(f)
            ss = pickle.load(f)
            hyps += h
            refs += r
            hypscores += hs
            refscores += rs
            numphones += np
            subsets += ss
        # Cleanup
        os.remove(fi)
        os.remove(fi_pk)
    with open(opts.out_file, 'w') as fout:
        fout.write('\n'.join(concat_list))
    with open(opts.out_file.replace('.txt', '.pk'), 'wb') as fout:
        pickle.dump(hyps, fout)
        pickle.dump(refs, fout)
        pickle.dump(hypscores, fout)
        pickle.dump(refscores, fout)
        pickle.dump(numphones, fout)
        pickle.dump(subsets, fout)


if __name__ == '__main__':
    import optparse
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    # Data
    parser.add_option("--dataDir", dest="dataDir", type="string",
                      default="/deep/group/speech/speech/awni/kaldi-stanford/kaldi-trunk/egs/swbd/s5b/exp/train_ctc/")
    parser.add_option('--alisDir', dest='alisDir', type='string', default='')
    parser.add_option("--numFiles", dest="numFiles", type="int", default=384)
    parser.add_option(
        "--inputDim", dest="inputDim", type="int", default=INPUT_DIM)
    parser.add_option("--rawDim", dest="rawDim", type="int", default=RAW_DIM)
    parser.add_option('--out_file', dest='out_file', type='string', default='hyp.txt')
    parser.add_option('--start_file', dest='start_file', type='int', default=1)
    parser.add_option('--parallel', action='store_true', default=False, help='Decode files across multiple machines')

    (opts, args) = parser.parse_args()

    if opts.parallel:
        runParallel(opts)
    else:
        runSeq(opts)
