import os
import re
import time
import numpy as np
from os.path import join as pjoin
import dataLoader as dl
import cPickle as pickle
from joblib import Parallel, delayed
from decoder.decoder_config import SPACE, SCAIL_DATA_DIR,\
        INPUT_DIM, RAW_DIM, DATASET, DATA_SUBSET, SPECIALS_LIST
from cluster.config import NUM_CPUS, CLUSTER_DIR, PYTHON_CMD, SLEEP_SEC
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

# NOTE Just for SWBD
PATTERN = re.compile('[a-z\-\'\&\/ ]+', re.UNICODE)

MODEL_TYPE = 'rnn'

LIKELIHOODS_DIR = pjoin(SCAIL_DATA_DIR, 'ctc_loglikes_%s_%s' % (DATASET, DATA_SUBSET))

def decode_utterance(k, probs, labels, phone_map, lm=None):
    labels = np.array(labels, dtype=np.int32)
    probs = probs.astype(np.float64)

    alpha = 1.5
    beta = 1.5
    hyp0, hypscore, truescore, align = decode(probs,
            alpha=alpha, beta=beta, beam=150, method='clm2', clm=lm)

    # Filter away special symbols and strip away starting
    # and ending spaces

    ref = []
    for a in labels:
        token = phone_map[a]
        # Filter away special symbols, numbers, and extra spaces
        if (token == SPACE and len(ref) > 0 and ref[-1] != SPACE)\
                or (token not in SPECIALS_LIST and PATTERN.match(token)):
            ref.append(token)
    while len(ref) and ref[-1] == SPACE:
        ref = ref[:-1]
    while len(ref) and ref[0] == SPACE:
        ref = ref[1:]

    hyp = []
    for a in hyp0:
        token = phone_map[a]
        # Filter away special symbols, numbers, and extra spaces
        if (token == SPACE and len(hyp) > 0 and hyp[-1] != SPACE)\
                or (token not in SPECIALS_LIST and PATTERN.match(token)):
            hyp.append(token)
    while len(hyp) and hyp[-1] == SPACE:
        hyp = hyp[:-1]
    while len(hyp) and hyp[0] == SPACE:
        hyp = hyp[1:]

    return (hyp, ref, hypscore, truescore, align)


def runSeq(opts):
    fid = open(opts.out_file, 'w')
    phone_map = get_char_map(opts.dataDir)
    print phone_map
    print len(phone_map)

    alisDir = opts.alisDir if opts.alisDir else opts.dataDir
    loader = dl.DataLoader(opts.dataDir, opts.rawDim, opts.inputDim, alisDir)

    hyps = list()
    refs = list()
    hypscores = list()
    refscores = list()
    numphones = list()
    subsets = list()
    alignments = list()

    if MODEL_TYPE != 'ngram':
        cfg_file = '/deep/u/zxie/rnnlm/13/cfg.json'
        params_file = '/deep/u/zxie/rnnlm/13/params.pk'
        #cfg_file = '/deep/u/zxie/dnn/11/cfg.json'
        #params_file = '/deep/u/zxie/dnn/11/params.pk'

        cfg = load_config(cfg_file)
        model_class, model_hps = get_model_class_and_params(MODEL_TYPE)
        opt_hps = OptimizerHyperparams()
        model_hps.set_from_dict(cfg)
        opt_hps.set_from_dict(cfg)

        clm = model_class(None, model_hps, opt_hps, train=False, opt='nag')
        with open(params_file, 'rb') as fin:
            clm.from_file(fin)
    else:
        from srilm import LM
        from decoder_config import LM_ARPA_FILE
        print 'Loading %s...' % LM_ARPA_FILE
        clm = LM(LM_ARPA_FILE)
        print 'Done.'
    #clm = None

    for i in range(opts.start_file, opts.start_file + opts.numFiles):
        data_dict, alis, keys, _ = loader.loadDataFileDict(i)
        # For later alignments
        keys = sorted(keys)

        # For Switchboard filter
        if DATA_SUBSET == 'eval2000':
            if SWBD_SUBSET == 'swbd':
                keys = [k for k in keys if k.startswith('sw')]
            elif SWBD_SUBSET == 'callhome':
                keys = [k for k in keys if k.startswith('en')]

        ll_file = pjoin(LIKELIHOODS_DIR, 'loglikelihoods_%d.pk' % i)
        ll_fid = open(ll_file, 'rb')
        probs_dict = pickle.load(ll_fid)

        # Parallelize decoding over utterances
        print 'Decoding utterances in parallel, n_jobs=%d, file=%d' % (NUM_CPUS, i)
        decoded_utts = Parallel(n_jobs=NUM_CPUS)(delayed(decode_utterance)(k, probs_dict[k], alis[k], phone_map, lm=clm) for k in keys)

        for k, (hyp, ref, hypscore, refscore, align) in zip(keys, decoded_utts):
            if refscore is None:
                refscore = 0.0
            if hypscore is None:
                hypscore = 0.0
            hyp = replace_contractions(hyp)
            fid.write(k + ' ' + ' '.join(hyp) + '\n')

            hyps.append(hyp)
            refs.append(ref)
            hypscores.append(hypscore)
            refscores.append(refscore)
            numphones.append(len(alis[k]))
            subsets.append('callhm' if k.startswith('en') else 'swbd')
            alignments.append(align)

    fid.close()

    # Pickle some values for computeStats.py
    pkid = open(opts.out_file.replace('.txt', '.pk'), 'wb')
    pickle.dump(hyps, pkid)
    pickle.dump(refs, pkid)
    pickle.dump(hypscores, pkid)
    pickle.dump(refscores, pkid)
    pickle.dump(numphones, pkid)
    pickle.dump(subsets, pkid)
    pickle.dump(alignments, pkid)
    pkid.close()


def runNode(node, job, opts):
    alisDir = opts.alisDir if opts.alisDir else opts.dataDir

    # Create decoding command for each file
    cmd = '%s ../runDecode.py --dataDir %s --alisDir %s --numFiles 1 --start_file %d --out_file %s.%d' % (PYTHON_CMD, opts.dataDir, alisDir, job, opts.out_file, job)
    print cmd

    full_cmd = 'cd %s/../%s-utils; source ~/.bashrc; ' % (CLUSTER_DIR, DATASET)
    full_cmd += '; ' + cmd
    print full_cmd
    log_file = '/tmp/%s_decode%s.log' % (DATASET, job)
    run_cpu_job(node, full_cmd, stdout=open(log_file, 'w'), blocking=False)
    return None


def runParallel(opts):
    # FIXME Currently assumes you run this from ./${DATASET}-utils

    # First distribute all the jobs
    unsub_jobs = list()
    for i in xrange(opts.start_file, opts.start_file + opts.numFiles):
        unsub_jobs.append(i)

    # Sort files by size so that biggest files are running the whole time
    # Note that we pop from the end
    unsub_jobs = sorted(unsub_jobs, key=lambda x: os.path.getsize(pjoin(LIKELIHOODS_DIR, 'loglikelihoods_%d.pk' % x)))

    while len(unsub_jobs):
        # Get free machines and split the files between them
        free_nodes = get_free_nodes('deep')  # PARAM
        for free_node in free_nodes:
            if len(unsub_jobs):
                job = unsub_jobs.pop()
                runNode(free_node, job, opts)
        if len(unsub_jobs):
            print '-' * 80
            print '%d jobs to be submitted...' % len(unsub_jobs)
            print '-' * 80
            time.sleep(SLEEP_SEC)

    # Now wait until all the jobs are finished
    # FIXME Hacky, just checks whether files exist and aren't empty
    def jobs_left():
        wait_count = 0
        for i in xrange(opts.start_file, opts.start_file + opts.numFiles):
            fi = '%s.%d' % (opts.out_file, i)
            if not os.path.exists(fi) or os.path.getsize(fi) == 0:
                wait_count += 1
        return wait_count
    while jobs_left() != 0:
        print '-' * 80
        print 'Waiting for %d jobs to finish...' % jobs_left()
        print '-' * 80
        time.sleep(SLEEP_SEC)

    # Now merge the results together into single file like
    # we would get by running sequentially
    concat_list = list()
    hyps = list()
    refs = list()
    hypscores = list()
    refscores = list()
    numphones = list()
    subsets = list()
    alignments = list()
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
            al = pickle.load(f)
            hyps += h
            refs += r
            hypscores += hs
            refscores += rs
            numphones += np
            subsets += ss
            alignments += al
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
        pickle.dump(alignments, fout)


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
