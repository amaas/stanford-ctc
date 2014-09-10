import os
import numpy as np
from os.path import join as pjoin
import editDistance as ed
import dataLoader as dl
import cPickle as pickle
from collections import defaultdict
from joblib import Parallel, delayed
from decoder.decoder_config import SPACE, SCAIL_DATA_DIR,\
        INPUT_DIM, RAW_DIM, DATASET
from cluster.config import NUM_CPUS, CLUSTER_DIR, PYTHON_CMD
from cluster.utils import get_free_nodes, run_cpu_job
from decoder.decoder_utils import decode

'''
Take probs outputted by runNNet.py forward props and run
decoding as well as compute error metrics
'''

def get_char_map(dataDir):
    kaldi_base = '/'.join(dataDir.split('/')[:-3]) + '/'
    with open(kaldi_base + 'ctc-utils/chars.txt', 'r') as fid:
        labels = [l.strip().split() for l in fid.readlines()]
        labels = dict((int(k), v) for v, k in labels)
    return labels


def decode_utterance(k, probs, labels, phone_map):
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
            alpha=0.0, beta=0.0, beam=40, method='clm2')

    return (hyp, ref, hypscore, truescore)


def runSeq(opts):
    totdist = numphones = 0
    lengthsH = []
    lengthsR = []
    scoresH = []
    scoresR = []
    fid = open(opts.out_file, 'w')

    phone_map = get_char_map(opts.dataDir)

    loader = dl.DataLoader(opts.dataDir, opts.rawDim, opts.inputDim)
    likelihoodsDir = pjoin(SCAIL_DATA_DIR, 'ctc_loglikes_%s' % DATASET)

    for i in range(opts.start_file, opts.start_file + opts.numFiles):
        data_dict, alis, keys, sizes = loader.loadDataFileDict(i)

        ll_file = pjoin(likelihoodsDir, 'loglikelihoods_%d.pk' % i)
        ll_fid = open(ll_file, 'rb')
        probs_dict = pickle.load(ll_fid)

        # Parallelize decoding over utterances

        print 'Decoding utterances in parallel, n_jobs=%d' % NUM_CPUS
        decoded_utts = Parallel(n_jobs=NUM_CPUS)(delayed(decode_utterance)(k, probs_dict[k], alis[k], phone_map) for k in keys)

        # Log stats
        # FIXME This should really be done as post-processing

        for k, (hyp, ref, hypscore, truescore) in zip(keys, decoded_utts):
            if truescore is None:
                truescore = 0.0
            if hypscore is None:
                hypscore = 0.0

            hyp = [phone_map[h] for h in hyp]
            lengthsH.append(float(len(hyp)))
            lengthsR.append(float(len(ref)))
            scoresH.append(hypscore)
            scoresR.append(truescore)
            print "Ref score %f" % (truescore)
            dist, ins, dels, subs, corr = ed.edit_distance(ref, hyp)
            print "Distance %d/%d, HYP Score %f, Ref Score %f" % (dist, len(ref), hypscore, truescore)
            fid.write(k + ' ' + ' '.join(hyp) + '\n')
            totdist += dist
            numphones += len(alis[k])

    print "Avg ref score %f" % (sum(scoresR) / len(scoresR))
    print "Avg hyp score %f, Avg ref score %f" % (sum(scoresH) / len(scoresH), sum(scoresR) / len(scoresR))
    fid.close()
    #with open("scores.bin", 'w') as fid2:
        #pickle.dump(scoresH, fid2)
        #pickle.dump(scoresR, fid2)
    print "Average Lengths HYP: %f REF: %f" % (np.mean(lengthsH), np.mean(lengthsR))
    print "CER : %f" % (100 * totdist / float(numphones))


def runNode(node, node_files_dict):
    # Create decoding command for each file
    cmds = list()
    fns = node_files_dict[node]
    if len(fns) == 0:
        return None
    for fn in fns:
        cmd = '%s ../runDecode.py --dataDir %s --numFiles 1 --start_file %d --out_file %s.%d' % (PYTHON_CMD, opts.dataDir, fn, opts.out_file, fn)
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
    free_nodes = get_free_nodes('gorgon')  # PARAM
    node_files_dict = defaultdict(list)
    for i in xrange(opts.start_file, opts.start_file + opts.numFiles):
        node_files_dict[free_nodes[i % len(free_nodes)]].append(i)

    # Now for each node run decoding of the files assigned to it
    Parallel(n_jobs=len(free_nodes))(delayed(runNode)(node, node_files_dict) for node in node_files_dict)

    # Now merge the results together into single file like
    # we would get by running sequentially
    concat_list = list()
    for i in xrange(opts.start_file, opts.start_file + opts.numFiles):
        fi = '%s.%d' % (opts.out_file, i)
        assert os.path.exists(fi), '%s does not exist' % fi
        with open(fi, 'r') as f:
            concat_list.append(f.read().strip())
        # Cleanup
        os.remove(fi)
    with open(opts.out_file, 'w') as fout:
        fout.write('\n'.join(concat_list))


if __name__ == '__main__':
    import optparse
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    # Data
    parser.add_option("--dataDir", dest="dataDir", type="string",
                      default="/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/swbd/s5b/exp/train_ctc/")
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
