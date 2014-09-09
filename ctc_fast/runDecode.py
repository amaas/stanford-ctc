import numpy as np
from os.path import join as pjoin
import editDistance as ed
import dataLoader as dl
import cPickle as pickle

from joblib import Parallel, delayed
from decoder_config import SPACE, NUM_CPUS, SCAIL_DATA_DIR,\
        INPUT_DIM, RAW_DIM
from decoder_utils import decode

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


def run(opts):
    totdist = numphones = 0
    lengthsH = []
    lengthsR = []
    scoresH = []
    scoresR = []
    fid = open('hyp.txt', 'w')

    phone_map = get_char_map(opts.dataDir)

    loader = dl.DataLoader(opts.dataDir, opts.rawDim, opts.inputDim)
    likelihoodsDir = pjoin(SCAIL_DATA_DIR, 'ctc_loglikes')

    for i in range(1, opts.numFiles + 1):
        data_dict, alis, keys, sizes = loader.loadDataFileDict(i)


        ll_file = pjoin(likelihoodsDir, 'loglikelihoods_%d.pk' % i)
        ll_fid = open(ll_file, 'rb')
        probs_dict = pickle.load(ll_fid)

        # Parallelize decoding over utterances

        print 'Decoding utterances in parallel, n_jobs=%d' % NUM_CPUS
        decoded_utts = Parallel(n_jobs=NUM_CPUS)(delayed(decode_utterance)(k, probs_dict[k], alis[k], phone_map) for k in keys)

        # Log stats

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
    with open("scores.bin", 'w') as fid2:
        pickle.dump(scoresH, fid2)
        pickle.dump(scoresR, fid2)
    print "Average Lengths HYP: %f REF: %f" % (np.mean(lengthsH), np.mean(lengthsR))
    print "CER : %f" % (100 * totdist / float(numphones))


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

    (opts, args) = parser.parse_args()

    run(opts)
