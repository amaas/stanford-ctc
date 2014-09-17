import numpy as np
import cPickle as pickle
import optparse
import dataLoader as dl
from joblib import Parallel, delayed
from os.path import join as pjoin
import new_decoder.decoder as decoder
from cluster.config import NUM_CPUS, CLUSTER_DIR, PYTHON_CMD

def decode_utterance_clm(k, probs, labels, charmap_file, lm_file):
    # setup decoder
    dec_lm = decoder.BeamLMDecoder()
    dec_lm.load_chars(charmap_file)
    dec_lm.load_lm(lm_file)

    hyp, hypscore = dec_lm.decode(probs.astype(np.double))

    # return (hyp, ref, hypscore, truescore)
    return hyp, None, hypscore, None



def runSeq(opts):
    #fid = open(opts.out_file, 'w')
    # phone_map = get_char_map(opts.dataDir)

    # initialize loader to not read actual data
    loader = dl.DataLoader(opts.ali_dir, -1, -1,load_ali=True,load_data=False)
    #likelihoodsDir = pjoin(SCAIL_DATA_DIR, 'ctc_loglikes_%s' % DATASET)

    hyps = list()
    refs = list()
    hypscores = list()
    refscores = list()
    numphones = list()

    for i in range(opts.start_file, opts.start_file + opts.num_files):
        data_dict, alis, keys, sizes = loader.loadDataFileDict(i)

        ll_file = pjoin(opts.lik_dir, 'loglikelihoods_%d.pk' % i)
        with open(ll_file, 'rb') as ll_fid:
            probs_dict = pickle.load(ll_fid)

        # Parallelize decoding over utterances

        print 'Decoding utterances in parallel, n_jobs=%d' % NUM_CPUS
        decoded_utts = Parallel(n_jobs=NUM_CPUS)(delayed(decode_utterance_clm)(k, probs_dict[k], alis[k], opts.charmap_file, opts.lm_file) for k in keys)

        for k, (hyp, ref, hypscore, refscore) in zip(keys, decoded_utts):
            if refscore is None:
                refscore = 0.0
            if hypscore is None:
                hypscore = 0.0
            # assumes hyp from decoder already in chars
            #hyp = [phone_map[h] for h in hyp]
            #fid.write(k + ' ' + ' '.join(hyp) + '\n')
            print k + ' ' + ' '.join(hyp) 
            hyps.append(hyp)
            refs.append(ref)
            hypscores.append(hypscore)
            refscores.append(refscore)
            numphones.append(len(alis[k]))

    #fid.close()

    # Pickle some values for computeStats.py
    with open(opts.out_file.replace('.txt', '.pk'), 'wb') as pkid:
        pickle.dump(hyps, pkid)
        pickle.dump(refs, pkid)
        pickle.dump(hypscores, pkid)
        pickle.dump(refscores, pkid)
        pickle.dump(numphones, pkid)



if __name__ == '__main__':

    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    # Data
    parser.add_option("--likDir", dest="lik_dir", type="string",
                      default="/scail/scratch/group/deeplearning/speech/amaas/kaldi-stanford/stanford-nnet/ctc_fast/swbd_eval2000_lik/")
    parser.add_option("--aliDir", dest="ali_dir", type="string",
                      default="/scail/scratch/group/deeplearning/speech/amaas/kaldi-stanford/stanford-nnet/ctc_fast/swbd_eval2000_lik/")
    parser.add_option("--charmapFile", dest="charmap_file", type="string",
                      default="/scail/scratch/group/deeplearning/speech/amaas/kaldi-stanford/stanford-nnet/ctc_fast/swbd_eval2000_lik/chars.txt")
    parser.add_option("--lmFile", dest="lm_file", type="string",
                      default="/scail/group/deeplearning/speech/amaas/kaldi-stanford/kaldi-trunk/egs/wsj/s6/data/local/lm/text_char.2g.arpa")

    parser.add_option("--numFiles", dest="num_files", type="int", default=23)
    parser.add_option('--start_file', dest='start_file', type='int', default=1)
    parser.add_option('--out_file', dest='out_file', type='string', default='hyp.txt')
    parser.add_option('--parallel', dest='parallel', action='store_true', default=False, help='Decode files across multiple machines')

    (opts, args) = parser.parse_args()

    runSeq(opts)
    # if opts.parallel:
    #     runParallel(opts)
    # else:
    #     runSeq(opts)

