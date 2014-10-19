import os
import numpy as np
import cPickle as pickle
import editDistance as ed

'''
Post-process outputs of decoding and compute stats like CER,
avg ref and hyp lengths, etc.
'''

def main(args):
    files = list()

    hyps = list()
    refs = list()
    hypscores = list()
    refscores = list()
    numphones = list()

    if args.num_files == 1:
        files.append(args.pk_file)
    else:
        for k in xrange(1, args.num_files + 1):
            files.append('%s.%d' % (args.pk_file, k))
    for f in files:
        # NOTE Make sure synced with order dumped in runDecode.py
        fid = open(f, 'rb')
        hyps += pickle.load(fid)
        refs += pickle.load(fid)
        hypscores += pickle.load(fid)
        refscores += pickle.load(fid)
        numphones += pickle.load(fid)
        fid.close()

        # Cleanup
        if args.num_files != 1:
            os.remove(f)

    if args.num_files != 1:
        # Save single file with all the data
        fid = open(args.pk_file, 'wb')
        pickle.dump(hyps, fid)
        pickle.dump(refs, fid)
        pickle.dump(hypscores, fid)
        pickle.dump(refscores, fid)
        pickle.dump(numphones, fid)

    hyp_lens = [len(s) for s in hyps]
    ref_lens = [len(s) for s in refs]

    totdist = 0.0
    num_sents_correct = 0
    correct_sents_len = 0
    for (hyp, ref, hypscore, refscore) in zip(hyps, refs, hypscores, refscores):
        dist, ins, dels, subs, corr = ed.edit_distance(ref, hyp)
        print 'Distance %d/%d, hyp score %f, ref score %f' % (dist, len(ref), hypscore, refscore)
        if dist == 0:
            num_sents_correct += 1
            correct_sents_len += len(ref)
        totdist += dist

    print 'Avg ref score %f' % (sum(refscores) / len(refscores))
    print 'Avg hyp score %f' % (sum(hypscores) / len(hypscores))

    print 'Average len hyp: %f' % np.mean(hyp_lens)
    print 'Average len ref: %f' % np.mean(ref_lens)

    print 'CER: %f' % (100.0 * totdist / np.sum(numphones))
    print '%d/%d sentences correct' % (num_sents_correct, len(hyps))
    print 'Average length of correct sentence: %f' % (correct_sents_len / float(num_sents_correct))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pk_file', default='hyp.pk', help='Pickle file with data')
    parser.add_argument('--num_files', type=int, default=1,
        help='If > 1, will find all files matching pk_file.#, merge them, and compute combined stats')
    args = parser.parse_args()
    main(args)
