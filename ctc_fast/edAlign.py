import cPickle as pickle
import numpy as np
from editDist import edit_distance as ed, ref_to_hyp
from progressbar import ProgressBar

'''
Compute and save alignments between argmax result and transcripts
for training LM on transcripts + CTC probs
'''

def main(args):

    ref2source = list()

    # NOTE Make sure synced with order dumped in runDecode.py
    with open(args.infile, 'rb') as fid:
        hyps = pickle.load(fid)
        print 'Loaded hyps'
        refs = pickle.load(fid)
        print 'Loaded refs'
        pickle.load(fid)  # hypscores
        pickle.load(fid)  # refscores
        pickle.load(fid)  # numphones
        pickle.load(fid)  # subsets
        alignments = pickle.load(fid)
        print 'Loaded alignments'
    print 'Loaded data'

    pbar = ProgressBar(maxval=len(hyps)).start()
    j = 0

    for (hyp, ref, align) in zip(hyps, refs, alignments):
        #print ref, len(hyp), len(ref), len(align)
        dist, eq, ins, dels, subs, errs_by_pos, hyp_corr, ref_corr = ed(hyp, ref)
        r2h = ref_to_hyp(hyp_corr, ref_corr)
        r2s = list()
        #print 'len align:', len(align)
        #print 'len hyp:', len(hyp)
        #print 'len ref:', len(ref)
        #print hyp
        #print ref
        #print r2h
        for k in xrange(len(r2h)):
            if len(align) == 0:  # empty hyp
                r2s.append(0)
                continue
            ind = r2h[k]
            if ind == len(align):
                ind -= 1  # edge case
            r2s.append(align[ind])
        ref2source.append(r2s)

        j += 1
        pbar.update(j)

    print '%d alignments computed' % len(ref2source)
    with open(args.outfile, 'wb') as fid:
        pickle.dump(np.array(ref2source), fid)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', default='hyp.pk', help='Pickle file with data')
    parser.add_argument('outfile', default='align.pk', help='Pickle file with data')
    parser.add_argument('--display', action='store_true')
    args = parser.parse_args()
    main(args)
