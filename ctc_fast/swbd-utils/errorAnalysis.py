'''
Error analysis
TODO Make generic/modular and move to nn
'''

import numpy as np
import cPickle as pickle
from editDist import edit_distance as ed
#from progressbar import ProgressBar
from colorama import Fore, Back


def disp_corr(hyp, ref):
    '''
    Display correspondences between hyp and ref
    '''
    pass


def disp_errs_by_pos(err_by_pos, out_file):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot(err_by_pos)
    #plt.show()
    plt.savefig(out_file)


def disp_err_corr(hyp_corr, ref_corr):
    hyp_str = ''
    ref_str = ''
    assert len(hyp_corr) == len(ref_corr)
    for k in xrange(len(hyp_corr)):
        if hyp_corr[k] == '[space]':
            hc = ' '
        elif hyp_corr[k] == '<ins>':
            hc = Back.GREEN + ' ' + Back.RESET
        else:
            hc = hyp_corr[k]

        if ref_corr[k] == '[space]':
            rc = ' '
        elif ref_corr[k] == '<del>':
            rc = Back.RED + ' ' + Back.RESET
        else:
            rc = ref_corr[k]

        if hc != rc and len(hc) == 1 and len(rc) == 1:
            hc = Back.BLUE + Fore.BLACK + hc + Fore.RESET + Back.RESET
            rc = Back.BLUE + Fore.BLACK + rc + Fore.RESET + Back.RESET
        hyp_str += hc
        ref_str += rc
    print hyp_str
    print ref_str


def compute_and_display_stats(hyps, refs, hypscores, refscores, numphones, subsets, subset=None):
    # Filter by subset
    if subset:
        print 'USING SUBSET: %s' % subset
        filt = subsets == subset
        hyps = hyps[filt]
        refs = refs[filt]
        hypscores = hypscores[filt]
        refscores = refscores[filt]
        numphones = numphones[filt]

    '''
    Compute stats
    '''

    hyp_lens = [len(s) for s in hyps]
    ref_lens = [len(s) for s in refs]

    max_hyp_len = max(hyp_lens)
    tot_errs_by_pos = np.zeros(max_hyp_len)
    counts_by_pos = np.zeros(max_hyp_len, dtype=np.int32)

    tot_dist = tot_eq = tot_ins = tot_dels = tot_subs = 0.0
    num_sents_correct = 0
    correct_sents_len = 0
    frac_wrong = list()

    #pbar = ProgressBar(maxval=len(hyps)).start()

    k = 0
    for (hyp, ref, hypscore, refscore) in zip(hyps, refs, hypscores, refscores):
        dist, eq, ins, dels, subs, errs_by_pos, hyp_corr, ref_corr = ed(hyp, ref)
        tot_eq += eq
        tot_ins += ins
        tot_dels += dels
        tot_subs += subs
        tot_errs_by_pos[0:errs_by_pos.shape[0]] += errs_by_pos
        counts_by_pos[0:errs_by_pos.shape[0]] += 1
        k += 1
        #pbar.update(k)

        if dist == 0:
            num_sents_correct += 1
            correct_sents_len += len(ref)
        tot_dist += dist
        # NOTE Smoothing
        frac_wrong.append(dist / float(len(ref) + 1.0))

        disp_err_corr(hyp_corr, ref_corr)
        print

    '''
    Display aggregate stats
    '''

    print 'avg len hyp: %f' % np.mean(hyp_lens)
    print 'avg len ref: %f' % np.mean(ref_lens)
    print 'avg num phones: %f' % np.mean(numphones)

    print 'avg ref score: %f' % (sum(refscores) / len(refscores))
    print 'avg hyp score: %f' % (sum(hypscores) / len(hypscores))

    tot_hyp_len = float(np.sum(hyp_lens))
    print 'frac eq: %f ins: %f del: %f sub: %f' %\
        tuple(np.array([tot_eq, tot_ins, tot_dels, tot_subs]) / tot_hyp_len)

    print 'CER: %f' % (100.0 * tot_dist / np.sum(numphones))

    print '%d/%d sents correct' % (num_sents_correct, len(hyps))
    print 'avg len of correct sent: %f' % (correct_sents_len / float(num_sents_correct))

    disp_errs_by_pos(tot_errs_by_pos / counts_by_pos, 'err_by_pos.%s.png' % ('all' if not subset else subset))


def main(args):
    '''
    Read in data
    '''

    # NOTE Make sure synced with order dumped in runDecode.py
    fid = open(args.pk_file, 'rb')
    hyps = np.array(pickle.load(fid))
    refs = np.array(pickle.load(fid))
    hypscores = np.array(pickle.load(fid))
    refscores = np.array(pickle.load(fid))
    numphones = np.array(pickle.load(fid))
    #subsets = np.array(pickle.load(fid))
    subsets = None
    fid.close()

    compute_and_display_stats(hyps, refs, hypscores, refscores, numphones, subsets, subset=None)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pk_file', default='hyp.pk', help='Pickle file with data')
    args = parser.parse_args()
    main(args)
