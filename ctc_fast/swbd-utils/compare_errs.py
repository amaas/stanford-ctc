import pickle
import numpy as np
from editDist import edit_distance as ed
from errorAnalysis import disp_err_corr

'''
Given two pickle files, compare the hyps from each to the ref
'''


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pk1')
    parser.add_argument('pk2')
    args = parser.parse_args()

    # Read in hyps and refs

    fid = open(args.pk1, 'rb')
    hyps1 = np.array(pickle.load(fid))
    refs = np.array(pickle.load(fid))
    fid.close()

    fid = open(args.pk2, 'rb')
    hyps2 = np.array(pickle.load(fid))
    fid.close()

    hyp1_better = 0
    hyp2_better = 0
    for (h1, h2, ref) in zip(hyps1, hyps2, refs):
        dist1, eq1, ins1, dels1, subs1, errs_by_pos1, hyp_corr1, ref_corr1 = ed(h1, ref)
        dist2, eq2, ins2, dels2, subs2, errs_by_pos2, hyp_corr2, ref_corr2 = ed(h2, ref)

        if dist1 < dist2:
            hyp1_better += 1
        elif dist2 < dist1:
            hyp2_better += 1

        # FIXME Just display cases where hyp2 is better
        if dist2 < dist1:
            disp_err_corr(hyp_corr1, ref_corr1)
            disp_err_corr(hyp_corr2, ref_corr2)
            print

    print '%d cases hyp1 better' % hyp1_better
    print '%d cases hyp2 better' % hyp2_better
    print '%d cases tied' % (len(refs) - hyp1_better - hyp2_better)
