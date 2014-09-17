import numpy as np
import cPickle as pickle
from decoder_utils import collapse_seq
import difflib
from pprint import pprint
import editDistance as ed
from fabric.colors import green, red, blue

'''
Given 2 pickle files of decoded utterances and statistics from runDecode.py,
compare the two
'''


def main(args):
    with open(args.pk_file1, 'rb') as fin:
        hyps1 = pickle.load(fin)
        refs = pickle.load(fin)
        pickle.load(fin)  # hypscores
        pickle.load(fin)  # refscores
        numphones = pickle.load(fin)
    with open(args.pk_file2, 'rb') as fin:
        hyps2 = pickle.load(fin)
    assert len(hyps1) == len(hyps2), 'hyps have different lengths'

    differ = difflib.Differ()

    num_diff = 0
    hyp1_better = 0
    hyp2_better = 0
    for (hyp1, hyp2, ref) in zip(hyps1, hyps2, refs):
        if hyp1 == hyp2:
            continue
        num_diff += 1

        label1 = 'hyp1:'
        label2 = 'hyp2:'

        if args.score:
            dist1, _, _, _, _ = ed.edit_distance(ref, hyp1)
            dist2, _, _, _, _ = ed.edit_distance(ref, hyp2)
            if dist1 < dist2:
                hyp1_better += 1
                label1 = blue(label1)
                label2 = red(label2)
            else:
                hyp2_better += 1
                label1 = red(label1)
                label2 = blue(label2)

        print label1, collapse_seq(hyp1)
        print label2, collapse_seq(hyp2)
        pprint(list(differ.compare([collapse_seq(hyp1)], [collapse_seq(hyp2)])))
        print green(' ref:'), collapse_seq(ref)
        print '-' * 80

    if args.score:
        print 'hyp1 better: %d' % hyp1_better
        print 'hyp2 better: %d' % hyp2_better
    print 'Differ on %d/%d utts' % (num_diff, len(refs))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pk_file1', help='Pickle file with data for 1st decode run')
    parser.add_argument('pk_file2', help='Pickle file with data for 2nd decode run')
    parser.add_argument('--score', action='store_true', default=False, help='Compare scores using edit distance from reference')
    args = parser.parse_args()
    main(args)
