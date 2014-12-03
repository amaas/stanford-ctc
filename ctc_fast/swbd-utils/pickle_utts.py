import re
import pickle
from runDecode import PATTERN

'''
Write utts from text files to pickle format so that we can
run error analysis
'''

def load_utts(f):
    utts = list()
    with open(f, 'r') as fin:
        lines = fin.read().strip().splitlines()
    for l in lines:
        utt = l.split(' ', 1)[1].strip()
        utt = ''.join([c for c in utt if PATTERN.match(c)])
        utt = re.sub('\s\s+', ' ', utt)
        utts.append(utt)
    return utts

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('hyp', help='text file containing hyps')
    parser.add_argument('ref', help='text file containing refs')
    parser.add_argument('out', help='pickle file to write outputs to')
    args = parser.parse_args()

    hyps = load_utts(args.hyp)
    refs = load_utts(args.ref)
    # NOTE Assuming that they're in the same order...
    assert len(hyps) == len(refs)
    print '%s utts' % len(hyps)

    numphones = [len(r) for r in refs]
    # Fill in bogus values that don't affect CER computation
    hypscores = [0 for r in refs]
    refscores = [0 for r in refs]
    subsets = [None for r in refs]
    alignments = [None for r in refs]

    pkid = open(args.out, 'wb')
    pickle.dump(hyps, pkid)
    pickle.dump(refs, pkid)
    pickle.dump(hypscores, pkid)
    pickle.dump(refscores, pkid)
    pickle.dump(numphones, pkid)
    pickle.dump(subsets, pkid)
    pickle.dump(alignments, pkid)
    pkid.close()
