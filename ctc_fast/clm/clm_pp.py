import math
from srilm import LM
from decoder_config import LM_ARPA_FILE, LM_ORDER
from prep_text import preproc_transcript, preproc_utts

'''
Compute perplexity of character LM on some transcripts
'''

def compute_pp(text, lm, order):
    # Working in log space
    pp = 0.0

    n = 0
    for utt in text:
        # Skip p(<s> | []) = -inf
        for k in range(1, len(utt)):
            c = utt[k]
            seq = utt[max(0, k-order+1):k]
            #print c, seq
            # Work w/ reversed context due to srilm
            seq = seq[::-1]
            pp = pp + -1 * lm.logprob_strings(c, seq)
            n += 1

    pp = pp / n

    # NOTE Assumes probs given back in log10
    pp = 10 ** pp

    return pp

if __name__ == '__main__':
    NUM_LINES = float('inf')

    print 'Loading LM %s' % LM_ARPA_FILE
    lm = LM(LM_ARPA_FILE)
    print 'Done.'

    # NOTE FIXME These texts still have hesitation symbols and the like
    with open('/scail/group/deeplearning/speech/zxie/kaldi-stanford/kaldi-trunk/egs/swbd/s5b/data/train/text', 'r') as fin:
        transcript = fin.read()

    utts = preproc_transcript(transcript, num_lines=NUM_LINES)
    text = preproc_utts(utts)
    pp = compute_pp(text, lm, LM_ORDER)

    print 'Perplexity: %f' % pp
    print 'Bits/char: %f' % math.log(pp, 2)
