import math
from srilm import LM
from decoder_config import SPACE, SPECIALS_LIST,\
        LM_ARPA_FILE, LM_ORDER

'''
Compute perplexity of character LM on some transcripts
'''

def preproc_transcript(transcript, num_lines=float('inf')):
    utts = list()
    l = 0
    for line in transcript.split('\n'):
        # Remove utterance ids and lowercase
        utt = line.split(' ', 1)[1].lower()
        utts.append(utt)
        l += 1
        if l >= num_lines:
            break
    return utts

def preproc_utts(utts):
    '''
    Convert raw text into character language model format
    Split characters up, add start and end symbols, replace spaces
    Avoid splitting up characters in specials tokens list
    '''
    utt_sents = [utt.split(' ') for utt in utts]
    # Split up characters while preserving special characters and adding back spaces
    text = [[list(sent[k] + (' ' if k < len(sent) - 1 else ''))
            if sent[k] not in SPECIALS_LIST else [sent[k]]
            for k in xrange(len(sent))] for sent in utt_sents]
    # Flatten nested list
    text = [[c for w in s for c in w] for s in text]
    # Add start and end symbols and replace spaces with space token
    text = [['<s>'] + [c if c != ' ' else SPACE for c in utt] + ['</s>'] for utt in text]
    return text

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
        transcript = fin.read().strip()

    utts = preproc_transcript(transcript, num_lines=NUM_LINES)
    text = preproc_utts(utts)
    pp = compute_pp(text, lm, LM_ORDER)

    print 'Perplexity: %f' % pp
    print 'Bits/char: %f' % math.log(pp, 2)
