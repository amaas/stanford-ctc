from srilm import LM
from decoder_config import SPACE

'''
Compute perplexity of character LM on some transcripts
'''

def preproc_transcript(transcript, num_lines=float('inf')):
    utts = list()
    l = 0
    for line in transcript.split('\n'):
        # Remove utterance ids
        utt = line.split(' ', 1)[1]
        utts.append(utt)
        l += 1
        if l >= num_lines:
            break
    return utts

def preproc_utts(utts):
    '''
    Convert raw text into character language model format
    '''
    # Split characters up, add start and end symbols, replace spaces
    text = [['<s>'] + list(utt) + ['</s>'] for utt in utts]
    text = [[c if c != ' ' else SPACE for c in utt] for utt in text]
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
    ORDER = 5
    NUM_LINES = float('inf')

    lm = LM('/scail/data/group/deeplearning/u/zxie/biglm/lms/biglm.%dg.arpa' % ORDER)

    with open('/scail/group/deeplearning/speech/zxie/kaldi-stanford/kaldi-trunk/egs/swbd/s5b/data/train/text', 'r') as fin:
        transcript = fin.read().strip()

    utts = preproc_transcript(transcript, num_lines=NUM_LINES)
    text = preproc_utts(utts)
    pp = compute_pp(text, lm, ORDER)

    print 'Perplexity: %f' % pp
