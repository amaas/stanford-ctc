import numpy as np
from srilm import LM
from decoder_config import LM_ARPA_FILE, SPACE

'''
Sample text from character language model
'''


def sample_continuation(s, lm, order):
    n = lm.vocab.max_interned()
    probs = np.empty(n, dtype=np.float64)
    for k in range(1, n + 1):
        probs[k-1] = 10 ** lm.logprob_strings(lm.vocab.extern(k), s[0:5])
    probs = probs / np.sum(probs)
    c = np.random.choice(range(1, n + 1), p=probs)
    c = lm.vocab.extern(c)
    return [c]


if __name__ == '__main__':
    #clm = LM(LM_ARPA_FILE)
    lm = LM('/scail/data/group/deeplearning/u/zxie/biglm/lms/biglm.5g.arpa')

    ORDER = 5
    SAMPLE_LENGTH = 100

    for j in range(5):
        sample_string = ['<s>']
        for k in range(SAMPLE_LENGTH):
            # Work in reverse
            sample_string = sample_continuation(sample_string, lm, ORDER) + sample_string

        s = [c if c != SPACE else ' ' for c in sample_string]
        print ''.join(s[::-1])
