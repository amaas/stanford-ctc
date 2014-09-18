import numpy as np
from srilm import LM
from decoder_config import LM_ARPA_FILE, SPACE, LM_ORDER

'''
Sample text from character language model
'''


def sample_continuation(s, lm, order, alpha=1.0):
    # Higher alpha -> more and more like most likely sequence
    n = lm.vocab.max_interned()
    probs = np.empty(n, dtype=np.float64)
    for k in range(1, n + 1):
        # NOTE Assumes log10
        probs[k-1] = (10 ** lm.logprob_strings(lm.vocab.extern(k), s[0:5])) ** alpha
    probs = probs / np.sum(probs)
    c = np.random.choice(range(1, n + 1), p=probs)
    c = lm.vocab.extern(c)
    return [c]


if __name__ == '__main__':
    print 'Loading LM...'
    lm = LM(LM_ARPA_FILE)
    print 'Done.'

    SAMPLE_LENGTH = 100
    ALPHA = 1.0

    for j in range(5):
        # NOTE List is in reverse
        sample_string = ['<s>']
        for k in range(SAMPLE_LENGTH):
            # Work in reverse
            sample_string = sample_continuation(sample_string, lm, LM_ORDER, alpha=ALPHA) + sample_string
            # Don't sample after </s>, get gibberish
            if sample_string[0] == '</s>':
                break

        s = [c if c != SPACE else ' ' for c in sample_string]
        print ''.join(s[::-1])
