import numpy as np
from decoder_utils import load_data, load_nnet, int_to_char,\
        collapse_seq, decode, load_chars

if __name__ == '__main__':
    print 'Loading data'
    fnum = 1
    data_dict, alis, keys = load_data(fnum)
    print 'Loading neural net'
    rnn = load_nnet()

    for k in range(0, 10):
        data, labels = data_dict[keys[k]], np.array(alis[keys[k]],
                dtype=np.int32)

        hyp, hypScore, refscore = decode(data, labels, rnn,
                alpha=1.0, beta=0.0, beam=10, method='clm')
        hyp_pmax, _, _ = decode(data, labels, rnn,
                alpha=1.0, beta=0.0, method='pmax')

        chars = load_chars()
        if labels is not None:
            print '  labels:', collapse_seq(int_to_char(labels, chars))
        print ' top hyp:', collapse_seq(int_to_char(hyp, chars))
        print 'pmax hyp:', collapse_seq(int_to_char(hyp_pmax, chars))
        print 'score:', hypScore
        #print 'ref score:', refScore
