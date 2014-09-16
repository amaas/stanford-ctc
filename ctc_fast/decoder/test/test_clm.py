import numpy as np
from decoder_utils import load_data, load_nnet, int_to_char,\
        collapse_seq, decode, load_char_map

if __name__ == '__main__':
    print 'Loading data'
    fnum = 1
    data_dict, alis, keys = load_data(fnum)
    print 'Loading neural net'
    rnn = load_nnet()

    for k in range(0, 10):
        data, labels = data_dict[keys[k]], np.array(alis[keys[k]],
                dtype=np.int32)

        probs = rnn.costAndGrad(data, labels)
        probs = np.log(probs.astype(np.float64) + 1e-30)

        hyp, hypScore, refscore = decode(probs,
                alpha=0.0, beta=0.0, beam=40, method='clm2')
        hyp_pmax, _, _ = decode(probs,
                alpha=1.0, beta=0.0, method='pmax')

        char_map = load_char_map()
        if labels is not None:
            print '  labels:', collapse_seq(int_to_char(labels, char_map))
        print ' top hyp:', collapse_seq(int_to_char(hyp, char_map))
        print 'pmax hyp:', collapse_seq(int_to_char(hyp_pmax, char_map))
        print 'score:', hypScore
        #print 'ref score:', refScore
