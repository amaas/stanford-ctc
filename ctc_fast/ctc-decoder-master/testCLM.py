import numpy as np
from prefixTree import load_chars
from profileDecoder import load_data, load_nnet, char_seq_to_string,\
        int_to_char

if __name__ == '__main__':
    print 'Loading data'
    data_dict, alis, keys = load_data()
    print 'Loading neural net'
    # NOTE Net loads prefix tree and LM
    rnn = load_nnet()

    for k in range(10):
        data, labels = data_dict[keys[k]], np.array(alis[keys[k]],
                dtype=np.int32)
        hyp, hypScore, refScore = rnn.costAndGrad(data, labels)

        print '-' * 80
        chars = load_chars()
        if labels is not None:
            print 'labels:', char_seq_to_string(int_to_char(labels, chars))
        print 'top hyp:', char_seq_to_string(int_to_char(hyp, chars))
        print 'score:', hypScore
        #print 'ref score:', refScore
