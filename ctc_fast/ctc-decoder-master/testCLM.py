from profileDecoder import load_data, load_nnet, char_seq_to_string

if __name__ == '__main__':
    print 'Loading data'
    data, labels = load_data()
    print 'Loading neural net'
    # NOTE Net loads prefix tree and LM
    rnn = load_nnet()

    print 'Computing likelihoods'
    hyp, hypScore, refScore = rnn.costAndGrad(data, labels)

    print '-' * 80
    if labels is not None:
        print 'labels:', char_seq_to_string(int_to_char(labels, rnn.pt))
    prefix = int_to_char(hyp, rnn.pt)
    print 'top hyp:', char_seq_to_string(prefix)
    print 'score:', hypScore
    print 'ref score:', refScore
