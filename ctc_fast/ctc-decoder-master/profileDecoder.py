from decoder_utils import load_data, load_nnet, decode

if __name__ == '__main__':
    print 'Loading data'
    data_dict, alis, keys = load_data()
    print 'Loading neural net'
    rnn = load_nnet()

    method = 'bg'

    # Profiling setup
    import pstats
    import cProfile
    import pyximport
    pyximport.install()
    cProfile.runctx('decode(data, labels, rnn, method=%s)' % method,
            globals(), locals(), 'costAndGrad.lprof')
    s = pstats.Stats('costAndGrad.lprof')
    s.strip_dirs().sort_stats('time').print_stats()
