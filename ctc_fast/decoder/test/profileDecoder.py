import numpy as np
from decoder_utils import load_data, load_nnet, decode

if __name__ == '__main__':
    print 'Loading data'
    data_dict, alis, keys = load_data()
    print 'Loading neural net'
    rnn = load_nnet()

    data, labels = data_dict[keys[0]], np.array(alis[keys[0]],
            dtype=np.int32)

    probs = rnn.costAndGrad(data, labels)
    probs = np.log(probs.astype(np.float64) + 1e-30)

    method = 'bg'

    # Profiling setup
    import pstats
    import cProfile
    import pyximport
    pyximport.install()
    cProfile.runctx('decode(probs, alpha=0.0, beta=0.0, beam=10, \
            method=%s)' % method,
            globals(), locals(), 'costAndGrad.lprof')
    s = pstats.Stats('costAndGrad.lprof')
    s.strip_dirs().sort_stats('time').print_stats()
