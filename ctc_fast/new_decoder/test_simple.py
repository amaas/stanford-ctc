import numpy as np
import cPickle as pickle
import decoder



def main(in_file, char_file, ali_file,num_to_print):
    with open(in_file,'r') as f: 
        ll_dict = pickle.load(f)

    # read char mapping (need it here for alignments)
    with open(char_file,'r') as f:
        phone_list = map(lambda x: x.rstrip().split()[0], f.readlines())
    # prepend symbol for blank
    phone_list.insert(0,'_')
        
    # read alignments
    with open(ali_file,'r') as f:
        ali_dict = {}
        for l in f.readlines():
            l_split = l.rstrip().split()
            ali_dict[l_split[0]] = ''.join([phone_list[int(x)] for x in l_split[1:]])

    # create a decoder
    dec = decoder.ArgmaxDecoder()
    dec.load_chars(char_file)
    n_printed = 0
    for i,k in enumerate(ll_dict):
        
        hyp = dec.decode(ll_dict[k].astype(np.double))
        print hyp
        print ali_dict[k]

        n_printed+= 1
        if n_printed >= num_to_print:
            break



if __name__=='__main__':
    ll_path = '/scail/scratch/group/deeplearning/speech/amaas/kaldi-stanford/stanford-nnet/ctc_fast/swbd_eval2000_lik/'
    in_file = ll_path + 'loglikelihoods_1.pk'
    char_file = ll_path + 'chars.txt'
    ali_file = ll_path + 'alis1.txt'

    main(in_file, char_file,ali_file,10)

