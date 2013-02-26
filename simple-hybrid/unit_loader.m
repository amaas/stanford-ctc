% AMAAS test setup for loader script
addpath ../util/

dat_dir = ['/scail/group/deeplearning/speech/awni/kaldi-stanford/',...
    'kaldi-trunk/egs/swbd/s5/exp/nn_data/'];
dat_dir = 'tmp/';

[f, a, utt_dat] = load_kaldi_data(dat_dir);

disp(size(f))
disp(size(a))
disp(size(utt_dat.keys))
disp(size(utt_dat.sizes))
