% AMAAS test setup for loader script
addpath ../util/

dat_dir = ['/scail/group/deeplearning/speech/awni/kaldi-stanford/',...
    'kaldi-trunk/egs/swbd/s5/exp/nn_data_100k_fmllr/'];

file_num = 1; %specifies which file to load
featDim = 500;

[f, utt_dat, a] = load_kaldi_data(dat_dir,file_num,featDim);

disp(size(f))
disp(size(utt_dat.keys))
disp(size(utt_dat.sizes))
disp(size(a))
