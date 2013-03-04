% AMAAS test setup for loader script
addpath ../util/

dat_dir = ['/scail/group/deeplearning/speech/awni/kaldi-stanford/',...
    'kaldi-trunk/egs/swbd/s5/exp/nn_data_100k/'];

file_num = 1; %specifies which file to load

[f, a, utt_dat] = load_kaldi_data(dat_dir,file_num);

disp(size(f))
disp(size(a))
disp(size(utt_dat.keys))
disp(size(utt_dat.sizes))
