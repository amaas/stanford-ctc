% AMAAS test setup for loader script
addpath ../util/
%amaas_git_root = '/afs/cs.stanford.edu/u/amaas/scratch/audio/kaldi-stanford/';
git_root = '/scail/group/deeplearning/speech/awni/kaldi-stanford/';

%fdir = '/afs/cs.stanford.edu/u/amaas/scratch/audio/kaldi-stanford/kaldi-trunk/egs/swbd/s5/data-fmllr/';
%alidir = '/afs/cs.stanford.edu/u/amaas/scratch/audio/kaldi-stanford/kaldi-trunk/egs/swbd/s5/exp/tri4a_ali_100k_nodup/';

dat_dir = [git_root 'kaldi-trunk/egs/swbd/s5/exp/nn_data/'];


[f, a, utt_dat] = load_kaldi_data(dat_dir);

disp(size(f))
disp(size(a))
disp(size(utt_dat.keys))
disp(size(utt_dat.sizes))
