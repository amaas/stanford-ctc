% AMAAS test setup for loader script
addpath ../util/
git_root = '/afs/cs.stanford.edu/u/amaas/scratch/audio/kaldi-stanford/';
fdir = '/afs/cs.stanford.edu/u/amaas/scratch/audio/kaldi-stanford/kaldi-trunk/egs/swbd/s5/data-fmllr/';
alidir = '/afs/cs.stanford.edu/u/amaas/scratch/audio/kaldi-stanford/kaldi-trunk/egs/swbd/s5/exp/tri4a_ali_100k_nodup/';

% try it
[f, a] = load_kaldi_data(git_root, fdir, alidir);