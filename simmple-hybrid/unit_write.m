%%Test write_likelihoods function

addpath ../util;
git_root = '/afs/cs.stanford.edu/u/awni/luster_awni/kaldi-stanford/';
dat_dir = [git_root 'kaldi-trunk/egs/swbd/s5/exp/nn_data/'];

nn_model = 'dummy_model';  %not yet used by write script

write_likelihoods(git_root,dat_dir,nn_model);

