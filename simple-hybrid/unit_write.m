%%Test write_likelihoods function

addpath ../util;
git_root = '/afs/cs.stanford.edu/u/awni/luster_awni/kaldi-stanford/';
dat_dir = [git_root 'kaldi-trunk/egs/swbd/s5/exp/nn_data_dev/'];

nn_model = 'dummy_model';  %not yet used by write script
num_files = 4; %number of files utterances split amongst

write_likelihoods(git_root,dat_dir,num_files,nn_model);
