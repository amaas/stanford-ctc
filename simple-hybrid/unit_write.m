%%Test write_likelihoods function

addpath ../util;
git_root = '/afs/cs.stanford.edu/u/awni/luster_awni/kaldi-stanford/';

%location to read and write data
dat_dir = [git_root 'kaldi-trunk/egs/swbd/s5/exp/nn_data_eval/'];

%location of nueral net params
nn_model = [git_root, 'stanford-nnet/simple-hybrid/', ...
                    '4hidden_1024_diff_lr/spNet_3.mat'];

num_files = 5; %number of files utterances split amongst

write_likelihoods(git_root,dat_dir,num_files,nn_model);
