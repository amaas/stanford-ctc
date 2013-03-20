%%Test write_likelihoods function

addpath ../util;
git_root = '/afs/cs.stanford.edu/u/awni/luster_awni/kaldi-stanford/';


%location to read data for forward prop
dat_in = [git_root 'kaldi-trunk/egs/swbd/s5/exp/nn_data_eval/'];

nn_base = [git_root, 'stanford-nnet/simple-hybrid/2hidden_2048_relu_fmllr/'];
%location to write log-likelihoods
dat_out = nn_base;
%location of nueral net params
nn_model = [nn_base 'spNet_2.mat'];

num_files = 5; %number of files utterances split amongst

write_likelihoods(git_root,dat_in,dat_out,num_files,nn_model);
