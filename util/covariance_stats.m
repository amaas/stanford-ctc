function [] = covariance_stats(fn)

dat_in = ['/scail/group/deeplearning/speech/awni/kaldi-stanford/',...
    'kaldi-trunk/egs/swbd/s5/exp/nn_data_full_fbank/'];

dat_out = ['/scail/group/deeplearning/speech/awni/kaldi-stanford/',...
    'kaldi-trunk/egs/swbd/s5/exp/nn_data_full_fbank_pca/'];


featDim = 744; %dimension of data

%% Build covariance matrix

fprintf('Calculating covariance matrix\n');

% Compute covariance matrix for one file and save
[feats] = load_kaldi_data(dat_in,fn,featDim);

numFeats = size(feats,2);

sigma = feats'*feats;

fprintf('File %d stastistics collected\n',fn);


sigma = sigma/numFeats;
dat_name=sprintf('covariance%d.mat',fn);
save([data_out dat_name],'sigma','numFeats');


