function [] = pca_project(fn)

dat_in = ['/scail/group/deeplearning/speech/awni/kaldi-stanford/',...
    'kaldi-trunk/egs/swbd/s5/exp/nn_data_eval_fbank_alis/'];

dat_out = ['/scail/group/deeplearning/speech/awni/kaldi-stanford/',...
    'kaldi-trunk/egs/swbd/s5/exp/nn_data_eval_fbank_alis_pca_512/'];

featFormat = [dat_out 'feats%d.bin'];

pca_dim = 512;
featDim = 744; %dimension of data

load([dat_in 'pca.mat']);

U = V(:,1:pca_dim);

%% Reduce dimension and write out

fprintf('Writing file\n');

[feats] = load_kaldi_data(dat_in,fn,featDim);

% pca the data
feats = U'*feats';

fid = fopen(sprintf(featFormat,fn),'w');    

% write features
fwrite(fid,feats,'float');
clear feats;

% close files
fclose(fid);
fprintf('Completed writing file %d\n',fn);


