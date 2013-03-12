function [feats, utt_dat, alis] = load_kaldi_data(data_dir, file_num, ...
                                                  featDim)
%% Returns feats and alis matrix and vector with feature and alignment data
% Input:
% data_dir is subdirectory with written feats binary file and ali and key txt files
%    e.g. /afs/cs.stanford.edu/u/awni/luster_awni/kaldi-trunk/egs/swbd/s5/exp/nn_data/

featsf = sprintf([data_dir 'feats%d.bin'],file_num);
alisf = sprintf([data_dir 'alis%d.txt'],file_num);
keysf = sprintf([data_dir 'keys%d.txt'],file_num);

if ~exist(featsf,'file')
    fprintf('Feature file %s does not exists\n',featsf);
    return;
end;

%Load features from binary file
fid = fopen(featsf);
feats = fread(fid,[featDim inf],'float');

%Load alignments from txt file if it exists
if exist(alisf,'file')
    alis = load(alisf);
    %Add 1 to 0-indexed kaldi state labels
    alis = alis+1; 
else
    alis = []; %no alignments loaded i.e. testing
end;

%Load key strings from txt file
utt_dat = {};

if exist(keysf,'file')
    [utt_dat.keys utt_dat.sizes]= textread(keysf,'%s %d');
else
    utt_dat.keys=[];
    utt_dat.sizes=[];
end;

%numSamples = sum(utt_dat.sizes);
%featdim = size(feats,1)/numSamples;

%feats = reshape(feats',featDim,numSamples)';
feats = feats';
end
