function [feats, alis, utt_dat] = load_kaldi_data(data_dir, file_num)
%% Returns feats and alis matrix and vector with feature and alignment data
% Input:
% data_dir is subdirectory with written feats binary file and ali and key txt files
%    e.g. /afs/cs.stanford.edu/u/awni/luster_awni/kaldi-trunk/egs/swbd/s5/exp/nn_data/

feats = sprintf([data_dir 'feats%d.bin'],file_num);
alis = sprintf([data_dir 'alis%d.txt'],file_num);
keys = sprintf([data_dir 'keys%d.txt'],file_num);

if ~exist(feats,'file')
    fprintf('Feature file %s does not exists\n',feats);
    return;
elseif ~exist(alis,'file')
    fprintf('Alignment file %s does not exists\n',alis);
    return;
elseif ~exist(keys,'file')
    fprintf('Keys file %s does not exists\n',keys);
    return;
end

%Load features from binary file
fid = fopen(feats);
feats = fread(fid,'float');

%Load alignments from txt file
alis = load(alis);

%Add 1 to 0-indexed kaldi state labels
alis = alis+1; 

%Load key strings from txt file
utt_dat = {};
[utt_dat.keys utt_dat.sizes]= textread(keys,'%s %d');

numSamples = size(alis,1);
featdim = size(feats,1)/numSamples;

feats = reshape(feats',featdim,numSamples)';

end
