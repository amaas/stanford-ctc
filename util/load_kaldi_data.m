function [feats, alis, utt_dat] = load_kaldi_data_new(data_dir)
  %% Returns feats and alis matrix and vector with feature and alignment data
  % Input:
  % data_dir is subdirectory with written feats binary file and ali and key txt files
  %    e.g. /afs/cs.stanford.edu/u/awni/luster_awni/kaldi-trunk/egs/swbd/s5/exp/nn_data/

  
  feats = [data_dir 'feats'];

  alis = [data_dir 'alis.txt'];
  keys = [data_dir 'keys.txt'];

  %Load features from binary file
  fid = fopen(feats);
  feats = fread(fid,'float');

  %Load alignments from txt file
  alis = load(alis);

  %Load key strings from txt file
  utt_dat = {};
  [utt_dat.keys utt_dat.sizes]= textread(keys,'%s %d');

  numSamples = size(alis,1);
  featdim = size(feats,1)/numSamples;

  feats = reshape(feats',featdim,numSamples)';

end
