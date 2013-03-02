function [feats, alis] = load_kaldi_data_old(git_root, feats_subdir, ali_subdir)
  %% Returns feats and alis matrix and vector with feature and alignment data
  % Input:
  % kaldi_dir is top level location of example 
  %    e.g. /afs/cs.stanford.edu/u/awni/luster_awni/kaldi-trunk/egs/swbd/s5
  % feats_subdir is subdirectory with feats to transform and load [must already be aligned!]
  %    e.g. data-fmllr/train_100k_nodup
  % ali_subid is location of alignments corresponding to features
  %    e.g. exp/tri4a_ali_100k_nodup

  %Root of Kaldi Directory with compiled code
  % kr = '/scail/group/deeplearning/speech/awni/kaldi-trunk';
  % TODO better handling of paths since we wont always use swbd
  kaldi_dir = [git_root, 'kaldi-trunk/egs/swbd/s5'];
 
  path = '';
  dList = {'utils', 'src/bin', 'src/stanford-bin', 'tools/openfst/bin', ...
      'src/fstbin', 'src/gmmbin', 'src/featbin', 'src/lm', 'src/sgmmbin',...
      'src/sgmm2bin', 'src/fgmmbin', 'src/latbin', 'src/nnetbin', ...
      'src/nnet-cpubin'};
  
  for d = dList
      path = [path, git_root, 'kaldi-trunk/', d{1}, '/:'];
  end;
  path = [path, git_root, 'kaldi-trunk/'];

  %   %Path with all utils and bins
  %   path=[kaldi_dir '/utils/:' kr '/src/bin:' kr '/src/stanford-bin:' kr '/tools/openfst/bin:'...
  %         kr '/src/fstbin/:' kr '/src/gmmbin/:' kr '/src/featbin/:' kr '/src/lm/:' kr ...
  %         '/src/sgmmbin/:' kr '/src/sgmm2bin/:' kr '/src/fgmmbin/:' kr '/src/latbin/:' kr...
  %         '/src/nnetbin:' kr '/src/nnet-cpubin:' kaldi_dir];


  % Set path so sub processes know where everything is
  setenv('PATH', [getenv('PATH') ';D:' path]);


  
  %build tmp_feats.txt and tmp_alis.txt from scp file using init_features.sh subroutine
  tmp_feats = [pwd '/tmp_feats'];
  tmp_alis = [pwd '/tmp_alis.txt'];

  init_script = [kaldi_dir '/stanford-utils/init_features.sh'];
  
  command = [init_script ' ' kaldi_dir ' '  feats_subdir ' ' ...
             ali_subdir ' ' tmp_feats ' ' tmp_alis];
  disp(command);

  [s r] = system(command);     
  assert(~s,['Failed to build text file of train data, err: ' r]);

  %Load features and alignments from temporary binary and txt files
  fid = fopen(tmp_feats);
  feats = fread(fid,'float');

  alis = load(tmp_alis);

  numSamples = size(alis,1);
  featdim = size(feats,1)/numSamples;

  feats = reshape(feats',featdim,numSamples)';

  %Remove feats and alis txt files
  system(['rm ' tmp_feats]);
  system(['rm ' tmp_alis]);




end
