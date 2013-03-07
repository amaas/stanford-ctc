%% setup paths
addpath ../util;
%basePath = '/afs/cs.stanford.edu/u/amaas/scratch/audio/audio_repo/matlab_wd/';
%minFuncPath = '/afs/cs/u/amaas/scratch/matlab_trunk/parallel_proto/minFunc/';
%rpcPath ='/afs/cs/u/amaas/scratch/matlab_trunk/temporal_learning/matlabserver_r1';
%jacketPath = '/afs/cs.stanford.edu/package/jacket/2.0-20111221/jacket/engine/';
% 
% addpath(minFuncPath);
% addpath(basePath);
% addpath(rpcPath);

%% experiment parameters
eI = [];
eI.useGpu = 1;
eI.inputDim = 300;
eI.outputDim = 3034;
eI.layerSizes = [1024, 1024, 1024, 1024, eI.outputDim];
eI.lambda = 1e-7;
eI.activationFn = 'tanh';
eI.numFiles = 39; %number of files data is split into
eI.outputDir = '4hidden_1024_shuff/';
eI.datDir = ['/scail/group/deeplearning/speech/awni/kaldi-stanford/',...
'kaldi-trunk/egs/swbd/s5/exp/nn_data_100k_shuff/'];



if ~exist(eI.outputDir,'dir')
    mkdir(eI.outputDir);
end

%% initialize optimization (mini-batch) parameters
optimOpt = [];
eI.numEpoch = 10;
% 'adagrad' and 'sgd' are valid options here
% learning rate for adagrad should be higher
optimOpt.Method = 'sgd';
% setup learning rates Etc
eI.miniBatchSize = 512;
eI.sgdLearningRate = 1e-1;

%% setup gpu
if eI.useGpu
    addpath('/afs/cs.stanford.edu/package/jacket/2.0-20111221/jacket/engine');
    % TODO might want to warm up the GPU here
end;
%% call run script
disp(eI);
runMaster_spNet;
