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
eI.useGpu = 0;
eI.inputDim = 300;
eI.outputDim = 3034;
eI.layerSizes = [500, 500, eI.outputDim];
eI.lambda = 1e-5;
eI.activationFn = 'tanh';

%% initialize optimization (mini-batch) parameters
optimOpt = [];
eI.numEpoch = 1000;
% 'sgdMinFunc' uses non-minFunc sgd with learning rates in eI
optimOpt.Method = 'sgdMinFunc';
% setup learning rates Etc
eI.miniBatchSize = 256;
eI.sgdLearningRate = 1e-4;


%% call run script
disp(eI);
runMaster_spNet;
