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
eI.layerSizes = [2048, 2048, 2048, eI.outputDim];
eI.lambda = 1e-5;
eI.activationFn = 'tanh';

%% initialize optimization (mini-batch) parameters
optimOpt = [];
eI.numEpoch = 1000;
% 'adagrad' and 'sgd' are valid options here
% learning rate for adagrad should be higher
optimOpt.Method = 'adagrad';
% setup learning rates Etc
eI.miniBatchSize = 512;
eI.sgdLearningRate = 1e-3;

%% setup gpu
if eI.useGpu
    addpath('/afs/cs.stanford.edu/package/jacket/2.0-20111221/jacket/engine');
    % TODO might want to warm up the GPU here
end;
%% call run script
disp(eI);
runMaster_spNet;
