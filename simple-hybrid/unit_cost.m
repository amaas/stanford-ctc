%% setup paths
addpath ../util;
% need minfunc for gradient check
minFuncPath = '/afs/cs/u/amaas/scratch/matlab_trunk/parallel_proto/minFunc/';
addpath(minFuncPath);

%% experiment parameters
eI = [];
eI.useGpu = 0;
eI.inputDim = 3;
eI.outputDim = 4;
eI.layerSizes = [4 2, eI.outputDim];
eI.lambda = 0; %1e-5;
eI.activationFn = 'tanh';
disp(eI);
%% initialize weights and synthetic data
stack = initialize_weights(eI);
theta = stack2params(stack);
m = 100;
data = randn(m, eI.inputDim)';
labels = randn(eI.outputDim,m,1)';
[~,labels] = max(labels,[], 2);

%% test predicitons only
[~,~,~,~,~,~, pred] = spNetCostSlave(theta, eI, data, labels, true);
disp(pred);
%% check gradient
options.display = 'iter';
options.derivativeCheck = 'on';
[theta, fVal, coFlag, coInfo] = minFunc(@spNetCostSlave, ...
    theta, options, eI, data, labels);
