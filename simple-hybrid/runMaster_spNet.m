% This script called by startMaster_spNet
%% initialize weights
rand('seed',1);
stack = initialize_weights(eI);
theta = stack2params(stack);
% this is only used by adagrad
sumsqgrad = ones(size(theta)) * 1e-3;
%% load a theta instead
%% TODO fix loading partially trained model
startT = 0;
% if startT > 0
%     load(sprintf('tmp/spNet_%d.mat',startT));
%     startT = startT + 1;
%     %eI.nu=1e-6;
% end;

%% Setup data loading
dat_dir = ['/scail/group/deeplearning/speech/awni/kaldi-stanford/',...
    'kaldi-trunk/egs/swbd/s5/exp/nn_data_100k/'];
%dat_dir = 'tmp/';
% HACK loading tiny data instead
%load tmp/micro_feat.mat;

%% loop ower mini-batch epochs
% storing the mb data in globals
%global mbFeat;
%global mbLabel;
fValHist = [];
accHist = [];
for t = startT : eI.numEpoch
    %Make random permutation of file to load for each epoch
    fileList = randperm(eI.numFiles);

    for fn = 1 : eI.numFiles
        %load chunk of data
        [feat, label_ind, utt_dat] = load_kaldi_data(dat_dir,fileList(fn));
        assert(size(feat,1) == size(label_ind,1));
        numExamples = size(label_ind,1);

        % shuffle minibatches
        rp = randperm(numExamples);
        feat = feat(rp,:);
        label_ind = label_ind(rp);
        %% run optimizer
        % TODO split optimizer to separate function if necessary
        numMb = floor(numExamples / eI.miniBatchSize);

        tic;
        for m = 1 : eI.miniBatchSize : (eI.miniBatchSize * numMb)
            mbFeat = feat(m:(m+eI.miniBatchSize-1),:)';
            mbLabel = label_ind(m:(m+eI.miniBatchSize-1));        
            [f, g, nc, ne] = spNetCostSlave(theta, eI, mbFeat, mbLabel);
            % choice of update rule
            if strcmpi(optimOpt.Method, 'adagrad')
                % AdaGrad
                sumsqgrad = sumsqgrad + g.^2;
                % be careful with floating point precision with sumsqgrad
                theta = theta - (eI.sgdLearningRate*g) ./ sqrt(sumsqgrad);
            else
                % SGD
                theta = theta - (eI.sgdLearningRate * g);
            end;
            fValHist = [fValHist; f];
            accHist = [accHist; nc/ne];
        end;
        toc;

        %% cache - save after seeing every utterance in each file
        % every eI.numFiles saves will be a full pass over all the data
        fullFilename = sprintf([eI.outputDir 'spNet_%d.mat'], t);
        save(fullFilename, 'eI','theta','stack','fValHist','accHist');
    end;

end;

