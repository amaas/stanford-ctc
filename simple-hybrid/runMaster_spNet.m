% This script called by startMaster_spNet
%% initialize weights
rand('seed',1);
stack = initialize_weights(eI);
theta = stack2params(stack);
% this is only used by adagrad
sumsqgrad = ones(size(theta));
%% load a theta instead
%% TODO fix loading partially trained model
startT = 0;
% if startT > 0
%     load(sprintf('tmp/spNet_%d.mat',startT));
%     startT = startT + 1;
%     %eI.nu=1e-6;
% end;

%% Setup data loading
dat_dir =eI.datDir;

%% loop ower mini-batch epochs
% storing the mb data in globals
%global mbFeat;
%global mbLabel;
ceValHist = [];
wValHist = [];
accHist = [];
%load tmp/debug_theta.mat;
% setup momentum
grad_sum = zeros(size(theta));
iter = uint64(0);
for t = startT : eI.numEpoch
    %Make random permutation of file to load for each epoch
    fileList = randperm(eI.numFiles);

    for fn = 1 : eI.numFiles
        %load chunk of data
        [feat, utt_dat, label_ind] = load_kaldi_data(dat_dir,fileList(fn));
        %load tmp/micro_feat.mat;
        %label_ind = label_ind + 1;
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
            [f, g, nc, ne, ceCost, wCost] = spNetCostSlave(theta, eI, mbFeat, mbLabel);
            % choice of update rule
            if strcmpi(optimOpt.Method, 'adagrad')
                % AdaGrad
                sumsqgrad = sumsqgrad + g.^2;
                % be careful with floating point precision with sumsqgrad
                theta = theta - (eI.sgdLearningRate*g) ./ sqrt(sumsqgrad);
            else
                % SGD
                grad_sum = eI.momentum * grad_sum + g;
                theta = theta - (eI.sgdLearningRate * grad_sum);
            end;            
            ceValHist = [ceValHist; ceCost];
            wValHist = [wValHist; wCost];
            accHist = [accHist; nc/ne];
            iter = iter + 1;
            % change momentum after set number of iterations
            if iter > eI.momentumIncrease
                eI.momentum = 0.9;
            end;
                
        end;
        toc;
        %figure(1);plot(theta,'kx'); 
        %         figure(2);
        %         xInd = eI.miniBatchSize .* (1:numel(accHist)); plot(xInd, ceValHist,'kx'); hold on; plot(xInd, wValHist,'rx'); plot(xInd, accHist,'bx'); title(eI.outputDir); hold off;
        %         drawnow;
        %% cache - save after seeing every utterance in each file
        % every eI.numFiles saves will be a full pass over all the data
        % fullFilename = sprintf([eI.outputDir 'spNet_e%d_f%d.mat'], t,fn);
        fullFilename = sprintf([eI.outputDir 'spNet_e%d.mat'], t);
        save(fullFilename, 'eI','theta', 'iter', 'ceValHist','wValHist','accHist');
    end;
        % optimOpt.Method = 'adagrad';
end;

