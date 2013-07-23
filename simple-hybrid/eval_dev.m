% run a bunch of models on the dev set
addpath ../util;
model_base = '../../../../runtime/audio/phoneme_class/icml2013/';
model_list = {'single_2048_2048_2048_2048_relu',...
    'single_2048_2048_2048_2048_reluhard',...
    'single_2048_2048_2048_2048_tanh'};

% 'single_2048_2048_2048_2048_1024_relu',...
% 'single_2048_2048_2048_2048_relu',...
%    'single_2048_2048_2048_2048_reluhard'};

%  'single_2048_2048_2048_2048_relu',...
% 	      'single_2048_2048_2048_2048_reluhard',...
% 	      'single_2048_2048_2048_2048_tanh'};

%'single_2048_2048_2048_2048_512_relu',...
%    'single_2048_2048_2048_2048_1024_relu',...
%    'single_2048_2048_2048_2048_1222_relu'};


%'single_2048_2048_2048_2048_128_relu',...
%    'single_2048_2048_2048_2048_256_relu',...
    


%       


% model_list = {'single_2048_2048_512_tanh',...
%     'single_2048_2048_512_relu',...
%     'single_2048_2048_512_reluhard',...    
%     'single_1024_1024_1024_1024_reluhard',...
%     'single_2048_2048_relu',...
%     'single_2048_2048_2048_relu',...
%     'single_2048_2048_tanh',...
%     'single_4096_2048_512_relu',...
%     'single_1024_1024_1024_1024_relu',...
%     'single_1024_1024_1024_1024_tanh',...
%     'single_1024_1024_1024_1024_512_relu'};
%% load data to use for dev set. assuming file 32 witheld from training
eI = [];
eI.datDir = '/scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/swbd/s5/exp/nn_data_full_fmllr/';
eI.inputDim = 300;
[feat, utt_dat, label_ind] = load_kaldi_data(eI.datDir,32, eI.inputDim);
feat = feat(1:10000,:)';
label_ind = label_ind(1:10000);
% feat = feat(1:25000,:)';
% label_ind = label_ind(1:25000);
%% load model
for m = 1:numel(model_list)    
    fprintf('%s\t',model_list{m});

    for ep = 1 : -1 : 0
      cur_model = [model_base, model_list{m}, sprintf('/spNet_restart_e%d.mat',ep)];
      if exist(cur_model,'file')
        break
    end;end;
    
    load(cur_model);
    %% report training set performance
    %fprintf('ep: %d iter: %d  ce: %f  acc: %f\n', ep, iter,...
	%    mean(ceValHist((end-10000):end)), mean(accHist((end-10000):end)));
    fprintf('%d\t%d\t%f\t%f\t', ep, iter,...
	    mean(ceValHist((end-10000):end)), mean(accHist((end-10000):end)));

    figure(1);xInd = (1:numel(accHist)); plot(xInd, ceValHist,'kx'); hold on; plot(xInd, wValHist,'rx'); plot(xInd, accHist,'bx'); title(eI.outputDir); hold off;
    drawnow; 
    %% eval on dev set
    eI.useGpu = 0;
    % eI.activationFn = 'relu-hard';
    [f, ~, nc, ne, ceCost, wCost, hAct] = spNetCostSlave(theta, eI, feat, label_ind, true);
    %fprintf('dev set.  ce: %f  acc: %f\n', ceCost, nc/ne);    
    fprintf('%f\t%f\n', ceCost, nc/ne);    
    
    %pause;
end;
