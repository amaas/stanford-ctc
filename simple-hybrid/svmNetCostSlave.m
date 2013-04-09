function [ cost, grad, numCorrect, numExamples, ceCost, wCost, pred_prob] = spNetCostSlave( theta, eI, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, eI);
numHidden = numel(eI.layerSizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% setup GPU vars
if (eI.useGpu)
    data = gsingle(data);
end;
%% forward prop
% logistic layers
for i = 1 : numHidden
    if i == 1
        hAct{i} = stack{i}.W * data;
    else
        hAct{i} = stack{i}.W * hAct{i-1};
    end;
    % bias term
    hAct{i}= bsxfun(@plus, hAct{i}, stack{i}.b);      
    %% nonlinearity
    if strcmpi(eI.activationFn,'tanh')
        hAct{i} = tanh(hAct{i});
    elseif strcmpi(eI.activationFn,'logistic')
        hAct{i} = 1./(1+exp(-hAct{i}));
    elseif strcmpi(eI.activationFn,'relu')
        hAct{i} = max(hAct{i}, 0.01 .* hAct{i});
    elseif strcmpi(eI.activationFn,'relu-hard')
        hAct{i} = max(hAct{i}, 0.0);
    else
        fprintf(1,'unrecognized activation function: %s',eI.activationFn);
    end;
end
%% svm layer (just a linear predictor)
pred_prob = bsxfun(@plus, stack{end}.W * hAct{end-1}, stack{end}.b);

%% return here if only predictions desired. 
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];
  numExamples = size(data,2);
  % % HACK return hidden activations instead of output
  % pred_prob = hAct{end-1};  
  return;
end;

%% get labels as matrix
[n,m] = size(data);
% groundTruth = sparse(labels, 1:m, 1,eI.layerSizes(end),m);
% if eI.useGpu
%     groundTruth = gsingle(full(groundTruth));
% end;
%% compute accuracy
[~,pred] = max(pred_prob);
% accList = [accList; mean(pred'==curLabels)];
% numExList = [numExList; m];
numCorrect = double(sum(pred'==labels));

%% compute cost
Y = bsxfun(@(y,ypos) 2*(y==ypos)-1, labels', (1:eI.outputDim)');
margin = max(0, 1 - Y .* pred_prob);
cost = (0.5/m * sum(stack{end}.W(:).^2)) + eI.C * sum(mean(margin.^2, 2));
delta = - 2*eI.C/m * (margin .* Y);

%% compute gradient for SVM layer
gradStack{end}.W = delta*hAct{end-1}' + 1/m * stack{end}.W;
gradStack{end}.b = sum(delta, 2);
% prop error through SM layer
delta = stack{end}.W'*delta * m;
%% gradient for hidden layers
for i = numHidden:-1:1
    %% prop through activation function
    if strcmpi(eI.activationFn,'tanh')
        delta = delta .* (1 - hAct{i}.^2);
    elseif strcmpi(eI.activationFn,'logistic')
        delta = delta .* hAct{i} .* (1 - hAct{i});
    elseif strcmpi(eI.activationFn,'relu')
        % derivative only changes for places where hAct < 0
        delta(hAct{i} < 0) = 0.01 .* delta(hAct{i} < 0);
    elseif strcmpi(eI.activationFn,'relu-hard')
        % derivative only changes for places where hAct < 0
        delta(hAct{i} <= 0) = 0.0;
    else
        fprintf(1,'unrecognized activation function: %s',eI.activationFn);
    end;    
    % gradient for weight matrix
    if i > 1
        gradStack{i}.W = (1/m) * delta*hAct{i-1}';
    else
        % case for data as input
        gradStack{i}.W = (1/m) * delta*data';
    end;    
    % gradient for bias
    gradStack{i}.b = (1/m) * sum(delta,2);
    % prop through weights for lower layer
    if i > 1
        delta = stack{i}.W' * delta;
    end;
end;
%% add weight norm penalties for non-bias terms
wCost = 0;
% logistic layers
for i=1:numHidden+1
    % GPU and CPU versions because norm is slow on cpu
    if eI.useGpu
        wCost = wCost + norm(stack{i}.W,'fro')^2;
    else
        wCost = wCost + sum(stack{i}.W(:).^2);
    end;
    gradStack{i}.W =  gradStack{i}.W  + ...
        2 * eI.lambda * stack{i}.W;
end
% scale wCost once it contains all weight norms
wCost = wCost * eI.lambda;
%% reshape gradients into vector
[grad] = stack2params(gradStack);
%% compute final cost
ceCost = cost;
cost = cost + wCost;
%% return from gpu. should be no-op on CPU
grad = full(double(grad));
cost = full(double(cost));
ceCost = full(double(ceCost));
wCost = full(double(wCost));
numExamples = m;
end



