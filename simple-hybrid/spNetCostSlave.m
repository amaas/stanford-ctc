function [ cost, grad, numCorrect, numExamples, ceCost, wCost] = spNetCostSlave( theta, eI, data, labels)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%global data;
%global labels;
%% setup aggregators
cost=0;
wCost = 0;
grad=[];
%% reshape into network
stack = params2stack(theta, eI);
numHidden = numel(eI.layerSizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% setup GPU vars
if (eI.useGpu)
    data = gdouble(data);
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
    else
        delta('unrecognized activation function: %s',eI.activationFn);
    end;
end
%% softmax layer
hAct{end} = exp(bsxfun(@plus, stack{end}.W * hAct{end-1}, stack{end}.b));
hAct{end} = bsxfun(@rdivide,hAct{end},sum(hAct{end}));

%% get labels as matrix
[n,m] = size(data);
groundTruth = sparse(labels, 1:m, 1,eI.layerSizes(end),m);
if eI.useGpu
    groundTruth = gdouble(full(groundTruth));
end;
%% compute accuracy
[~,pred] = max(hAct{end});
% accList = [accList; mean(pred'==curLabels)];
% numExList = [numExList; m];
numCorrect = double(sum(pred'==labels));

%% compute cost
cost = cost - sum(sum(log(hAct{end}) .* (groundTruth)));

%% compute gradient for SM layer
delta = hAct{end}-groundTruth;
gradStack{end}.W = delta*hAct{end-1}';
gradStack{end}.b = sum(delta, 2);
% prop error through SM layer
delta = stack{end}.W'*delta;
%% gradient for hidden layers
for i = numHidden:-1:1
    %% prop through activation function
    if strcmpi(eI.activationFn,'tanh')
        delta = delta .* (1 - hAct{i}.^2);
    elseif strcmpi(eI.activationFn,'logistic')
        delta = delta .* hAct{i} .* (1 - hAct{i});
    else
        delta('unrecognized activation function: %s',eI.activationFn);
    end;    
    % gradient for weight matrix
    if i > 1
        gradStack{i}.W = delta*hAct{i-1}';
    else
        % case for data as input
        gradStack{i}.W = delta*data';
    end;    
    % gradient for bias
    gradStack{i}.b = sum(delta,2);
    % prop through weights for lower layer
    if i > 1
        delta = stack{i}.W' * delta;
    end;
end;
%% add weight norm penalties for non-bias terms
numExamples = size(data,2);
% logistic layers
for i=1:numHidden+1
    % GPU and CPU versions because norm is slow on cpu
    if eI.useGpu
        wCost = wCost + norm(stack{i}.W,'fro')^2;
    else
        wCost = wCost + sum(stack{i}.W(:).^2);
    end;
    gradStack{i}.W =  gradStack{i}.W  + ...
        2 * numExamples * eI.lambda * stack{i}.W;
end
% scale wCost once it contains all weight norms
wCost = wCost * eI.lambda * numExamples;
%% reshape gradients into vector
[grad] = stack2params(gradStack);
%% compute final cost
ceCost = cost;
cost = cost + wCost;
%% return from gpu. should be no-op on CPU
grad = double(grad);
cost = double(cost);
ceCost = double(ceCost);
wCost = double(wCost);
end



