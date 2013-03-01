function [ stack ] = initialize_weights( eI )
%INITIALIZE_WEIGHTS Random weight structures for a network architecture
%   eI describes a network via the fields layerSizes, inputDim, and outputDim 
%   
%   This uses Xavier's weight initialization tricks for better backprop
%   See: X. Glorot, Y. Bengio. Understanding the difficulty of training 
%        deep feedforward neural networks. AISTATS 2010.

%% initialize hidden layers
stack = cell(1, numel(eI.layerSizes));
for l = 1 : numel(eI.layerSizes)
    if l > 1
        prevSize = eI.layerSizes(l-1);
    else
        prevSize = eI.inputDim;
    end;
    curSize = eI.layerSizes(l);
    % Xaxier's scaling factor
    s = sqrt(6) / sqrt(prevSize + curSize);
    stack{l}.W = rand(curSize, prevSize)*2*s - s;
    stack{l}.b = zeros(curSize, 1);
end
