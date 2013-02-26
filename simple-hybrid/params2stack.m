function stack = params2stack(params, eI)

% Converts a flattened parameter vector into a nice "stack" structure 
% for us to work with. This is useful when you're building multilayer
% networks.
%
% stack = params2stack(params, netconfig)
%
% params - flattened parameter vector
% eI - auxiliary variable containing 
%             the configuration of the network
%


% Map the params (a vector into a stack of weights)
depth = numel(eI.layerSizes);
stack = cell(depth,1);
prevLayerSize = eI.inputDim; % the size of the previous layer
curPos = 1;                  % mark current position in parameter vector

for d = 1:depth
    % Create layer d
    stack{d} = struct;

    hidden = eI.layerSizes(d);
    % Extract weights
    wlen = double(hidden * prevLayerSize);
    stack{d}.W = reshape(params(curPos:curPos+wlen-1), hidden, prevLayerSize);
    curPos = curPos+wlen;

    % Extract bias
    blen = hidden;
    stack{d}.b = reshape(params(curPos:curPos+blen-1), hidden, 1);
    curPos = curPos+blen;
    
    % Set previous layer size
    prevLayerSize = hidden;
end

end