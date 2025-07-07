function plotId = selectSimId(arg, data)    
%selectSimId Selects simulation index based on a criterion.
%   arg:  'maxX', 'maxZ', 'minX', 'minZ', or 'randm'
%   data: cell array of Fault3D objects

% 1. Find valid Fault3D entries
%mask = cellfun(@(x) isa(x, 'Fault3D'), data);
mask = cellfun( @(x) ~isempty(x), data );

validIndices = find(mask);

% 2. Argument-based selection
directionalArgs = {'maxX', 'maxZ', 'minX', 'minZ'};
if any(strcmp(arg, directionalArgs))
    % Only compute uperm once for these cases
    uperm = cell2mat(cellfun(@(x) x.Perm, data(validIndices), 'UniformOutput', false));

    switch arg
        case 'maxX'
            [~, idx] = max(uperm(:, 1));
        case 'maxZ'
            [~, idx] = max(uperm(:, 3));
        case 'minX'
            [~, idx] = min(uperm(:, 1));
        case 'minZ'
            [~, idx] = min(uperm(:, 3));
    end
    plotId = validIndices(idx);

elseif strcmp(arg, 'randm')
    if isempty(validIndices)
        error('No valid Fault3D entries found.');
    end
    plotId = validIndices(randi(numel(validIndices)));

else
    error('Unknown selection argument: %s', arg);
end
end