function pool = ensurePredictParallelPool(numWorkers)
% Ensure a local parallel pool is available and synchronized with the
% current MATLAB path / MRST module state.

if nargin < 1 || isempty(numWorkers)
    numWorkers = [];
end

assert(license('test', 'Distrib_Computing_Toolbox') || ...
       ~isempty(ver('parallel')), ...
       ['Parallel Computing Toolbox is required for UseParallel = true. ' ...
        'Run in serial mode or install the toolbox.'])

pool = gcp('nocreate');
if isempty(pool)
    if isempty(numWorkers)
        pool = parpool('local');
    else
        pool = parpool('local', numWorkers);
    end
elseif ~isempty(numWorkers) && pool.NumWorkers ~= numWorkers
    warning(['Existing parallel pool has %d workers; requested %d. ' ...
             'Reusing the existing pool.'], pool.NumWorkers, numWorkers)
end

currentPath = path;
spmd
    path(currentPath);
end
end
