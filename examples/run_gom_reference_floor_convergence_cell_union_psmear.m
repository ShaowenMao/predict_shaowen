function results = run_gom_reference_floor_convergence_cell_union_psmear(outputDir)
%RUN_GOM_REFERENCE_FLOOR_CONVERGENCE_CELL_UNION_PSMEAR
%Run the rigorous GOM PREDICT convergence study with cell_union_psmear.
%
% This is the same reference-floor convergence design used for the original
% fixed-stratigraphy GOM setup:
%   - 3 independent reference ensembles
%   - 20,000 realizations per reference ensemble
%   - repeated small ensembles with N = 20 ... 2000
%   - 30 repeats per tested N
%   - all six windows, famp1 ... famp6
%
% The only intended method change is:
%   SmearOverlapRule = 'cell_union_psmear'
%
% Before running this function on the compute desktop, start MRST, for
% example:
%   run('C:\path\to\mrst\startup.m')
%
% Then from MATLAB:
%   results = run_gom_reference_floor_convergence_cell_union_psmear;
%
% Output is written to:
%   examples/gom_reference_floor_cell_union_psmear_full
%
% The output folder is intentionally separate from gom_reference_floor_full
% so the legacy convergence study is not overwritten.

thisDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(thisDir);
addpath(thisDir);
addpath(fullfile(repoRoot, 'classes'));
addpath(fullfile(repoRoot, 'functions'));
addpath(genpath(fullfile(repoRoot, 'utils')));

if nargin < 1 || isempty(outputDir)
    outputDir = fullfile(thisDir, 'gom_reference_floor_cell_union_psmear_full');
end

assert(exist('mrstModule', 'file') == 2, ...
    ['MRST is not on the MATLAB path. Run MRST startup.m before calling ' ...
     'run_gom_reference_floor_convergence_cell_union_psmear.'])

results = gom_perm_reference_floor_convergence( ...
    outputDir, ...
    'SmearOverlapRule', 'cell_union_psmear', ...
    'ReferenceNsim', 20000, ...
    'NumReferences', 3, ...
    'TestNsims', [20 50 100 200 300 400 500 750 1000 1500 2000], ...
    'NumRepeats', 30, ...
    'Windows', {'famp1','famp2','famp3','famp4','famp5','famp6'}, ...
    'UseParallel', true, ...
    'NumWorkers', 16, ...
    'Resume', true, ...
    'ShowProgress', true, ...
    'MakePlots', true);
end
