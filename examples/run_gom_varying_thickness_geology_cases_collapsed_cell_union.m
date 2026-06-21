function results = run_gom_varying_thickness_geology_cases_collapsed_cell_union(outputDir)
%RUN_GOM_VARYING_THICKNESS_GEOLOGY_CASES_COLLAPSED_CELL_UNION
% Run all 162 thickness/geology PREDICT ensembles with the revised setup.
%
% This is the production convenience runner for the new non-destructive
% driver:
%   gom_perm_varying_thickness_geology_cases_collapsed_cell_union
%
% It uses:
%   - all six thickness scenarios,
%   - all 27 geology cases,
%   - all six throw windows,
%   - Nsim = 2000 valid PREDICT realizations per scenario/window/case,
%   - adjacent same-lithology layer collapse,
%   - SmearOverlapRule = 'cell_union_psmear',
%   - resumable checkpoints,
%   - parallel execution with 16 workers.
%
% Before running, start MRST in MATLAB, for example:
%   run('C:\path\to\mrst\startup.m')
%
% Usage:
%   run_gom_varying_thickness_geology_cases_collapsed_cell_union
%   run_gom_varying_thickness_geology_cases_collapsed_cell_union( ...
%       'D:\my_output_folder')

if nargin < 1 || isempty(outputDir)
    thisDir = fileparts(mfilename('fullpath'));
    outputDir = fullfile(thisDir, ...
        'thickness_scenario_data_collapsed_cell_union');
end

assert(exist('mrstModule', 'file') == 2, ...
    ['MRST is not on the MATLAB path. Run startup.m in your MRST folder ' ...
     'before calling this runner.'])

results = gom_perm_varying_thickness_geology_cases_collapsed_cell_union( ...
    outputDir, ...
    'Nsim', 2000, ...
    'UseParallel', true, ...
    'NumWorkers', 16, ...
    'Resume', true, ...
    'ShowProgress', true, ...
    'BatchSize', 200, ...
    'SmearOverlapRule', 'cell_union_psmear');
end
