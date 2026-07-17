%RUN_CASE01_PE_BRANCH_MEDOID_TEST Build and validate the simplified Case 01 input.
%
% The rigorous full-slice reservoir-ready MAT remains unchanged. This test
% writes a separate Pe-branch-medoid artifact and review figures under the
% requested output folder.
%
% Optional environment variables:
%   PE_BRANCH_FULL_SLICE_MAT
%   PE_BRANCH_TEST_OUTPUT
%   PE_BRANCH_MIN_LOG10_GAP   default 1.0
%   PE_BRANCH_MIN_COUNT       default 2
%   PE_BRANCH_MAX_BRANCHES    default 3

clear; clc;

scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);

defaultFullSlice = fullfile('D:\codex_gom', 'UQ_workflow', ...
    'pc_kr_upscaling', 'upscaled_porosity_cases_01_03_04_07', ...
    'reservoir_ready_integrated', ...
    'reservoir_ready_s05_c012_case01.mat');
defaultOutput = fullfile('D:\codex_gom', 'UQ_workflow', ...
    'pc_kr_upscaling', 'case01_pe_branch_medoid_test');

fullSliceMat = envOrDefault('PE_BRANCH_FULL_SLICE_MAT', defaultFullSlice);
outputDir = envOrDefault('PE_BRANCH_TEST_OUTPUT', defaultOutput);
minGap = numericEnvOrDefault('PE_BRANCH_MIN_LOG10_GAP', 1.0);
minCount = numericEnvOrDefault('PE_BRANCH_MIN_COUNT', 2);
maxBranches = numericEnvOrDefault('PE_BRANCH_MAX_BRANCHES', 3);

fprintf('\n=== Build Case 01 Pe-branch-medoid reservoir inputs ===\n');
outputs = build_pe_branch_medoid_reservoir_inputs( ...
    fullSliceMat, outputDir, ...
    'MinLog10PeGap', minGap, ...
    'MinBranchCount', minCount, ...
    'MaxBranches', maxBranches);

figureDir = fullfile(outputDir, 'figures');
figures = plot_pe_branch_medoid_reduction( ...
    fullSliceMat, outputs.matFile, figureDir);

fprintf('\n=== Case 01 branch summary ===\n');
disp(outputs.branchSummary(:, {'Window', 'PeBranchId', 'SliceCount', ...
    'MinEntryPcBar', 'MedianEntryPcBar', 'MaxEntryPcBar', ...
    'MedoidSliceIndex', 'MedoidEntryPcBar', ...
    'MedoidBulkSgMax', 'MeanCurveDistanceToMedoid'}));
fprintf('\n=== Case 01 QA ===\n');
disp(outputs.qaTable);
fprintf('Pc review figure: %s\n', figures.pcPng);
fprintf('Assignment map: %s\n', figures.assignmentPng);


function value = envOrDefault(name, defaultValue)
% Return an environment value or a supplied default.

value = strtrim(string(getenv(name)));
if value == ""
    value = string(defaultValue);
end
end


function value = numericEnvOrDefault(name, defaultValue)
% Parse a finite numeric environment value or use a default.

text = strtrim(string(getenv(name)));
if text == ""
    value = defaultValue;
else
    value = str2double(text);
    assert(isfinite(value), 'PeBranchTest:InvalidEnvironmentValue', ...
        '%s must be a finite number.', name);
end
end
