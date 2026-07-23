function contextCsv = prepare_production_replay_batch( ...
        selectionCsv, outputRoot, dataRoot, predictCodeRoot, mrstRoot, ...
        verifyToleranceLog10)
%PREPARE_PRODUCTION_REPLAY_BATCH Replay an arbitrary production task batch.
%
% CONTEXTCSV = PREPARE_PRODUCTION_REPLAY_BATCH(SELECTIONCSV, OUTPUTROOT,
% DATAROOT, PREDICTCODEROOT, MRSTROOT, TOLERANCE) replays every unique
% PREDICT realization in a checkpoint-centered selection table, enforces
% exact permeability verification, and writes the context table consumed
% by the unchanged invasion-percolation Pc driver.
%
% Fine-scale replay MAT files are intentionally written beneath OUTPUTROOT.
% The Engaging checkpoint worker deletes that temporary directory only after
% compact Pc products and provenance have been validated and published.

arguments
    selectionCsv (1, 1) string
    outputRoot (1, 1) string
    dataRoot (1, 1) string
    predictCodeRoot (1, 1) string
    mrstRoot (1, 1) string
    verifyToleranceLog10 (1, 1) double {mustBePositive} = 1.0e-3
end

selectionCsv = absolutePath(selectionCsv);
outputRoot = absolutePath(outputRoot);
dataRoot = absolutePath(dataRoot);
predictCodeRoot = absolutePath(predictCodeRoot);
mrstRoot = absolutePath(mrstRoot);

assert(exist(selectionCsv, 'file') == 2, ...
    'ProductionReplay:MissingSelection', ...
    'Missing checkpoint selection CSV: %s', selectionCsv);
assert(exist(dataRoot, 'dir') == 7, ...
    'ProductionReplay:MissingDataRoot', ...
    'Missing frozen PREDICT input root: %s', dataRoot);
assert(exist(predictCodeRoot, 'dir') == 7, ...
    'ProductionReplay:MissingCodeRoot', ...
    'Missing frozen PREDICT code root: %s', predictCodeRoot);

thisDir = fileparts(mfilename('fullpath'));
examplesDir = fileparts(thisDir);
addpath(examplesDir, '-begin');
frozenExamplesDir = fullfile(predictCodeRoot, 'examples');
assert(exist(frozenExamplesDir, 'dir') == 7, ...
    'ProductionReplay:MissingFrozenExamples', ...
    'Frozen PREDICT examples folder not found: %s', frozenExamplesDir);
% Resolve the replay implementation from the frozen physics commit while
% retaining this production-only orchestration function from the live repo.
addpath(frozenExamplesDir, '-begin');
setupMrst(mrstRoot);

selection = readtable(selectionCsv, 'TextType', 'string');
required = {'task_id', 'task_key_sha256', 'geology_id', ...
    'scenario_index', 'scenario_label', 'scenario_name', 'case_index', ...
    'case_label', 'faulting_depth_m', 'sand_vcl', 'clay_vcl', ...
    'case_id', 'case_name', 'case_category', 'window', 'slice_index', ...
    'assigned_state', 'sampling_mode', 'sampling_pool', ...
    'selected_sample_index', 'exact_replay_seed', ...
    'source_checkpoint_file', 'source_seed_base', ...
    'source_num_attempts', 'source_num_rejected', ...
    'log_kxx', 'log_kyy', 'log_kzz'};
requireColumns(selection, required, 'checkpoint selection');
assert(height(selection) > 0, ...
    'ProductionReplay:EmptySelection', 'Checkpoint selection is empty.');
assert(numel(unique(selection.task_id)) == height(selection), ...
    'ProductionReplay:DuplicateTask', ...
    'Checkpoint selection contains duplicate task IDs.');
assert(numel(unique(selection.geology_id)) == 1, ...
    'ProductionReplay:MixedGeology', ...
    'One checkpoint batch must contain exactly one geology.');
assert(numel(unique(lower(selection.window))) == 1, ...
    'ProductionReplay:MixedWindow', ...
    'One checkpoint batch must contain exactly one window.');
assert(numel(unique(str2double(selection.case_id))) == 1, ...
    'ProductionReplay:MixedLevel3Case', ...
    'One production replay batch must contain exactly one Level-3 case ID.');

replayDir = fullfile(outputRoot, 'replay_unique');
tableDir = fullfile(outputRoot, 'tables');
ensureFolder(replayDir);
ensureFolder(tableDir);

fprintf('\n=== Replay checkpoint-centered production batch ===\n');
fprintf('Selection: %s\n', selectionCsv);
fprintf('Tasks: %d | geology: %s | window: %s\n', ...
    height(selection), selection.geology_id(1), selection.window(1));

summary = replay_selected_predict_realizations( ...
    selectionCsv, replayDir, ...
    'DataRoot', dataRoot, ...
    'PredictCodeRoot', predictCodeRoot, ...
    'SmearOverlapRule', 'cell_union_psmear', ...
    'CollapseAdjacentLithology', true, ...
    'VerifyToleranceLog10', verifyToleranceLog10, ...
    'RequireDirectReplay', true);

assert(height(summary) == height(selection), ...
    'ProductionReplay:Incomplete', ...
    'Expected %d replay rows, found %d.', ...
    height(selection), height(summary));
requireColumns(summary, ...
    {'SourceRow', 'VerificationStatus', 'MaxAbsLog10Diff', 'OutputFile'}, ...
    'replay summary');
sourceRows = numericColumn(summary.SourceRow);
assert(isequal(sort(sourceRows), (1:height(selection))'), ...
    'ProductionReplay:SourceRows', ...
    'Replay SourceRow values do not cover the selection exactly once.');
selection = selection(sourceRows, :);

status = string(summary.VerificationStatus);
maxDiff = numericColumn(summary.MaxAbsLog10Diff);
bad = status ~= "matched" | ~isfinite(maxDiff) | ...
    maxDiff > verifyToleranceLog10;
if any(bad)
    error('ProductionReplay:VerificationFailed', ...
        ['%d of %d checkpoint replay tasks failed exact permeability ', ...
         'verification (tolerance %.3g log10 units).'], ...
        sum(bad), height(summary), verifyToleranceLog10);
end

summary.TaskId = selection.task_id;
summary.TaskKeySha256 = selection.task_key_sha256;
summary.GeologyId = selection.geology_id;
summary.ScenarioIndex = numericColumn(selection.scenario_index);
summary.ScenarioLabel = selection.scenario_label;
summary.ScenarioName = selection.scenario_name;
summary.CaseIndex = numericColumn(selection.case_index);
summary.CaseLabel = selection.case_label;
summary.FaultingDepthM = numericColumn(selection.faulting_depth_m);
summary.SandVcl = numericColumn(selection.sand_vcl);
summary.ClayVcl = numericColumn(selection.clay_vcl);
summary.Level3CaseId = numericColumn(selection.case_id);
summary.Level3CaseName = selection.case_name;
summary.CaseCategory = selection.case_category;
summary.Window = selection.window;
summary.SliceIndex = numericColumn(selection.slice_index);
summary.AssignedState = selection.assigned_state;
summary.SamplingMode = selection.sampling_mode;
summary.SamplingPool = selection.sampling_pool;
summary.SelectedSampleIndex = numericColumn(selection.selected_sample_index);
summary.ReplaySeed = numericColumn(selection.exact_replay_seed);
summary.LogKxx = numericColumn(selection.log_kxx);
summary.LogKyy = numericColumn(selection.log_kyy);
summary.LogKzz = numericColumn(selection.log_kzz);

contextCsv = fullfile(tableDir, 'replay_summary_context.csv');
writetable(summary, contextCsv);
fprintf('Verified %d/%d replay tasks; maximum log10(k) difference %.6g.\n', ...
    height(summary), height(summary), max(maxDiff));
fprintf('Saved production replay context: %s\n', contextCsv);
end


function setupMrst(mrstRoot)
% Initialize MRST modules required by exact PREDICT replay.

startupFile = fullfile(mrstRoot, 'startup.m');
assert(exist(startupFile, 'file') == 2, ...
    'ProductionReplay:MissingMrst', ...
    'MRST startup.m not found: %s', startupFile);
run(startupFile);
mrstModule add mrst-gui coarsegrid upscaling incomp mpfa mimetic ...
    deckformat ad-core ad-props ad-blackoil
mrstVerbose off
end


function requireColumns(T, required, label)
% Require all named columns in a table.

missing = setdiff(required, T.Properties.VariableNames);
assert(isempty(missing), 'ProductionReplay:MissingColumns', ...
    '%s is missing columns: %s', label, strjoin(missing, ', '));
end


function values = numericColumn(values)
% Convert a table variable to a finite numeric column.

if ~isnumeric(values)
    values = str2double(string(values));
end
values = double(values(:));
assert(all(isfinite(values)), 'ProductionReplay:NonfiniteColumn', ...
    'Expected a finite numeric column.');
end


function pathValue = absolutePath(pathValue)
% Normalize an input path without requiring the destination to exist.

pathValue = string(java.io.File(char(pathValue)).getAbsolutePath());
end


function ensureFolder(folderPath)
% Create a folder if it does not already exist.

if exist(folderPath, 'dir') ~= 7
    mkdir(folderPath);
end
end
