%PREPARE_FULL87_REPLAY_MEDIAN_EXAMPLES Replay selected full-87 PREDICT rows.
%
% This script prepares the fine-scale replay inputs used by the production
% Pc and dynamic-Kr upscaling workflows. It does not compute any proxy Pc/Kr
% curves. Its only job is to:
%
%   1. read the Level-3 field-slice sampling table;
%   2. select the requested geology/case/slice/window rows;
%   3. replay the exact selected PREDICT realizations;
%   4. save a replay summary table with Level-3 context.
%
% The rigorous upscaling scripts then consume:
%
%   D:\codex_gom\UQ_workflow\full87_replay_median_examples\tables\
%   replay_summary_with_full87_context_s05_c012_cases_01_03_04_07.csv

clear; clc;

scriptDir = fileparts(mfilename('fullpath'));
examplesDir = fileparts(scriptDir);
repoRoot = fileparts(examplesDir);
addpath(examplesDir);

cfg = struct();
cfg.outputRoot = envOrDefault("FULL87_REPLAY_OUTPUT_ROOT", ...
    fullfile('D:', 'codex_gom', 'UQ_workflow', ...
    'full87_replay_median_examples'));
cfg.fieldSamplingCsv = envOrDefault("FULL87_REPLAY_FIELD_SAMPLING_CSV", ...
    fullfile('D:', 'codex_gom', 'UQ_workflow', ...
    'texas_offshore_field_sampling', ...
    'texas_field_slice_window_values.csv'));
cfg.dataRoot = envOrDefault("FULL87_REPLAY_DATA_ROOT", ...
    fullfile(examplesDir, 'thickness_scenario_data'));
cfg.replayPredictCodeRoot = envOrDefault("PREDICT_REPLAY_CODE_ROOT", ...
    defaultReplayCodeRoot(repoRoot));
cfg.mrstRoot = envOrDefault("MRST_ROOT", defaultMrstRoot());
cfg.verifyToleranceLog10 = str2double(envOrDefault( ...
    "FULL87_REPLAY_VERIFY_TOLERANCE_LOG10", "1.0e-3"));
cfg.maxRows = str2double(envOrDefault("FULL87_REPLAY_MAX_ROWS", "Inf"));
cfg.geologyId = string(envOrDefault("FULL87_REPLAY_GEOLOGY_ID", "s05_c012"));
cfg.caseIds = parseCaseIds(envOrDefault("FULL87_REPLAY_CASE_IDS", "1,3,4,7"));
cfg.windows = ["famp1", "famp2", "famp3", "famp4", "famp5", "famp6"];

cfg.inputDir = fullfile(cfg.outputRoot, 'inputs');
cfg.replayDir = fullfile(cfg.outputRoot, 'replay_unique');
cfg.tableDir = fullfile(cfg.outputRoot, 'tables');
cfg.logDir = fullfile(cfg.outputRoot, 'logs');
ensureFolder(cfg.inputDir);
ensureFolder(cfg.replayDir);
ensureFolder(cfg.tableDir);
ensureFolder(cfg.logDir);

fprintf('\n=== Build full-87 replay selection ===\n')
selectionCsv = buildFull87Selection(cfg);
selectionTable = readtable(selectionCsv, 'TextType', 'string');

fprintf('\n=== Replay exact selected PREDICT realizations ===\n')
setupMrstForReplay(cfg.mrstRoot);
replaySummaryCsv = fullfile(cfg.replayDir, 'replay_summary.csv');
if exist(replaySummaryCsv, 'file') == 2
    replaySummary = readtable(replaySummaryCsv, 'TextType', 'string');
    if replaySummaryIsReusable(replaySummary, selectionTable, cfg)
        fprintf('Using existing replay summary: %s\n', replaySummaryCsv);
    else
        fprintf('Existing replay summary is incomplete or failed verification; rerunning replay.\n');
        replaySummary = runReplay(selectionCsv, cfg);
    end
else
    replaySummary = runReplay(selectionCsv, cfg);
end

assertReplaySummaryVerified(replaySummary, selectionTable, cfg);
replaySummary = attachSelectionContext(replaySummary, selectionTable);
caseToken = caseTokenFromIds(cfg.caseIds);
contextCsv = fullfile(cfg.tableDir, sprintf( ...
    'replay_summary_with_full87_context_%s_%s.csv', ...
    cfg.geologyId, caseToken));
writetable(replaySummary, contextCsv);

fprintf('Saved replay summary with context: %s\n', contextCsv);
fprintf('\nFull-87 replay preparation complete.\n')
fprintf('Output root: %s\n', cfg.outputRoot);


function replaySummary = runReplay(selectionCsv, cfg)
% Replay selected PREDICT realizations and verify against stored log(k).

replaySummary = replay_selected_predict_realizations( ...
    selectionCsv, cfg.replayDir, ...
    'DataRoot', cfg.dataRoot, ...
    'MaxRows', cfg.maxRows, ...
    'PredictCodeRoot', cfg.replayPredictCodeRoot, ...
    'VerifyToleranceLog10', cfg.verifyToleranceLog10);
end


function tf = replaySummaryIsReusable(replaySummary, selectionTable, cfg)
% Return true when an existing replay summary is complete and verified.

expectedRows = expectedReplayRowCount(selectionTable, cfg);
if height(replaySummary) ~= expectedRows
    tf = false;
    return
end

required = {'MaxAbsLog10Diff', 'VerificationStatus'};
if ~all(ismember(required, replaySummary.Properties.VariableNames))
    tf = false;
    return
end

replayDiffs = str2double(string(replaySummary.MaxAbsLog10Diff));
numBadRows = sum(replayDiffs > cfg.verifyToleranceLog10 | ...
    ~isfinite(replayDiffs) | ...
    string(replaySummary.VerificationStatus) ~= "matched");
tf = numBadRows == 0;
end


function assertReplaySummaryVerified(replaySummary, selectionTable, cfg)
% Stop production upscaling when replay is incomplete or does not verify.

expectedRows = expectedReplayRowCount(selectionTable, cfg);
if height(replaySummary) ~= expectedRows
    error('ReplayVerification:Incomplete', ...
        'Expected %d replay rows but found %d.', ...
        expectedRows, height(replaySummary));
end

required = {'MaxAbsLog10Diff', 'VerificationStatus'};
if ~all(ismember(required, replaySummary.Properties.VariableNames))
    error('ReplayVerification:MissingColumns', ...
        'Replay summary is missing verification columns.');
end

replayDiffs = str2double(string(replaySummary.MaxAbsLog10Diff));
statuses = string(replaySummary.VerificationStatus);
bad = replayDiffs > cfg.verifyToleranceLog10 | ~isfinite(replayDiffs) | ...
      statuses ~= "matched";
if any(bad)
    finiteBadDiffs = replayDiffs(bad & isfinite(replayDiffs));
    if isempty(finiteBadDiffs)
        maxBadDiff = NaN;
    else
        maxBadDiff = max(finiteBadDiffs);
    end
    error('ReplayVerification:Mismatch', ...
        ['%d of %d replay rows failed exact-permeability verification ' ...
         '(tolerance %.3g log10 units; largest mismatch %.6g). ' ...
         'Pc/Kr upscaling was not started.'], ...
        sum(bad), height(replaySummary), cfg.verifyToleranceLog10, maxBadDiff);
end
end


function expectedRows = expectedReplayRowCount(selectionTable, cfg)
% Account for intentionally truncated smoke-test replay selections.

expectedRows = height(selectionTable);
if isfinite(cfg.maxRows)
    expectedRows = min(expectedRows, max(0, floor(cfg.maxRows)));
end
end


function selectionCsv = buildFull87Selection(cfg)
% Extract one replay row for each requested case/slice/window combination.

assert(exist(cfg.fieldSamplingCsv, 'file') == 2, ...
    'Missing field sampling table: %s', cfg.fieldSamplingCsv);

T = readtable(cfg.fieldSamplingCsv, 'TextType', 'string');
caseId = str2double(string(T.case_id));
mask = T.geology_id == cfg.geologyId & ismember(caseId, cfg.caseIds);
S = T(mask, :);
assert(height(S) > 0, 'No rows found for geology %s.', cfg.geologyId);

S.Level3ReplaySet = repmat("full87_four_examples", height(S), 1);
S.InFull87ReplaySet = true(height(S), 1);
S.case_id_numeric = str2double(string(S.case_id));
S.slice_index_numeric = str2double(string(S.slice_index));
S.window_order = windowOrder(S.window);
S = sortrows(S, {'case_id_numeric', 'slice_index_numeric', 'window_order'});

expectedRows = numel(cfg.caseIds) * 87 * numel(cfg.windows);
assert(height(S) == expectedRows, ...
    'Expected %d rows but found %d.', expectedRows, height(S));

key = strcat(S.geology_id, "|", string(S.case_id), "|", ...
    string(S.slice_index), "|", S.window, "|", ...
    string(S.selected_sample_index));
[~, keep] = unique(key, 'stable');
assert(numel(keep) == height(S), ...
    'Selection contains duplicate case/slice/window/sample rows.');

selectionCsv = fullfile(cfg.inputDir, sprintf( ...
    '%s_%s_full87_replay_rows.csv', ...
    cfg.geologyId, caseTokenFromIds(cfg.caseIds)));
writetable(S, selectionCsv);
fprintf('Prepared full-87 replay selection: %s (%d rows)\n', ...
    selectionCsv, height(S));
end


function T = attachSelectionContext(T, selectionTable)
% Attach field-sampling context to the replay summary table.

if height(T) ~= height(selectionTable)
    assert(ismember('SourceRow', T.Properties.VariableNames), ...
        ['Replay summary and selection table have different row counts, ', ...
        'and SourceRow is unavailable for subsetting.']);
    sourceRows = str2double(string(T.SourceRow));
    assert(all(isfinite(sourceRows)) && ...
        all(sourceRows >= 1) && all(sourceRows <= height(selectionTable)), ...
        'Replay summary SourceRow values do not match the selection table.');
    selectionTable = selectionTable(sourceRows, :);
end

T.GeologyId = selectionTable.geology_id;
T.ScenarioIndex = str2double(string(selectionTable.scenario_index));
T.ScenarioLabel = selectionTable.scenario_label;
T.ScenarioName = selectionTable.scenario_name;
T.CaseIndex = str2double(string(selectionTable.case_index));
T.CaseLabel = selectionTable.case_label;
T.FaultingDepthM = str2double(string(selectionTable.faulting_depth_m));
T.SandVcl = str2double(string(selectionTable.sand_vcl));
T.ClayVcl = str2double(string(selectionTable.clay_vcl));
T.Level3CaseId = str2double(string(selectionTable.case_id));
T.Level3CaseName = selectionTable.case_name;
T.CaseCategory = selectionTable.case_category;
T.CaseStrength = selectionTable.case_strength;
T.PatternName = selectionTable.pattern_name;
T.Orientation = selectionTable.orientation;
T.Window = selectionTable.window;
T.SliceIndex = str2double(string(selectionTable.slice_index));
T.DrawGroupIndex = str2double(string(selectionTable.draw_group_index));
T.AssignedState = selectionTable.assigned_state;
T.SamplingMode = selectionTable.sampling_mode;
T.SamplingPool = selectionTable.sampling_pool;
T.SelectedSampleIndex = str2double(string(selectionTable.selected_sample_index));
T.LogKxx = str2double(string(selectionTable.log_kxx));
T.LogKyy = str2double(string(selectionTable.log_kyy));
T.LogKzz = str2double(string(selectionTable.log_kzz));
end


function rootPath = defaultReplayCodeRoot(repoRoot)
% Prefer the archived replay code root when available; otherwise use repo.

archivedRoot = fullfile('D:', 'codex_gom', ...
    'predict_shaowen_replay_2647b6d');
if exist(archivedRoot, 'dir') == 7
    rootPath = archivedRoot;
else
    rootPath = repoRoot;
end
end


function rootPath = defaultMrstRoot()
% Return the default MRST root for the current platform.

if ispc
    rootPath = fullfile('C:', 'Users', 'Shaow', 'OneDrive', 'MIT', ...
        'mrst-2025a', 'SINTEF-AppliedCompSci-MRST-75749fa');
else
    rootPath = fullfile('/home', 'shaowen', 'orcd', 'pool', ...
        'predict_shaowen', 'software', 'mrst-current');
end
end


function setupMrstForReplay(mrstRoot)
% Initialize MRST because PREDICT replay uses MRST geometry utilities.

if exist('mrstModule', 'file') == 2
    return
end
startupFile = fullfile(mrstRoot, 'startup.m');
assert(exist(startupFile, 'file') == 2, ...
    'MRST startup.m not found: %s', startupFile);
run(startupFile);
try
    mrstModule add mrst-gui mimetic upscaling incomp coarsegrid deckformat ...
        ad-core ad-props ad-blackoil
catch err
    warning('Could not add all standard MRST modules for replay: %s', ...
        err.message);
end
end


function value = envOrDefault(name, defaultValue)
% Read an environment variable or return a default string/path.

raw = getenv(char(name));
if isempty(raw)
    value = defaultValue;
else
    value = raw;
end
end


function ids = parseCaseIds(textValue)
% Parse comma-separated Level-3 case ids.

parts = regexp(char(textValue), '\s*,\s*', 'split');
ids = str2double(parts);
ids = ids(isfinite(ids));
assert(~isempty(ids), 'No valid Level-3 case ids were provided.');
end


function token = caseTokenFromIds(caseIds)
% Build a stable case-token string such as cases_01_03_04_07.

parts = strings(1, numel(caseIds));
for i = 1:numel(caseIds)
    parts(i) = sprintf('%02d', caseIds(i));
end
token = "cases_" + strjoin(parts, "_");
end


function order = windowOrder(windowNames)
% Numeric sorting key for famp1 ... famp6.

text = string(windowNames);
order = nan(size(text));
for i = 1:numel(text)
    digits = regexp(char(text(i)), '\d+', 'match', 'once');
    order(i) = str2double(digits);
end
end


function ensureFolder(folderPath)
% Create a folder when needed.

if exist(folderPath, 'dir') ~= 7
    mkdir(folderPath);
end
end
