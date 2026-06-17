function summaryTable = replay_selected_predict_realizations(selectionCsv, outputDir, varargin)
%REPLAY_SELECTED_PREDICT_REALIZATIONS Recreate selected PREDICT realizations.
%
% This utility reruns selected window-level PREDICT realizations from the
% Level-3 sampling tables and exports the fine-scale fields that were not
% saved by the original 2000-realization production runs.
%
% The original production runs are not modified. Each replayed realization
% is written to a new MAT file in outputDir and is verified against the
% saved upscaled permeability vector.
%
% Required inputs:
%   selectionCsv - CSV containing selected window rows. The recommended
%                  table is:
%                  level3_selected_vector_provenance_and_replay_status.csv
%   outputDir    - folder where replayed MAT files and a summary CSV go.
%
% Useful filters:
%   'RowIndices'      - row numbers in selectionCsv to replay.
%   'GeologyIds'      - e.g. {'s03_c001'}.
%   'ScenarioLabels'  - e.g. {'scenario_03_high_sand_uniform'}.
%   'CaseLabels'      - geology case labels, e.g.
%                       {'case_001_zf0050_svcl010_cvcl040'}.
%   'Level3CaseIds'   - Level-3 case IDs, e.g. 4 for fault-wide high.
%   'Windows'         - e.g. {'famp1','famp2'}.
%   'MaxRows'         - maximum number of filtered rows to replay.
%
% Saved fields:
%   replay.G              - MRST fine grid.
%   replay.CG             - coarse grid used for upscaling.
%   replay.Grid           - fine-scale porosity, permeability, vcl, units.
%   replay.SectionDetails - 2D segment material maps and properties.
%   replay.PermMD         - replayed upscaled [kxx, kyy, kzz] in mD.
%   replay.Poro           - upscaled porosity.
%   replay.Vcl            - upscaled Vcl.
%   verification          - comparison with the saved selected vector.

if nargin < 1 || strlength(string(selectionCsv)) == 0
    selectionCsv = fullfile('D:', 'codex_gom', 'UQ_workflow', ...
        'three_level_sampling_162', 'summary', ...
        'level3_selected_vector_provenance_and_replay_status.csv');
end

if nargin < 2 || strlength(string(outputDir)) == 0
    outputDir = fullfile(pwd, 'results', 'selected_predict_replay');
end

examplesDir = fileparts(mfilename('fullpath'));
defaultDataRoot = fullfile(examplesDir, 'thickness_scenario_data');

parser = inputParser;
parser.addParameter('DataRoot', defaultDataRoot, @(x) ischar(x) || isstring(x));
parser.addParameter('RowIndices', [], @(x) isempty(x) || isnumeric(x));
parser.addParameter('GeologyIds', [], @(x) isempty(x) || iscell(x) || ischar(x) || isstring(x));
parser.addParameter('ScenarioLabels', [], @(x) isempty(x) || iscell(x) || ischar(x) || isstring(x));
parser.addParameter('CaseLabels', [], @(x) isempty(x) || iscell(x) || ischar(x) || isstring(x));
parser.addParameter('Level3CaseIds', [], @(x) isempty(x) || isnumeric(x) || isstring(x) || iscell(x));
parser.addParameter('Windows', [], @(x) isempty(x) || iscell(x) || ischar(x) || isstring(x));
parser.addParameter('MaxRows', inf, @(x) isnumeric(x) && isscalar(x) && x >= 1);
parser.addParameter('CorrCoef', 0.6, @(x) isnumeric(x) && isscalar(x));
parser.addParameter('BaseSeed', 1729, @(x) isnumeric(x) && isscalar(x));
parser.addParameter('VerifyToleranceLog10', 1e-8, @(x) isnumeric(x) && isscalar(x) && x > 0);
parser.addParameter('RequireDirectReplay', false, @(x) islogical(x) && isscalar(x));
parser.parse(varargin{:});
opt = parser.Results;

setupPredictPaths();
assert(exist('mrstModule', 'file') == 2, ...
       ['MRST is not on the MATLAB path. Run startup.m in your MRST ' ...
        'folder before replaying PREDICT realizations.'])
mrstModule add mrst-gui coarsegrid upscaling incomp mpfa mimetic
mrstVerbose off

ensureFolder(outputDir);

selectionTable = readtable(selectionCsv, 'TextType', 'string');
selectionTable.SourceRow = (1:height(selectionTable))';
selectionTable = attachSourceMetadata(selectionTable, opt.DataRoot, opt.BaseSeed);
selectionTable = filterSelectionTable(selectionTable, opt);

if isfinite(opt.MaxRows) && height(selectionTable) > opt.MaxRows
    selectionTable = selectionTable(1:opt.MaxRows, :);
end

assert(height(selectionTable) > 0, 'No rows matched the requested replay filters.')

[scenarioTable, geologyCaseTable] = loadReplayDefinitionTables(opt.DataRoot);
U = makeReplayUpscalingOptions();

summaryRows = {};
for irow = 1:height(selectionTable)
    row = selectionTable(irow, :);
    fprintf('Replaying row %d/%d: %s %s %s Level3 case %s window %s\n', ...
            irow, height(selectionTable), char(row.geology_id), ...
            char(row.scenario_label), char(row.case_label), ...
            char(string(row.case_id)), char(row.window));

    scenarioWindow = getScenarioWindowRow(scenarioTable, row.scenario_label, row.window);
    caseInfo = getGeologyCaseRow(geologyCaseTable, row.case_label);
    baseWindowOpt = getWindowOptions(char(row.window));
    variedWindowOpt = applyThicknessGeologyCase(baseWindowOpt, scenarioWindow, caseInfo);
    mySect = buildFaultedSection(variedWindowOpt);

    [seed, attemptIndex, replayMode] = resolveReplaySeed(row, opt, mySect, variedWindowOpt, U);
    rng(seed, 'twister');
    [replay, ok, errMsg] = runSingleWindowPermRealizationDetailed( ...
        mySect, variedWindowOpt, opt.CorrCoef, U);

    sourcePermMD = loadSourcePermRow(row, opt.DataRoot);
    verification = makeVerification(row, replay, ok, errMsg, seed, attemptIndex, ...
                                    replayMode, opt.VerifyToleranceLog10, sourcePermMD);
    sourceRow = row;

    fileName = makeReplayFileName(row, seed);
    outputFile = fullfile(outputDir, fileName);
    save(outputFile, 'replay', 'verification', 'sourceRow', '-v7.3');

    summaryRows(end+1, :) = {row.SourceRow, row.geology_id, row.scenario_label, ...
        row.scenario_name, row.case_label, row.case_id, row.case_name, ...
        row.window, row.assigned_state, row.sampling_pool, ...
        row.selected_sample_index, seed, attemptIndex, replayMode, ...
        verification.Status, verification.MaxAbsLog10Diff, outputFile}; %#ok<AGROW>
end

summaryTable = cell2table(summaryRows, 'VariableNames', ...
    {'SourceRow', 'GeologyId', 'ScenarioLabel', 'ScenarioName', ...
     'CaseLabel', 'Level3CaseId', 'Level3CaseName', 'Window', ...
     'AssignedState', 'SamplingPool', 'SelectedSampleIndex', ...
     'ReplaySeed', 'AttemptIndex', 'ReplayMode', 'VerificationStatus', ...
     'MaxAbsLog10Diff', 'OutputFile'});

summaryCsv = fullfile(outputDir, 'replay_summary.csv');
writetable(summaryTable, summaryCsv);
fprintf('Saved replay summary: %s\n', summaryCsv);
end


function setupPredictPaths()
% Add repository folders needed by the PREDICT examples.

thisFile = mfilename('fullpath');
examplesDir = fileparts(thisFile);
repoRoot = fileparts(examplesDir);
pathsToAdd = {repoRoot, ...
              fullfile(repoRoot, 'classes'), ...
              fullfile(repoRoot, 'functions'), ...
              fullfile(repoRoot, 'utils'), ...
              fullfile(repoRoot, 'utils', 'mrst-based')};
for i = 1:numel(pathsToAdd)
    if exist(pathsToAdd{i}, 'dir')
        addpath(pathsToAdd{i});
    end
end
end


function U = makeReplayUpscalingOptions()
% Match the upscaling options used in the production thickness runs.

U.useAcceleration = 1;
U.method = 'tpfa';
U.coarseDims = [1 1 1];
U.flexible = true;
U.exportJutulInputs = false;
end


function [scenarioTable, geologyCaseTable] = loadReplayDefinitionTables(dataRoot)
% Load scenario and geology-case definitions from the production data root.

scenarioFile = fullfile(dataRoot, 'thickness_scenario_definitions.csv');
caseFile = fullfile(dataRoot, 'geology_case_definitions.csv');
assert(exist(scenarioFile, 'file') == 2, 'Missing scenario table: %s', scenarioFile)
assert(exist(caseFile, 'file') == 2, 'Missing geology case table: %s', caseFile)

scenarioTable = readtable(scenarioFile, 'TextType', 'string');
geologyCaseTable = readtable(caseFile, 'TextType', 'string');
end


function T = attachSourceMetadata(T, dataRoot, baseSeed)
% Attach production metadata so replay can recover seed and source status.

metadataFile = fullfile(dataRoot, 'tables', 'varying_thickness_geology_run_metadata.csv');
if exist(metadataFile, 'file') ~= 2
    warning('Production metadata not found: %s', metadataFile)
    return
end

M = readtable(metadataFile, 'TextType', 'string');
if ~any(strcmp(T.Properties.VariableNames, 'source_checkpoint_file'))
    T.source_checkpoint_file = strings(height(T), 1);
end
if ~any(strcmp(T.Properties.VariableNames, 'source_seed_base'))
    T.source_seed_base = NaN(height(T), 1);
end
if ~any(strcmp(T.Properties.VariableNames, 'source_num_attempts'))
    T.source_num_attempts = NaN(height(T), 1);
end
if ~any(strcmp(T.Properties.VariableNames, 'source_num_rejected'))
    T.source_num_rejected = NaN(height(T), 1);
end
T.source_seed_base = tableColumnToDouble(T, 'source_seed_base');
T.source_num_attempts = tableColumnToDouble(T, 'source_num_attempts');
T.source_num_rejected = tableColumnToDouble(T, 'source_num_rejected');

for i = 1:height(T)
    match = strcmp(string(M.ScenarioLabel), string(T.scenario_label(i))) & ...
            strcmp(string(M.Window), string(T.window(i))) & ...
            strcmp(string(M.CaseLabel), string(T.case_label(i)));
    if nnz(match) == 1
        mid = find(match, 1);
        T.source_checkpoint_file(i) = M.CheckpointFile(mid);
        T.source_seed_base(i) = M.SeedBase(mid);
        T.source_num_attempts(i) = M.NumAttempts(mid);
        T.source_num_rejected(i) = M.NumRejected(mid);
    elseif ~any(strcmp(T.Properties.VariableNames, 'source_seed_base')) || ...
           ismissing(T.source_seed_base(i))
        windowId = parseWindowId(T.window(i));
        scenarioId = tableValueAsDouble(T(i, :), 'scenario_index');
        caseIndex = tableValueAsDouble(T(i, :), 'case_index');
        T.source_seed_base(i) = makeScenarioCaseSeed( ...
            baseSeed, scenarioId, windowId, caseIndex);
        T.source_num_rejected(i) = NaN;
    end
end
end


function T = filterSelectionTable(T, opt)
% Apply user-requested row/geology/window/case filters.

keep = true(height(T), 1);
if ~isempty(opt.RowIndices)
    rowKeep = false(height(T), 1);
    ids = opt.RowIndices(:);
    ids = ids(ids >= 1 & ids <= height(T));
    rowKeep(ids) = true;
    keep = keep & rowKeep;
end
if ~isempty(opt.GeologyIds)
    keep = keep & ismember(string(T.geology_id), string(opt.GeologyIds));
end
if ~isempty(opt.ScenarioLabels)
    keep = keep & ismember(string(T.scenario_label), string(opt.ScenarioLabels));
end
if ~isempty(opt.CaseLabels)
    keep = keep & ismember(string(T.case_label), string(opt.CaseLabels));
end
if ~isempty(opt.Level3CaseIds)
    keep = keep & ismember(string(T.case_id), string(opt.Level3CaseIds));
end
if ~isempty(opt.Windows)
    keep = keep & ismember(lower(string(T.window)), lower(string(opt.Windows)));
end
T = T(keep, :);
end


function scenarioWindow = getScenarioWindowRow(scenarioTable, scenarioLabel, window)
% Return the scenario row for one selected window.

match = strcmp(string(scenarioTable.ScenarioLabel), string(scenarioLabel)) & ...
        strcmpi(string(scenarioTable.Window), string(window));
assert(nnz(match) == 1, ...
       'Expected exactly one scenario row for %s / %s.', ...
       char(scenarioLabel), char(window))
scenarioWindow = scenarioTable(match, :);
end


function caseInfo = getGeologyCaseRow(geologyCaseTable, caseLabel)
% Return one geology case row.

match = strcmp(string(geologyCaseTable.CaseLabel), string(caseLabel));
assert(nnz(match) == 1, 'Expected exactly one geology case row for %s.', char(caseLabel))
caseInfo = geologyCaseTable(match, :);
end


function variedWindowOpt = applyThicknessGeologyCase(baseWindowOpt, scenarioWindow, caseInfo)
% Apply one fixed-grid thickness pattern and one geology parameter case.

variedWindowOpt = baseWindowOpt;
variedWindowOpt.zf = [caseInfo.FaultingDepth, caseInfo.FaultingDepth];
variedWindowOpt.vcl = { ...
    patternToVcl(scenarioWindow.FWPattern, caseInfo.SandVcl, caseInfo.ClayVcl), ...
    patternToVcl(scenarioWindow.HWPattern, caseInfo.SandVcl, caseInfo.ClayVcl)};
end


function vcl = patternToVcl(pattern, sandVcl, clayVcl)
% Convert an S/C lithology pattern into layer Vcl values.

pattern = upper(char(pattern));
vcl = nan(1, numel(pattern));
vcl(pattern == 'S') = sandVcl;
vcl(pattern == 'C') = clayVcl;
assert(all(isfinite(vcl)), 'Pattern must contain only S and C values.')
end


function mySect = buildFaultedSection(windowOpt)
% Build a FaultedSection object matching the production driver.

footwall = Stratigraphy(windowOpt.thick{1}, windowOpt.vcl{1}, ...
                        'Dip', windowOpt.dip(1), ...
                        'DepthFaulting', windowOpt.zf(1), ...
                        'DepthBurial', windowOpt.zmax{1});
hangingwall = Stratigraphy(windowOpt.thick{2}, windowOpt.vcl{2}, ...
                           'Dip', windowOpt.dip(2), ...
                           'IsHW', 1, ...
                           'NumLayersFW', footwall.NumLayers, ...
                           'DepthFaulting', windowOpt.zf(2), ...
                           'DepthBurial', windowOpt.zmax{2});

if isfield(windowOpt, 'totThick')
    mySect = FaultedSection(footwall, hangingwall, windowOpt.fDip, ...
                            'maxPerm', windowOpt.maxPerm, ...
                            'totThick', windowOpt.totThick);
else
    mySect = FaultedSection(footwall, hangingwall, windowOpt.fDip, ...
                            'maxPerm', windowOpt.maxPerm);
end
mySect = mySect.getMatPropDistr();
end


function [seed, attemptIndex, replayMode] = resolveReplaySeed(row, opt, mySect, windowOpt, U)
% Determine the exact RNG seed for a selected valid realization.

selectedIndex = tableValueAsDouble(row, 'selected_sample_index');
directSeed = tableValueAsDouble(row, 'exact_replay_seed');
seedBase = tableValueAsDouble(row, 'source_seed_base');
numRejected = tableValueAsDouble(row, 'source_num_rejected');

if isfinite(directSeed) && directSeed > 0
    seed = round(directSeed);
    attemptIndex = seed - round(seedBase) + 1;
    replayMode = "direct_seed_from_selection_table";
    return
end

assert(isfinite(seedBase), 'Could not determine source seed base for replay row.')
assert(isfinite(selectedIndex), 'Could not determine selected sample index for replay row.')

if isfinite(numRejected) && numRejected == 0
    attemptIndex = round(selectedIndex);
    seed = round(seedBase) + attemptIndex - 1;
    replayMode = "direct_seed_no_rejections";
    return
end

if opt.RequireDirectReplay
    error(['Selected row requires valid-attempt remapping because the ' ...
           'source run had rejected realizations. Set RequireDirectReplay=false.'])
end

[seed, attemptIndex] = findAttemptSeedForValidIndex( ...
    mySect, windowOpt, opt.CorrCoef, U, round(seedBase), round(selectedIndex));
replayMode = "remapped_by_replaying_valid_attempts";
end


function [seed, attemptIndex] = findAttemptSeedForValidIndex(mySect, windowOpt, rho, U, seedBase, selectedIndex)
% Replay attempts until the selected valid realization index is reached.

validCount = 0;
attemptIndex = 0;
maxAttempts = selectedIndex + 1000;
while validCount < selectedIndex && attemptIndex < maxAttempts
    attemptIndex = attemptIndex + 1;
    seed = seedBase + attemptIndex - 1;
    rng(seed, 'twister');
    [replay, ok] = runSingleWindowPermRealizationDetailed(mySect, windowOpt, rho, U); %#ok<ASGLU>
    if ok
        validCount = validCount + 1;
    end
end
assert(validCount == selectedIndex, ...
       'Could not find selected valid realization %d within %d attempts.', ...
       selectedIndex, maxAttempts)
end


function [replay, ok, errMsg] = runSingleWindowPermRealizationDetailed(mySect, windowOpt, rho, U)
% Run one realization and retain fine-scale fields for export.

replay = struct();
ok = false;
errMsg = "";

try
    nSeg = getNSeg(mySect.Vcl, mySect.IsClayVcl, mySect.DepthFaulting);
    myFaultSection = Fault2D(mySect, windowOpt.fDip);
    myFault = Fault3D(myFaultSection, mySect);

    if U.flexible
        [myFault, Urun] = myFault.getSegmentationLength(U, nSeg.fcn);
    else
        myFault = myFault.getSegmentationLength(U, nSeg.fcn);
        Urun = U;
    end

    G = [];
    sectionDetails = repmat(struct( ...
        'MatProps', [], 'MatMap', [], 'Grid', [], 'Poro', [], ...
        'PermMD', [], 'Vcl', []), 1, numel(myFault.SegLen));

    for k = 1:numel(myFault.SegLen)
        myFaultSection = myFaultSection.getMaterialProperties(mySect, ...
                                                              'corrCoef', rho);
        myFaultSection.MatProps.thick = myFault.Thick;
        if isempty(G)
            G = makeFaultGrid(myFault.Thick, myFault.Disp, ...
                              myFault.Length, myFault.SegLen, Urun);
        end

        smear = Smear(mySect, myFaultSection, G, 1);
        myFaultSection = myFaultSection.placeMaterials(mySect, smear, G);

        sectionDetails(k).MatProps = myFaultSection.MatProps;
        sectionDetails(k).MatMap = myFaultSection.MatMap;
        sectionDetails(k).Grid = myFaultSection.Grid;
        sectionDetails(k).Poro = mean(myFaultSection.Grid.poro);
        sectionDetails(k).PermMD = myFaultSection.Perm ./ (milli*darcy);
        sectionDetails(k).Vcl = mean(myFaultSection.Grid.vcl);

        myFault = myFault.assignExtrudedVals(G, myFaultSection, k);
    end

    [myFault, CG, Urun] = myFault.upscaleProps(G, Urun);

    replay.PermMD = myFault.Perm ./ (milli*darcy);
    replay.Log10PermMD = log10(replay.PermMD);
    replay.Poro = myFault.Poro;
    replay.Vcl = myFault.Vcl;
    replay.Thick = myFault.Thick;
    replay.SegLen = myFault.SegLen;
    replay.Dip = myFault.Dip;
    replay.Disp = myFault.Disp;
    replay.Length = myFault.Length;
    replay.Grid = myFault.Grid;
    replay.G = G;
    replay.CG = CG;
    replay.U = Urun;
    replay.SectionDetails = sectionDetails;
    ok = all(isfinite(replay.PermMD)) && all(replay.PermMD > 0);
catch ME
    errMsg = string(ME.message);
    replay.ErrorIdentifier = string(ME.identifier);
    replay.ErrorMessage = errMsg;
end
end


function sourcePermMD = loadSourcePermRow(row, dataRoot)
% Load the exact selected source row from predict_runs.mat, if available.

sourcePermMD = [NaN NaN NaN];
selectedIndex = tableValueAsDouble(row, 'selected_sample_index');
if ~isfinite(selectedIndex)
    return
end

matFile = fullfile(dataRoot, 'data', char(row.scenario_label), ...
                   char(row.window), char(row.case_label), 'predict_runs.mat');
if exist(matFile, 'file') ~= 2
    return
end

S = load(matFile, 'perms');
idx = round(selectedIndex);
if isfield(S, 'perms') && idx >= 1 && idx <= size(S.perms, 1)
    sourcePermMD = S.perms(idx, :);
end
end


function verification = makeVerification(row, replay, ok, errMsg, seed, attemptIndex, replayMode, tol, sourcePermMD)
% Compare replayed upscaled permeability with the saved selected vector.

verification = struct();
verification.Seed = seed;
verification.AttemptIndex = attemptIndex;
verification.ReplayMode = replayMode;
verification.Ok = ok;
verification.ErrorMessage = errMsg;

savedLog = [tableValueAsDouble(row, 'log_kxx'), ...
            tableValueAsDouble(row, 'log_kyy'), ...
            tableValueAsDouble(row, 'log_kzz')];
verification.CsvLog10PermMD = savedLog;
verification.SourcePermMD = sourcePermMD;
if all(isfinite(sourcePermMD)) && all(sourcePermMD > 0)
    referenceLog = log10(sourcePermMD);
    verification.Reference = "source_predict_runs_mat";
else
    referenceLog = savedLog;
    verification.Reference = "selection_csv";
end
verification.ReferenceLog10PermMD = referenceLog;
verification.CsvMinusReferenceLog10Diff = savedLog - referenceLog;

if ok
    verification.ReplayedLog10PermMD = replay.Log10PermMD;
    verification.Log10Diff = replay.Log10PermMD - referenceLog;
    verification.MaxAbsLog10Diff = max(abs(verification.Log10Diff));
    if verification.MaxAbsLog10Diff <= tol
        verification.Status = "matched";
    else
        verification.Status = "mismatch";
    end
else
    verification.ReplayedLog10PermMD = [NaN NaN NaN];
    verification.Log10Diff = [NaN NaN NaN];
    verification.MaxAbsLog10Diff = NaN;
    verification.Status = "replay_failed";
end
end


function value = tableValueAsDouble(row, name)
% Robustly read one scalar from a one-row table.

value = NaN;
if ~any(strcmp(row.Properties.VariableNames, name))
    return
end
raw = row.(name);
if istable(raw)
    raw = raw{1, 1};
elseif iscell(raw)
    raw = raw{1};
elseif numel(raw) > 1
    raw = raw(1);
end
if ismissing(raw)
    return
end
value = str2double(string(raw));
end


function values = tableColumnToDouble(T, name)
% Convert an existing table column to a numeric vector.

raw = T.(name);
values = NaN(height(T), 1);
for i = 1:height(T)
    item = raw(i);
    if iscell(item)
        item = item{1};
    end
    if ~ismissing(item)
        values(i) = str2double(string(item));
    end
end
end


function windowId = parseWindowId(window)
% Convert famp1...famp6 into numeric window ID.

token = regexp(char(window), '\d+', 'match', 'once');
assert(~isempty(token), 'Could not parse window ID from %s.', char(window))
windowId = str2double(token);
end


function seed = makeScenarioCaseSeed(baseSeed, scenarioId, windowId, caseIndex)
% Deterministic seed formula used by the production driver.

seed = baseSeed + 100000000*scenarioId + 10000000*windowId + 100000*caseIndex;
end


function fileName = makeReplayFileName(row, seed)
% Make a readable, filesystem-safe replay file name.

parts = {sprintf('sourceRow_%05d', row.SourceRow), ...
         char(row.geology_id), char(row.window), char(row.case_label), ...
         sprintf('level3case_%02d', round(tableValueAsDouble(row, 'case_id'))), ...
         sprintf('seed_%d', seed)};
fileName = [strjoin(parts, '__') '.mat'];
fileName = regexprep(fileName, '[^\w\-.]+', '_');
end


function ensureFolder(folderPath)
% Create a folder if needed.

if ~exist(folderPath, 'dir')
    mkdir(folderPath);
end
end


function opt = getWindowOptions(window)
% Window-specific paper inputs copied from the production thickness driver.

opt.window = window;
opt.maxPerm = 175; % [mD], max perm of Amp B interval (sand layers)

switch lower(window)
    case 'famp1'
        opt.thick = {[115.6143 28.8949], [37.6113 37.6861 37.6113 31.6005]};
        opt.vcl = {[0.3 0.65], [0.2878 0.65 0.2878 0.65]};
        opt.dip = [0, -12.0136];
        opt.fDip = 41.6345;
        opt.zf = [200, 200];
        opt.zmax = {[1912 1861], [1934 1909 1884 1860]};

    case 'famp2'
        opt.thick = {[36.9255 35.8537 36.8537 36.3111], ...
                     [36.5042 36.5042 36.4314 36.5042]};
        opt.vcl = {[0.2878 0.65 0.2878 0.65], ...
                   [0.2878 0.65 0.2878 0.65]};
        opt.dip = [0, -13.8951];
        opt.fDip = 43.2508;
        opt.zf = [200, 200];
        opt.zmax = {[1837.5 1812.5 1787.5 1762.5], ...
                    [1837.5 1812.5 1787.5 1762.5]};

    case 'famp3'
        opt.thick = {[35.8537 35.8537 35.8537 35.8537], ...
                     [35.8537 35.8537 35.8537 35.8537]};
        opt.vcl = {[0.2878 0.65 0.2878 0.65], ...
                   [0.2878 0.65 0.2878 0.65]};
        opt.dip = [0, -9.7683];
        opt.fDip = 43.8;
        opt.zf = [200, 200];
        opt.zmax = {[1738.8 1713.8 1688.8 1663.8], ...
                    [1738.8 1713.8 1688.8 1663.8]};

    case 'famp4'
        opt.thick = {[35.8537 35.8537 35.8537 35.9255], ...
                     [35.8537 35.8537 35.8537 35.9255]};
        opt.vcl = {[0.2878 0.65 0.2878 0.65], ...
                   [0.2878 0.65 0.2878 0.65]};
        opt.dip = [0, -4.9456];
        opt.fDip = 44.1811;
        opt.zf = [200, 200];
        opt.zmax = {[1638.8 1613.8 1588.8 1563.8], ...
                    [1638.8 1613.8 1588.8 1563.8]};

    case 'famp5'
        opt.thick = {[35.8537 35.8537 35.8537 35.8537], ...
                     [37.4901 35.2847 35.3553 35.2847]};
        opt.vcl = {[0.2878 0.65 0.2878 0.65], ...
                   [0.2878 0.65 0.2878 0.65]};
        opt.dip = [0, -5.2221];
        opt.fDip = 45.0685;
        opt.zf = [200, 200];
        opt.zmax = {[1538.8 1513.82 1488.75 1463.99], ...
                    [1538.8 1513.82 1488.75 1463.99]};

    case 'famp6'
        opt.thick = {[28.2932 33.1042 33.1699 33.1042], 127.6715};
        opt.vcl = {[0.2878 0.65 0.2878 0.65], 0.3266};
        opt.dip = [0, -5];
        opt.fDip = 46.0685;
        opt.zf = [200, 200];
        opt.zmax = {[1440.6 1417.5 1392.5 1367.5], 1400};

    otherwise
        error(['Unsupported window "' window '". Choose one of: ' ...
               'famp1, famp2, famp3, famp4, famp5, famp6.'])
end
end
