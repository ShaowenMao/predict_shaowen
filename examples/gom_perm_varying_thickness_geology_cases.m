function results = gom_perm_varying_thickness_geology_cases(outputDir, varargin)
% Generate PREDICT permeability datasets for six fixed-grid thickness cases.
%
% This driver extends gom_perm_varying_geology_cases by adding an outer
% loop over fixed-grid sand/clay pattern scenarios. The computational layer
% thicknesses are not changed. Instead, each scenario reassigns each
% existing layer as sand (S) or clay (C). Adjacent layers with the same
% lithology represent a thicker effective sand or clay package.
%
% Default experiment:
%   6 thickness/lithology scenarios
%   x 3 faulting depths
%   x 3 sand Vcl values
%   x 3 clay Vcl values
%   x 6 throw windows
%
% Usage:
%   results = gom_perm_varying_thickness_geology_cases()
%   results = gom_perm_varying_thickness_geology_cases('D:\codex_gom\thickness_cases')
%
% Name-value options are the same as gom_perm_varying_geology_cases, plus:
%   'ThicknessScenarios'     - scenario indices or labels to run.
%                              Default: [] means all six default scenarios.
%   'ThicknessScenarioFile'  - optional CSV with columns:
%                              ScenarioLabel, Window, FWPattern, HWPattern.
%                              Default: '' uses the six hard-coded designs.
%
% Output structure:
%   outputDir/
%     geology_case_definitions.csv
%     thickness_scenario_definitions.csv
%     varying_thickness_geology_case_definitions.mat
%     tables/
%       varying_thickness_geology_run_metadata.csv
%     data/<scenarioLabel>/<window>/<caseLabel>/
%       predict_runs.mat
%       predict_runs.progress.mat  (transient while incomplete)

if nargin < 1 || isempty(outputDir)
    outputDir = fullfile(pwd, 'gom_varying_thickness_geology_cases');
end

setupPredictPaths();

parser = inputParser;
parser.addParameter('Windows', {'famp1', 'famp2', 'famp3', 'famp4', 'famp5', 'famp6'}, ...
                    @(x) iscell(x) || isstring(x));
parser.addParameter('Nsim', 2000, @(x) isnumeric(x) && isscalar(x) && x >= 1);
parser.addParameter('FaultingDepths', [50 500 1000], ...
                    @(x) isnumeric(x) && isvector(x) && all(x > 0));
parser.addParameter('SandVclValues', [0.1 0.2 0.3], ...
                    @(x) isnumeric(x) && isvector(x) && all(x >= 0) && all(x <= 1));
parser.addParameter('ClayVclValues', [0.4 0.5 0.6], ...
                    @(x) isnumeric(x) && isvector(x) && all(x >= 0) && all(x <= 1));
parser.addParameter('IsClayVcl', 0.4, @(x) isnumeric(x) && isscalar(x) && x >= 0 && x <= 1);
parser.addParameter('CorrCoef', 0.6, @(x) isnumeric(x) && isscalar(x));
parser.addParameter('BaseSeed', 1729, @(x) isnumeric(x) && isscalar(x));
parser.addParameter('UseParallel', false, @(x) islogical(x) && isscalar(x));
parser.addParameter('NumWorkers', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 1));
parser.addParameter('Resume', true, @(x) islogical(x) && isscalar(x));
parser.addParameter('ShowProgress', true, @(x) islogical(x) && isscalar(x));
parser.addParameter('BatchSize', 200, @(x) isnumeric(x) && isscalar(x) && x >= 1);
parser.addParameter('ThicknessScenarios', [], ...
                    @(x) isempty(x) || isnumeric(x) || iscell(x) || ischar(x) || isstring(x));
parser.addParameter('ThicknessScenarioFile', '', @(x) ischar(x) || isstring(x));
parser.parse(varargin{:});
opt = parser.Results;

windows = cellstr(string(opt.Windows));
faultingDepths = sortUniqueNumericVector(opt.FaultingDepths(:)');
sandVclValues = sortUniqueNumericVector(opt.SandVclValues(:)');
clayVclValues = sortUniqueNumericVector(opt.ClayVclValues(:)');

assert(all(sandVclValues < opt.IsClayVcl), ...
       'All SandVclValues must be smaller than IsClayVcl.')
assert(all(clayVclValues >= opt.IsClayVcl), ...
       'All ClayVclValues must be at least IsClayVcl.')

assert(exist('mrstModule', 'file') == 2, ...
       ['MRST is not on the MATLAB path. Run startup.m in your MRST ' ...
        'folder before calling gom_perm_varying_thickness_geology_cases.'])
mrstModule add mrst-gui coarsegrid upscaling incomp mpfa mimetic
mrstVerbose off
if opt.UseParallel
    ensurePredictParallelPool(opt.NumWorkers);
end

U.useAcceleration = 1;
U.method = 'tpfa';
U.coarseDims = [1 1 1];
U.flexible = true;
U.exportJutulInputs = false;

dataDir = fullfile(outputDir, 'data');
tableDir = fullfile(outputDir, 'tables');
ensureFolder(outputDir);
ensureFolder(dataDir);
ensureFolder(tableDir);

geologyCaseTable = buildGeologyCaseDefinitionTable(faultingDepths, sandVclValues, clayVclValues);
thicknessScenarioTable = loadThicknessScenarioTable(opt.ThicknessScenarioFile);
thicknessScenarioTable = filterThicknessScenarios(thicknessScenarioTable, opt.ThicknessScenarios);
validateThicknessScenarioTable(thicknessScenarioTable, windows);

writetable(geologyCaseTable, fullfile(outputDir, 'geology_case_definitions.csv'));
writetable(thicknessScenarioTable, fullfile(outputDir, 'thickness_scenario_definitions.csv'));
save(fullfile(outputDir, 'varying_thickness_geology_case_definitions.mat'), ...
     'geologyCaseTable', 'thicknessScenarioTable', 'windows', 'opt', '-v7.3');

runMetaRows = {};
metadataCsv = fullfile(tableDir, 'varying_thickness_geology_run_metadata.csv');
scenarioIds = unique(thicknessScenarioTable.ScenarioIndex, 'stable');

for iscen = 1:numel(scenarioIds)
    scenarioId = scenarioIds(iscen);
    scenarioRows = thicknessScenarioTable(thicknessScenarioTable.ScenarioIndex == scenarioId, :);
    scenarioLabel = char(scenarioRows.ScenarioLabel(1));
    scenarioName = char(scenarioRows.ScenarioName(1));

    if opt.ShowProgress
        fprintf('\n=== Thickness scenario %s: %s ===\n', scenarioLabel, scenarioName);
    end

    for iw = 1:numel(windows)
        window = windows{iw};
        baseWindowOpt = getWindowOptions(window);
        scenarioWindow = getScenarioWindowRow(scenarioRows, window);

        if opt.ShowProgress
            fprintf('\n--- Window %s ---\n', window);
        end

        for icase = 1:height(geologyCaseTable)
            caseInfo = geologyCaseTable(icase, :);
            caseLabel = char(caseInfo.CaseLabel);
            caseDir = fullfile(dataDir, scenarioLabel, window, caseLabel);
            ensureFolder(caseDir);

            variedWindowOpt = applyThicknessGeologyCase(baseWindowOpt, scenarioWindow, caseInfo);
            mySect = buildFaultedSection(variedWindowOpt);

            seedBase = makeScenarioCaseSeed(opt.BaseSeed, scenarioId, iw, caseInfo.CaseIndex);
            checkpointFile = fullfile(caseDir, 'predict_runs.mat');
            progressFile = fullfile(caseDir, 'predict_runs.progress.mat');
            expected = struct( ...
                'Kind', 'VariedThicknessGeologyRun', ...
                'ScenarioIndex', scenarioId, ...
                'ScenarioLabel', scenarioLabel, ...
                'ScenarioName', scenarioName, ...
                'Window', window, ...
                'FWPattern', char(scenarioWindow.FWPattern), ...
                'HWPattern', char(scenarioWindow.HWPattern), ...
                'CaseIndex', caseInfo.CaseIndex, ...
                'CaseLabel', caseLabel, ...
                'TargetN', opt.Nsim, ...
                'SeedBase', seedBase, ...
                'CorrCoef', opt.CorrCoef, ...
                'FaultingDepth', caseInfo.FaultingDepth, ...
                'SandVcl', caseInfo.SandVcl, ...
                'ClayVcl', caseInfo.ClayVcl);

            loaded = false;
            if opt.Resume
                [~, meta, loaded] = loadFinalRunCheckpoint(checkpointFile, expected);
            end

            if loaded
                if exist(progressFile, 'file')
                    delete(progressFile);
                end
                if opt.ShowProgress
                    fprintf('  %s %s %s: resumed completed checkpoint\n', ...
                            scenarioLabel, window, caseLabel);
                end
            else
                label = sprintf('%s %s %s', scenarioLabel, window, caseLabel);
                [perms, meta] = runWindowPermSamplesResumable( ...
                    mySect, variedWindowOpt, opt.Nsim, opt.CorrCoef, U, ...
                    opt.UseParallel, seedBase, opt.ShowProgress, label, ...
                    progressFile, expected, opt.Resume, opt.BatchSize);
                saveFinalRunCheckpoint(checkpointFile, perms, meta, expected);
                if opt.ShowProgress
                    fprintf('  %s %s %s: saved %s\n', ...
                            scenarioLabel, window, caseLabel, checkpointFile);
                end
            end

            runMetaRows(end+1, :) = {scenarioId, scenarioLabel, scenarioName, ...
                                     window, char(scenarioWindow.FWPattern), ...
                                     char(scenarioWindow.HWPattern), ...
                                     caseInfo.CaseIndex, caseLabel, ...
                                     caseInfo.FaultingDepth, caseInfo.SandVcl, ...
                                     caseInfo.ClayVcl, opt.Nsim, meta.NumAttempts, ...
                                     meta.NumRejected, meta.AcceptanceRatio, ...
                                     meta.SeedBase, checkpointFile}; %#ok<AGROW>

            runMetaTable = cell2table(runMetaRows, 'VariableNames', ...
                {'ScenarioIndex', 'ScenarioLabel', 'ScenarioName', ...
                 'Window', 'FWPattern', 'HWPattern', 'CaseIndex', ...
                 'CaseLabel', 'FaultingDepth', 'SandVcl', 'ClayVcl', ...
                 'TargetN', 'NumAttempts', 'NumRejected', ...
                 'AcceptanceRatio', 'SeedBase', 'CheckpointFile'});
            writetable(runMetaTable, metadataCsv);
        end
    end
end

results = struct();
results.outputDir = outputDir;
results.thicknessScenarios = thicknessScenarioTable;
results.geologyCases = geologyCaseTable;
results.runMetadata = runMetaTable;
results.options = opt;

save(fullfile(outputDir, 'gom_varying_thickness_geology_cases_results.mat'), ...
     'results', '-v7.3');
end


function setupPredictPaths()
% Ensure the repository folders used by the examples are on the MATLAB path.

thisFile = mfilename('fullpath');
if isempty(thisFile)
    return
end

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


function caseTable = buildGeologyCaseDefinitionTable(faultingDepths, sandVclValues, clayVclValues)
% Build the 27-case geology table.

rows = {};
caseIndex = 0;
for iz = 1:numel(faultingDepths)
    for is = 1:numel(sandVclValues)
        for ic = 1:numel(clayVclValues)
            caseIndex = caseIndex + 1;
            zf = faultingDepths(iz);
            sandVcl = sandVclValues(is);
            clayVcl = clayVclValues(ic);
            label = sprintf('case_%03d_zf%04d_svcl%03d_cvcl%03d', ...
                            caseIndex, round(zf), round(100*sandVcl), ...
                            round(100*clayVcl));
            rows(end+1, :) = {caseIndex, label, zf, sandVcl, clayVcl}; %#ok<AGROW>
        end
    end
end

caseTable = cell2table(rows, 'VariableNames', ...
    {'CaseIndex', 'CaseLabel', 'FaultingDepth', 'SandVcl', 'ClayVcl'});
end


function scenarioTable = loadThicknessScenarioTable(filePath)
% Load user-supplied scenarios or use the six proposed default designs.

if nargin < 1 || strlength(string(filePath)) == 0
    scenarioTable = buildDefaultThicknessScenarioTable();
else
    scenarioTable = readtable(filePath, 'TextType', 'string');
    scenarioTable = normalizeThicknessScenarioTable(scenarioTable);
end
end


function scenarioTable = buildDefaultThicknessScenarioTable()
% Hard-coded six fixed-grid thickness/lithology scenarios proposed here.

rows = {
    1, 'scenario_01_low_sand_uniform',       'low sand, uniform',       'famp1', 'SC',   'SCCC';
    1, 'scenario_01_low_sand_uniform',       'low sand, uniform',       'famp2', 'SCCC', 'SCCC';
    1, 'scenario_01_low_sand_uniform',       'low sand, uniform',       'famp3', 'SCCC', 'SCCC';
    1, 'scenario_01_low_sand_uniform',       'low sand, uniform',       'famp4', 'SCCC', 'SCCC';
    1, 'scenario_01_low_sand_uniform',       'low sand, uniform',       'famp5', 'SCCC', 'SCCC';
    1, 'scenario_01_low_sand_uniform',       'low sand, uniform',       'famp6', 'SCCC', 'S';
    2, 'scenario_02_medium_sand_uniform',    'medium sand, uniform',    'famp1', 'SC',   'SCSC';
    2, 'scenario_02_medium_sand_uniform',    'medium sand, uniform',    'famp2', 'SCSC', 'SCSC';
    2, 'scenario_02_medium_sand_uniform',    'medium sand, uniform',    'famp3', 'SCSC', 'SCSC';
    2, 'scenario_02_medium_sand_uniform',    'medium sand, uniform',    'famp4', 'SCSC', 'SCSC';
    2, 'scenario_02_medium_sand_uniform',    'medium sand, uniform',    'famp5', 'SCSC', 'SCSC';
    2, 'scenario_02_medium_sand_uniform',    'medium sand, uniform',    'famp6', 'SCSC', 'S';
    3, 'scenario_03_high_sand_uniform',      'high sand, uniform',      'famp1', 'SC',   'SSSC';
    3, 'scenario_03_high_sand_uniform',      'high sand, uniform',      'famp2', 'SSSC', 'SSSC';
    3, 'scenario_03_high_sand_uniform',      'high sand, uniform',      'famp3', 'SSSC', 'SSSC';
    3, 'scenario_03_high_sand_uniform',      'high sand, uniform',      'famp4', 'SSSC', 'SSSC';
    3, 'scenario_03_high_sand_uniform',      'high sand, uniform',      'famp5', 'SSSC', 'SSSC';
    3, 'scenario_03_high_sand_uniform',      'high sand, uniform',      'famp6', 'SSSC', 'S';
    4, 'scenario_04_low_sand_nonuniform',    'low sand, nonuniform',    'famp1', 'SC',   'SCCS';
    4, 'scenario_04_low_sand_nonuniform',    'low sand, nonuniform',    'famp2', 'SCCS', 'CCCC';
    4, 'scenario_04_low_sand_nonuniform',    'low sand, nonuniform',    'famp3', 'CCCC', 'SCCS';
    4, 'scenario_04_low_sand_nonuniform',    'low sand, nonuniform',    'famp4', 'SCCS', 'CCCC';
    4, 'scenario_04_low_sand_nonuniform',    'low sand, nonuniform',    'famp5', 'CCCC', 'SCCC';
    4, 'scenario_04_low_sand_nonuniform',    'low sand, nonuniform',    'famp6', 'SCCC', 'S';
    5, 'scenario_05_medium_sand_nonuniform', 'medium sand, nonuniform', 'famp1', 'SC',   'SSSC';
    5, 'scenario_05_medium_sand_nonuniform', 'medium sand, nonuniform', 'famp2', 'SSSC', 'SCCC';
    5, 'scenario_05_medium_sand_nonuniform', 'medium sand, nonuniform', 'famp3', 'SCCC', 'SSCC';
    5, 'scenario_05_medium_sand_nonuniform', 'medium sand, nonuniform', 'famp4', 'SSCC', 'SCCC';
    5, 'scenario_05_medium_sand_nonuniform', 'medium sand, nonuniform', 'famp5', 'SCCC', 'SSSC';
    5, 'scenario_05_medium_sand_nonuniform', 'medium sand, nonuniform', 'famp6', 'SSSC', 'S';
    6, 'scenario_06_high_sand_nonuniform',   'high sand, nonuniform',   'famp1', 'SC',   'SSSC';
    6, 'scenario_06_high_sand_nonuniform',   'high sand, nonuniform',   'famp2', 'SSSC', 'SSSS';
    6, 'scenario_06_high_sand_nonuniform',   'high sand, nonuniform',   'famp3', 'SSSS', 'CSSC';
    6, 'scenario_06_high_sand_nonuniform',   'high sand, nonuniform',   'famp4', 'CSSC', 'SSSS';
    6, 'scenario_06_high_sand_nonuniform',   'high sand, nonuniform',   'famp5', 'SSSS', 'CSSC';
    6, 'scenario_06_high_sand_nonuniform',   'high sand, nonuniform',   'famp6', 'CSSC', 'S'};

scenarioTable = cell2table(rows, 'VariableNames', ...
    {'ScenarioIndex', 'ScenarioLabel', 'ScenarioName', ...
     'Window', 'FWPattern', 'HWPattern'});
end


function scenarioTable = normalizeThicknessScenarioTable(scenarioTable)
% Normalize imported scenario tables to the variables used by this driver.

required = {'ScenarioLabel', 'Window', 'FWPattern', 'HWPattern'};
for i = 1:numel(required)
    assert(any(strcmpi(scenarioTable.Properties.VariableNames, required{i})), ...
           'ThicknessScenarioFile must contain a %s column.', required{i})
end

names = scenarioTable.Properties.VariableNames;
for i = 1:numel(required)
    idx = find(strcmpi(names, required{i}), 1);
    names{idx} = required{i};
end
scenarioTable.Properties.VariableNames = names;

scenarioTable.ScenarioLabel = cellstr(string(scenarioTable.ScenarioLabel));
scenarioTable.Window = cellstr(lower(string(scenarioTable.Window)));
scenarioTable.FWPattern = cellstr(upper(string(scenarioTable.FWPattern)));
scenarioTable.HWPattern = cellstr(upper(string(scenarioTable.HWPattern)));

if ~any(strcmp(scenarioTable.Properties.VariableNames, 'ScenarioIndex'))
    labels = string(scenarioTable.ScenarioLabel);
    uniqueLabels = unique(labels, 'stable');
    scenarioIndex = zeros(height(scenarioTable), 1);
    for i = 1:numel(uniqueLabels)
        scenarioIndex(labels == uniqueLabels(i)) = i;
    end
    scenarioTable.ScenarioIndex = scenarioIndex;
end

if ~any(strcmp(scenarioTable.Properties.VariableNames, 'ScenarioName'))
    scenarioTable.ScenarioName = scenarioTable.ScenarioLabel;
else
    scenarioTable.ScenarioName = cellstr(string(scenarioTable.ScenarioName));
end

scenarioTable = scenarioTable(:, {'ScenarioIndex', 'ScenarioLabel', ...
                                  'ScenarioName', 'Window', ...
                                  'FWPattern', 'HWPattern'});
end


function scenarioTable = filterThicknessScenarios(scenarioTable, requested)
% Keep only selected scenario indices or labels.

if isempty(requested)
    return
end

requested = string(requested(:));

keep = ismember(string(scenarioTable.ScenarioIndex), requested) | ...
       ismember(string(scenarioTable.ScenarioLabel), requested) | ...
       ismember(string(scenarioTable.ScenarioName), requested);
scenarioTable = scenarioTable(keep, :);
assert(~isempty(scenarioTable), 'No requested thickness scenarios were found.')
end


function validateThicknessScenarioTable(scenarioTable, windows)
% Validate pattern length, allowed symbols, and at least one clay per window.

scenarioIds = unique(scenarioTable.ScenarioIndex, 'stable');
for iscen = 1:numel(scenarioIds)
    scenarioRows = scenarioTable(scenarioTable.ScenarioIndex == scenarioIds(iscen), :);

    for iw = 1:numel(windows)
        window = windows{iw};
        baseWindowOpt = getWindowOptions(window);
        scenarioWindow = getScenarioWindowRow(scenarioRows, window);

        fwPattern = char(scenarioWindow.FWPattern);
        hwPattern = char(scenarioWindow.HWPattern);
        assert(all(ismember(fwPattern, 'SC')), ...
               'Invalid FWPattern for scenario %s window %s.', ...
               char(scenarioWindow.ScenarioLabel), window)
        assert(all(ismember(hwPattern, 'SC')), ...
               'Invalid HWPattern for scenario %s window %s.', ...
               char(scenarioWindow.ScenarioLabel), window)
        assert(numel(fwPattern) == numel(baseWindowOpt.thick{1}), ...
               'FWPattern length mismatch for scenario %s window %s.', ...
               char(scenarioWindow.ScenarioLabel), window)
        assert(numel(hwPattern) == numel(baseWindowOpt.thick{2}), ...
               'HWPattern length mismatch for scenario %s window %s.', ...
               char(scenarioWindow.ScenarioLabel), window)
        assert(contains(fwPattern, 'C') || contains(hwPattern, 'C'), ...
               'Scenario %s window %s has no clay layer on either side.', ...
               char(scenarioWindow.ScenarioLabel), window)
    end
end
end


function scenarioWindow = getScenarioWindowRow(scenarioRows, window)
% Return one scenario row for a selected throw window.

match = strcmpi(string(scenarioRows.Window), string(window));
assert(nnz(match) == 1, ...
       'Expected exactly one thickness scenario row for window %s.', window)
scenarioWindow = scenarioRows(match, :);
end


function variedWindowOpt = applyThicknessGeologyCase(baseWindowOpt, scenarioWindow, caseInfo)
% Apply one fixed-grid thickness pattern and one geology case.

variedWindowOpt = baseWindowOpt;
variedWindowOpt.zf = [caseInfo.FaultingDepth, caseInfo.FaultingDepth];
variedWindowOpt.vcl = { ...
    patternToVcl(scenarioWindow.FWPattern, caseInfo.SandVcl, caseInfo.ClayVcl), ...
    patternToVcl(scenarioWindow.HWPattern, caseInfo.SandVcl, caseInfo.ClayVcl)};
end


function vcl = patternToVcl(pattern, sandVcl, clayVcl)
% Convert an S/C pattern into layer Vcl values.

pattern = upper(char(pattern));
vcl = nan(1, numel(pattern));
vcl(pattern == 'S') = sandVcl;
vcl(pattern == 'C') = clayVcl;
assert(all(isfinite(vcl)), 'Pattern must contain only S and C values.')
end


function mySect = buildFaultedSection(windowOpt)
% Build the faulted section object once for a given window.

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


function [perms, meta] = runWindowPermSamplesResumable(mySect, windowOpt, targetN, rho, U, useParallel, seedBase, showProgress, label, progressFile, checkpointInfo, resume, batchSize)
% Generate exactly targetN valid permeability samples with resumable batches.

state = initializeRunState(targetN, seedBase);
if resume
    [state, loaded] = loadProgressCheckpoint(progressFile, checkpointInfo, targetN);
    if loaded && showProgress
        fprintf('  %s: resumed partial progress (%d / %d valid)\n', ...
                label, state.NumValid, targetN);
    end
end

while state.NumValid < targetN
    remaining = targetN - state.NumValid;
    currentBatchN = min(batchSize, remaining);
    batchSeedBase = seedBase + state.NumAttempts;
    batchPerms = runWindowPermSampleBatch(mySect, windowOpt, currentBatchN, rho, U, ...
                                          useParallel, batchSeedBase);
    [validPerms, rejectedThisBatch] = sanitizeBatchPerms(batchPerms);
    takeCount = min(remaining, size(validPerms, 1));
    if takeCount > 0
        insertIds = state.NumValid + (1:takeCount);
        state.Perms(insertIds, :) = validPerms(1:takeCount, :);
        state.NumValid = state.NumValid + takeCount;
    end

    state.NumAttempts = state.NumAttempts + currentBatchN;
    state.NumRejected = state.NumRejected + rejectedThisBatch;
    state.BatchId = state.BatchId + 1;
    saveProgressCheckpoint(progressFile, state, checkpointInfo);

    if showProgress
        fprintf('  %s: batch %d, valid %d / %d, attempts %d, rejected %d\n', ...
                label, state.BatchId, state.NumValid, targetN, ...
                state.NumAttempts, state.NumRejected);
    end

    if state.NumAttempts > 20*targetN
        error(['Too many invalid permeability realizations while building ' ...
               '%s. Aborting after %d attempts.'], label, state.NumAttempts)
    end
end

perms = state.Perms(1:targetN, :);
meta = struct();
meta.Label = label;
meta.TargetN = targetN;
meta.NumReturned = size(perms, 1);
meta.NumAttempts = state.NumAttempts;
meta.NumRejected = state.NumRejected;
meta.AcceptanceRatio = targetN / state.NumAttempts;
meta.SeedBase = seedBase;
meta.CompletedOn = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));

if exist(progressFile, 'file')
    delete(progressFile);
end
end


function state = initializeRunState(targetN, seedBase)
% Preallocate one resumable run state.

state = struct();
state.Perms = nan(targetN, 3);
state.NumValid = 0;
state.NumAttempts = 0;
state.NumRejected = 0;
state.BatchId = 0;
state.SeedBase = seedBase;
end


function perms = runWindowPermSampleBatch(mySect, windowOpt, Nsim, rho, U, useParallel, seedBase)
% Generate one batch of permeability realizations.

perms = nan(Nsim, 3);
if useParallel
    parfor n = 1:Nsim
        rng(seedBase + n - 1, 'twister');
        perms(n, :) = runSingleWindowPermRealization(mySect, windowOpt, rho, U);
    end
else
    for n = 1:Nsim
        rng(seedBase + n - 1, 'twister');
        perms(n, :) = runSingleWindowPermRealization(mySect, windowOpt, rho, U);
    end
end
end


function perm = runSingleWindowPermRealization(mySect, windowOpt, rho, U)
% Run one 3D realization and return only the upscaled permeability in mD.

perm = nan(1, 3);

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
        myFault = myFault.assignExtrudedVals(G, myFaultSection, k);
    end

    [myFault, ~] = myFault.upscaleProps(G, Urun);
    perm = myFault.Perm ./ (milli*darcy);
catch
    % Rare geometry/material placement failures are treated as rejected
    % realizations so resumable batch generation can continue.
    perm(:) = NaN;
end
end


function [validPerms, numRejected] = sanitizeBatchPerms(perms)
% Keep only finite positive permeability realizations.

mask = all(perms > 0, 2) & all(isfinite(perms), 2);
validPerms = perms(mask, :);
numRejected = sum(~mask);
end


function [perms, meta, loaded] = loadFinalRunCheckpoint(filePath, expected)
% Load a completed case checkpoint if it matches the current request.

perms = [];
meta = struct();
loaded = false;

if ~exist(filePath, 'file')
    return
end

try
    S = load(filePath, 'perms', 'meta', 'checkpointInfo');
catch ME
    warning('Could not load checkpoint %s (%s). Regenerating it.', filePath, ME.message)
    return
end

if ~isfield(S, 'perms') || ~isfield(S, 'meta') || ~isfield(S, 'checkpointInfo')
    warning('Checkpoint %s is missing required variables. Regenerating it.', filePath)
    return
end

if ~isRunCheckpointCompatible(S.checkpointInfo, expected)
    return
end

if size(S.perms, 1) ~= expected.TargetN || size(S.perms, 2) ~= 3
    warning('Checkpoint %s has an unexpected permeability array size. Regenerating it.', filePath)
    return
end

if any(~isfinite(S.perms(:))) || any(S.perms(:) <= 0)
    warning('Checkpoint %s contains invalid permeability values. Regenerating it.', filePath)
    return
end

perms = S.perms;
meta = S.meta;
loaded = true;
end


function [state, loaded] = loadProgressCheckpoint(filePath, expected, targetN)
% Load a partial progress checkpoint if it matches the current request.

state = initializeRunState(targetN, expected.SeedBase);
loaded = false;

if ~exist(filePath, 'file')
    return
end

try
    S = load(filePath, 'state', 'checkpointInfo');
catch ME
    warning('Could not load progress checkpoint %s (%s). Restarting that case.', filePath, ME.message)
    return
end

if ~isfield(S, 'state') || ~isfield(S, 'checkpointInfo')
    warning('Progress checkpoint %s is missing required variables. Restarting that case.', filePath)
    return
end

if ~isRunCheckpointCompatible(S.checkpointInfo, expected)
    return
end

requiredFields = {'Perms', 'NumValid', 'NumAttempts', 'NumRejected', 'BatchId', 'SeedBase'};
for i = 1:numel(requiredFields)
    if ~isfield(S.state, requiredFields{i})
        warning('Progress checkpoint %s is incomplete. Restarting that case.', filePath)
        return
    end
end

if ~isequal(size(S.state.Perms), [targetN, 3])
    warning('Progress checkpoint %s has an unexpected size. Restarting that case.', filePath)
    return
end

numValid = S.state.NumValid;
if numValid < 0 || numValid > targetN
    warning('Progress checkpoint %s has an invalid NumValid. Restarting that case.', filePath)
    return
end

if S.state.NumAttempts > 20*targetN && numValid < targetN
    warning(['Progress checkpoint %s already exceeded the invalid-attempt ' ...
             'limit. Restarting that case.'], filePath)
    return
end

if numValid > 0
    validBlock = S.state.Perms(1:numValid, :);
    if any(~isfinite(validBlock(:))) || any(validBlock(:) <= 0)
        warning('Progress checkpoint %s contains invalid stored permeability values. Restarting that case.', filePath)
        return
    end
end

state = S.state;
loaded = true;
end


function tf = isRunCheckpointCompatible(checkpointInfo, expected)
% Check whether a checkpoint matches the current thickness/geology request.

requiredFields = {'Kind', 'ScenarioIndex', 'ScenarioLabel', 'ScenarioName', ...
                  'Window', 'FWPattern', 'HWPattern', 'CaseIndex', ...
                  'CaseLabel', 'TargetN', 'SeedBase', 'CorrCoef', ...
                  'FaultingDepth', 'SandVcl', 'ClayVcl'};
for i = 1:numel(requiredFields)
    if ~isfield(checkpointInfo, requiredFields{i})
        tf = false;
        return
    end
end

tf = strcmpi(string(checkpointInfo.Kind), string(expected.Kind)) && ...
     isequaln(checkpointInfo.ScenarioIndex, expected.ScenarioIndex) && ...
     strcmpi(string(checkpointInfo.ScenarioLabel), string(expected.ScenarioLabel)) && ...
     strcmpi(string(checkpointInfo.ScenarioName), string(expected.ScenarioName)) && ...
     strcmpi(string(checkpointInfo.Window), string(expected.Window)) && ...
     strcmpi(string(checkpointInfo.FWPattern), string(expected.FWPattern)) && ...
     strcmpi(string(checkpointInfo.HWPattern), string(expected.HWPattern)) && ...
     isequaln(checkpointInfo.CaseIndex, expected.CaseIndex) && ...
     strcmpi(string(checkpointInfo.CaseLabel), string(expected.CaseLabel)) && ...
     isequaln(checkpointInfo.TargetN, expected.TargetN) && ...
     isequaln(checkpointInfo.SeedBase, expected.SeedBase) && ...
     isequaln(checkpointInfo.CorrCoef, expected.CorrCoef) && ...
     isequaln(checkpointInfo.FaultingDepth, expected.FaultingDepth) && ...
     isequaln(checkpointInfo.SandVcl, expected.SandVcl) && ...
     isequaln(checkpointInfo.ClayVcl, expected.ClayVcl);
end


function saveFinalRunCheckpoint(filePath, perms, meta, checkpointInfo)
% Save one completed case checkpoint atomically.

checkpointInfo.CompletedOn = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
checkpointInfo.NumReturned = size(perms, 1);

folderPath = fileparts(filePath);
ensureFolder(folderPath);
tmpFilePath = [filePath '.tmp'];
if exist(tmpFilePath, 'file')
    delete(tmpFilePath);
end

save(tmpFilePath, 'perms', 'meta', 'checkpointInfo', '-v7.3');
movefile(tmpFilePath, filePath, 'f');
end


function saveProgressCheckpoint(filePath, state, checkpointInfo)
% Save one partial progress checkpoint atomically.

checkpointInfo.SavedOn = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
checkpointInfo.NumValid = state.NumValid;
checkpointInfo.NumAttempts = state.NumAttempts;

folderPath = fileparts(filePath);
ensureFolder(folderPath);
tmpFilePath = [filePath '.tmp'];
if exist(tmpFilePath, 'file')
    delete(tmpFilePath);
end

save(tmpFilePath, 'state', 'checkpointInfo', '-v7.3');
movefile(tmpFilePath, filePath, 'f');
end


function ensureFolder(folderPath)
% Create a folder if needed.

if ~exist(folderPath, 'dir')
    mkdir(folderPath);
end
end


function seed = makeScenarioCaseSeed(baseSeed, scenarioId, windowId, caseIndex)
% Deterministic seed for each scenario/window/case run.

seed = baseSeed + 100000000*scenarioId + 10000000*windowId + 100000*caseIndex;
end


function values = sortUniqueNumericVector(values)
% Sort a numeric vector and remove repeated entries.

values = sort(values(:)).';
if isempty(values)
    return
end

keepMask = [true, diff(values) ~= 0];
values = values(keepMask);
end


function opt = getWindowOptions(window)
% Window-specific paper inputs.

opt.window = window;
opt.maxPerm = 175; % [mD], max perm of Amp B interval (sand layers)

switch lower(window)
    case 'famp1' % bottom throw window
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

    case 'famp6' % top throw window
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
