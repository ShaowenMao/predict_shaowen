function results = gom_perm_varying_geology_cases(outputDir, varargin)
% Generate PREDICT permeability datasets for a 27-case varying-geology study.
%
% This driver keeps the layer architecture of the six paper throw windows
% fixed, but varies three geology controls:
%   1. faulting depth zf            = [50, 500, 1000] m
%   2. Vcl assigned to sand layers  = [0.1, 0.2, 0.3]
%   3. Vcl assigned to clay layers  = [0.4, 0.5, 0.6]
%
% The current paper inputs are treated as the base structural case. For
% each geology combination and each selected throw window, the code:
%   - keeps the original layer thicknesses and ordering unchanged
%   - replaces Vcl values only according to whether a base layer is
%     interpreted as sand or clay using the isClayVcl threshold
%   - sets the faulting depth on both sides to the requested zf value
%   - generates exactly Nsim valid permeability realizations
%   - saves restart-safe progress checkpoints after every batch
%
% Usage:
%   results = gom_perm_varying_geology_cases()
%   results = gom_perm_varying_geology_cases('D:\codex_gom\varying_geology')
%
% Name-value options:
%   'Windows'          - throw windows to run.
%                        Default: {'famp1',...,'famp6'}
%   'Nsim'             - number of PREDICT runs per case/window.
%                        Default: 2000
%   'FaultingDepths'   - faulting depths to test [m].
%                        Default: [50 500 1000]
%   'SandVclValues'    - Vcl values applied to all sand layers.
%                        Default: [0.1 0.2 0.3]
%   'ClayVclValues'    - Vcl values applied to all clay layers.
%                        Default: [0.4 0.5 0.6]
%   'IsClayVcl'        - threshold separating sand and clay layers.
%                        Default: 0.4
%   'CorrCoef'         - copula correlation coefficient. Default: 0.6
%   'BaseSeed'         - deterministic seed base. Default: 1729
%   'UseParallel'      - run realizations with parfor. Default: false
%   'NumWorkers'       - requested pool size when auto-starting.
%                        Default: []
%   'Resume'           - resume from completed and partial checkpoints.
%                        Default: true
%   'ShowProgress'     - print progress messages. Default: true
%   'BatchSize'        - save partial progress every BatchSize attempts.
%                        Default: 200
%
% Output structure:
%   outputDir/
%     case_definitions.csv
%     varying_geology_case_definitions.mat
%     tables/
%       varying_geology_run_metadata.csv
%     data/<window>/<caseLabel>/
%       predict_runs.mat
%       predict_runs.progress.mat  (transient while incomplete)
%
% Notes:
%   - Run MRST startup.m before calling this function.
%   - The default behavior runs all 27 geology cases for all 6 windows.
%   - A completed case is skipped automatically on restart.
%   - An interrupted case resumes from the last saved partial batch.

if nargin < 1 || isempty(outputDir)
    outputDir = fullfile(pwd, 'gom_varying_geology_cases');
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
        'folder before calling gom_perm_varying_geology_cases.'])
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

caseTable = buildCaseDefinitionTable(faultingDepths, sandVclValues, clayVclValues);
writetable(caseTable, fullfile(outputDir, 'case_definitions.csv'));
save(fullfile(outputDir, 'varying_geology_case_definitions.mat'), ...
     'caseTable', 'windows', 'opt', '-v7.3');

runMetaRows = {};
metadataCsv = fullfile(tableDir, 'varying_geology_run_metadata.csv');

for iw = 1:numel(windows)
    window = windows{iw};
    baseWindowOpt = getWindowOptions(window);
    if opt.ShowProgress
        fprintf('\n=== Window %s ===\n', window);
    end

    for icase = 1:height(caseTable)
        caseInfo = caseTable(icase, :);
        caseLabel = char(caseInfo.CaseLabel);
        caseDir = fullfile(dataDir, window, caseLabel);
        ensureFolder(caseDir);

        variedWindowOpt = applyGeologyCase(baseWindowOpt, caseInfo, opt.IsClayVcl);
        mySect = buildFaultedSection(variedWindowOpt);

        seedBase = makeVariedCaseSeed(opt.BaseSeed, iw, caseInfo.CaseIndex);
        checkpointFile = fullfile(caseDir, 'predict_runs.mat');
        progressFile = fullfile(caseDir, 'predict_runs.progress.mat');
        expected = struct( ...
            'Kind', 'VariedGeologyRun', ...
            'Window', window, ...
            'CaseIndex', caseInfo.CaseIndex, ...
            'CaseLabel', caseLabel, ...
            'TargetN', opt.Nsim, ...
            'SeedBase', seedBase, ...
            'CorrCoef', opt.CorrCoef, ...
            'FaultingDepth', caseInfo.FaultingDepth, ...
            'SandVcl', caseInfo.SandVcl, ...
            'ClayVcl', caseInfo.ClayVcl);

        if opt.Resume
            [perms, meta, loaded] = loadFinalRunCheckpoint(checkpointFile, expected);
        else
            perms = [];
            meta = struct();
            loaded = false;
        end

        if loaded
            if exist(progressFile, 'file')
                delete(progressFile);
            end
            if opt.ShowProgress
                fprintf('  %s %s: resumed completed checkpoint\n', window, caseLabel);
            end
        else
            label = sprintf('%s %s', window, caseLabel);
            [perms, meta] = runWindowPermSamplesResumable( ...
                mySect, variedWindowOpt, opt.Nsim, opt.CorrCoef, U, ...
                opt.UseParallel, seedBase, opt.ShowProgress, label, ...
                progressFile, expected, opt.Resume, opt.BatchSize);
            saveFinalRunCheckpoint(checkpointFile, perms, meta, expected);
            if opt.ShowProgress
                fprintf('  %s %s: saved %s\n', window, caseLabel, checkpointFile);
            end
        end

        runMetaRows(end+1, :) = {window, caseInfo.CaseIndex, caseLabel, ...
                                 caseInfo.FaultingDepth, caseInfo.SandVcl, ...
                                 caseInfo.ClayVcl, opt.Nsim, meta.NumAttempts, ...
                                 meta.NumRejected, meta.AcceptanceRatio, ...
                                 meta.SeedBase, checkpointFile}; %#ok<AGROW>

        runMetaTable = cell2table(runMetaRows, 'VariableNames', ...
            {'Window', 'CaseIndex', 'CaseLabel', 'FaultingDepth', ...
             'SandVcl', 'ClayVcl', 'TargetN', 'NumAttempts', ...
             'NumRejected', 'AcceptanceRatio', 'SeedBase', ...
             'CheckpointFile'});
        writetable(runMetaTable, metadataCsv);
    end
end

results = struct();
results.outputDir = outputDir;
results.caseDefinitions = caseTable;
results.runMetadata = runMetaTable;
results.options = opt;

save(fullfile(outputDir, 'gom_varying_geology_cases_results.mat'), ...
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
              fullfile(repoRoot, 'utils')};
for i = 1:numel(pathsToAdd)
    if exist(pathsToAdd{i}, 'dir')
        addpath(pathsToAdd{i});
    end
end
end


function caseTable = buildCaseDefinitionTable(faultingDepths, sandVclValues, clayVclValues)
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


function variedWindowOpt = applyGeologyCase(baseWindowOpt, caseInfo, isClayVcl)
% Apply one geology case while keeping the layer structure unchanged.

variedWindowOpt = baseWindowOpt;
variedWindowOpt.zf = [caseInfo.FaultingDepth, caseInfo.FaultingDepth];
variedWindowOpt.vcl = { ...
    remapLayerVcl(baseWindowOpt.vcl{1}, caseInfo.SandVcl, caseInfo.ClayVcl, isClayVcl), ...
    remapLayerVcl(baseWindowOpt.vcl{2}, caseInfo.SandVcl, caseInfo.ClayVcl, isClayVcl)};
end


function mappedVcl = remapLayerVcl(baseVcl, sandVcl, clayVcl, isClayVcl)
% Keep the sand/clay layer structure but replace the Vcl values.

mappedVcl = baseVcl;
maskClay = baseVcl >= isClayVcl;
mappedVcl(maskClay) = clayVcl;
mappedVcl(~maskClay) = sandVcl;
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
% Check whether a checkpoint matches the current geology request.

requiredFields = {'Kind', 'Window', 'CaseIndex', 'CaseLabel', 'TargetN', ...
                  'SeedBase', 'CorrCoef', 'FaultingDepth', 'SandVcl', ...
                  'ClayVcl'};
for i = 1:numel(requiredFields)
    if ~isfield(checkpointInfo, requiredFields{i})
        tf = false;
        return
    end
end

tf = strcmpi(string(checkpointInfo.Kind), string(expected.Kind)) && ...
     strcmpi(string(checkpointInfo.Window), string(expected.Window)) && ...
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


function seed = makeVariedCaseSeed(baseSeed, windowId, caseIndex)
% Deterministic seed for each window/case run.

seed = baseSeed + 100000000*windowId + 1000000*caseIndex;
end


function values = sortUniqueNumericVector(values)
% Sort a numeric vector and remove repeated entries.

values = sort(values(:)');
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
