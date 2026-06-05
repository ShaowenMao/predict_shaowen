function results = gom_w6_collapsed_sand_sensitivity(outputDir, varargin)
% Compare current fixed-grid W6 SSSC against a collapsed SC representation.
%
% This diagnostic tests whether representing consecutive footwall sand
% layers separately affects the W6 permeability distribution.
%
% Case tested:
%   window        = famp6
%   fault depth   = 1000 m
%   sand Vcl      = 0.1
%   clay Vcl      = 0.4
%
% Variants:
%   current_sssc_fixed_grid:
%       FW = S + S + S + C on the original four-layer W6 grid.
%   collapsed_sc:
%       FW = one thick S + C, where the three adjacent sand layers are
%       combined. The collapsed sand zmax is a thickness-weighted average
%       of the three original sand-layer zmax values.
%
% Example:
%   results = gom_w6_collapsed_sand_sensitivity( ...
%       'D:\codex_gom\w6_collapsed_sand_sensitivity', ...
%       'UseParallel', true, 'NumWorkers', 16);

if nargin < 1 || isempty(outputDir)
    outputDir = fullfile(pwd, 'w6_collapsed_sand_sensitivity');
end

setupPredictPaths();

parser = inputParser;
parser.addParameter('Nsim', 2000, @(x) isnumeric(x) && isscalar(x) && x >= 1);
parser.addParameter('FaultingDepth', 1000, @(x) isnumeric(x) && isscalar(x) && x > 0);
parser.addParameter('SandVcl', 0.1, @(x) isnumeric(x) && isscalar(x) && x >= 0 && x < 0.4);
parser.addParameter('ClayVcl', 0.4, @(x) isnumeric(x) && isscalar(x) && x >= 0.4 && x <= 1);
parser.addParameter('CorrCoef', 0.6, @(x) isnumeric(x) && isscalar(x));
parser.addParameter('BaseSeed', 20260605, @(x) isnumeric(x) && isscalar(x));
parser.addParameter('UseParallel', false, @(x) islogical(x) && isscalar(x));
parser.addParameter('NumWorkers', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 1));
parser.addParameter('Resume', true, @(x) islogical(x) && isscalar(x));
parser.addParameter('ShowProgress', true, @(x) islogical(x) && isscalar(x));
parser.addParameter('BatchSize', 200, @(x) isnumeric(x) && isscalar(x) && x >= 1);
parser.parse(varargin{:});
opt = parser.Results;

assert(exist('mrstModule', 'file') == 2, ...
       ['MRST is not on the MATLAB path. Run startup.m in your MRST ' ...
        'folder before calling this diagnostic.'])
mrstModule add mrst-gui coarsegrid upscaling incomp mpfa mimetic
mrstVerbose off

if opt.UseParallel
    ensurePredictParallelPool(opt.NumWorkers);
end

ensureFolder(outputDir);
ensureFolder(fullfile(outputDir, 'data'));
ensureFolder(fullfile(outputDir, 'tables'));

U.useAcceleration = 1;
U.method = 'tpfa';
U.coarseDims = [1 1 1];
U.flexible = true;
U.exportJutulInputs = false;

variants = buildW6DiagnosticVariants(opt);
summaryRows = cell(0, numel(summaryVariableNames()));
metadataRows = cell(0, numel(metadataVariableNames()));

for iv = 1:numel(variants)
    variant = variants(iv);
    variantDir = fullfile(outputDir, 'data', variant.Label);
    ensureFolder(variantDir);

    if opt.ShowProgress
        fprintf('\n=== W6 collapsed-sand diagnostic: %s ===\n', variant.Label);
        fprintf('  FW pattern: %s\n', variant.FWPattern);
        fprintf('  FW thickness: %s\n', mat2str(variant.FWThick, 5));
        fprintf('  FW zmax: %s\n', mat2str(variant.FWZmax, 5));
    end

    checkpointInfo = makeCheckpointInfo(variant, opt);
    finalFile = fullfile(variantDir, 'predict_runs.mat');
    progressFile = fullfile(variantDir, 'predict_runs.progress.mat');

    [perms, meta, loaded] = loadFinalRunCheckpoint(finalFile, checkpointInfo);
    if loaded
        if opt.ShowProgress
            fprintf('  loaded existing checkpoint: %s\n', finalFile);
        end
    else
        mySect = buildW6FaultedSection(variant, opt);
        [perms, meta] = runWindowPermSamplesResumable( ...
            mySect, variant, opt.Nsim, opt.CorrCoef, U, opt.UseParallel, ...
            checkpointInfo.SeedBase, opt.ShowProgress, variant.Label, ...
            progressFile, checkpointInfo, opt.Resume, opt.BatchSize);
        saveFinalRunCheckpoint(finalFile, perms, meta, checkpointInfo);
    end

    metadataRows(end+1, :) = metadataRow(variant, opt, meta, finalFile); %#ok<AGROW>
    summaryRows = [summaryRows; summarizeVariant(variant, perms)]; %#ok<AGROW>
end

summaryTable = cell2table(summaryRows, 'VariableNames', summaryVariableNames());
metadataTable = cell2table(metadataRows, 'VariableNames', metadataVariableNames());

writetable(summaryTable, fullfile(outputDir, 'tables', 'w6_collapsed_sand_sensitivity_summary.csv'));
writetable(metadataTable, fullfile(outputDir, 'tables', 'w6_collapsed_sand_sensitivity_metadata.csv'));

results = struct();
results.summary = summaryTable;
results.metadata = metadataTable;
results.variants = variants;
results.options = opt;
save(fullfile(outputDir, 'w6_collapsed_sand_sensitivity_results.mat'), ...
     'results', '-v7.3');

if opt.ShowProgress
    fprintf('\nSaved diagnostic outputs to %s\n', outputDir);
end
end


function variants = buildW6DiagnosticVariants(opt)
% Build current and collapsed W6 input variants.

base.window = 'famp6';
base.thickFW = [28.2932 33.1042 33.1699 33.1042];
base.thickHW = 127.6715;
base.zmaxFW = [1440.6 1417.5 1392.5 1367.5];
base.zmaxHW = 1400;
base.dipFW = 0;
base.dipHW = -5;
base.fDip = 46.0685;
base.maxPerm = 175;

current = base;
current.Label = 'current_sssc_fixed_grid';
current.Description = 'Current fixed-grid W6 nonuniform medium-sand representation.';
current.FWPattern = 'SSSC';
current.HWPattern = 'S';
current.FWThick = base.thickFW;
current.HWThick = base.thickHW;
current.FWZmax = base.zmaxFW;
current.HWZmax = base.zmaxHW;
current.FWVcl = [opt.SandVcl opt.SandVcl opt.SandVcl opt.ClayVcl];
current.HWVcl = opt.SandVcl;
current.CollapseNote = 'No collapse; three adjacent sand layers remain separate.';

collapsed = base;
collapsed.Label = 'collapsed_sc';
collapsed.Description = 'Collapsed W6 representation with adjacent footwall sands combined.';
collapsed.FWPattern = 'SC';
collapsed.HWPattern = 'S';
collapsed.FWThick = [sum(base.thickFW(1:3)), base.thickFW(4)];
collapsed.HWThick = base.thickHW;
collapsed.FWZmax = [weightedMean(base.zmaxFW(1:3), base.thickFW(1:3)), base.zmaxFW(4)];
collapsed.HWZmax = base.zmaxHW;
collapsed.FWVcl = [opt.SandVcl opt.ClayVcl];
collapsed.HWVcl = opt.SandVcl;
collapsed.CollapseNote = 'First three adjacent footwall sand layers collapsed into one sand package.';

variants = [current, collapsed];
end


function value = weightedMean(values, weights)
% Return a weighted mean using row-vector inputs.

value = sum(values(:) .* weights(:)) ./ sum(weights(:));
end


function mySect = buildW6FaultedSection(variant, opt)
% Build one W6 FaultedSection object for a diagnostic variant.

footwall = Stratigraphy(variant.FWThick, variant.FWVcl, ...
                        'Dip', variant.dipFW, ...
                        'DepthFaulting', opt.FaultingDepth, ...
                        'DepthBurial', variant.FWZmax);
hangingwall = Stratigraphy(variant.HWThick, variant.HWVcl, ...
                           'Dip', variant.dipHW, ...
                           'IsHW', 1, ...
                           'NumLayersFW', footwall.NumLayers, ...
                           'DepthFaulting', opt.FaultingDepth, ...
                           'DepthBurial', variant.HWZmax);

mySect = FaultedSection(footwall, hangingwall, variant.fDip, ...
                        'maxPerm', variant.maxPerm);
mySect = mySect.getMatPropDistr();
end


function [perms, meta] = runWindowPermSamplesResumable(mySect, variant, targetN, rho, U, useParallel, seedBase, showProgress, label, progressFile, checkpointInfo, resume, batchSize)
% Generate exactly targetN valid W6 permeability samples with checkpoints.

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
    batchPerms = runWindowPermSampleBatch(mySect, variant, currentBatchN, rho, U, ...
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


function perms = runWindowPermSampleBatch(mySect, variant, Nsim, rho, U, useParallel, seedBase)
% Generate one batch of permeability realizations.

perms = nan(Nsim, 3);
if useParallel
    parfor n = 1:Nsim
        rng(seedBase + n - 1, 'twister');
        perms(n, :) = runSingleWindowPermRealization(mySect, variant, rho, U);
    end
else
    for n = 1:Nsim
        rng(seedBase + n - 1, 'twister');
        perms(n, :) = runSingleWindowPermRealization(mySect, variant, rho, U);
    end
end
end


function perm = runSingleWindowPermRealization(mySect, variant, rho, U)
% Run one 3D realization and return upscaled permeability in mD.

perm = nan(1, 3);

try
    nSeg = getNSeg(mySect.Vcl, mySect.IsClayVcl, mySect.DepthFaulting);
    myFaultSection = Fault2D(mySect, variant.fDip);
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
    % Geometry/material placement failures are rejected and regenerated.
    perm(:) = NaN;
end
end


function [validPerms, numRejected] = sanitizeBatchPerms(perms)
% Keep only finite positive permeability realizations.

mask = all(perms > 0, 2) & all(isfinite(perms), 2);
validPerms = perms(mask, :);
numRejected = sum(~mask);
end


function rows = summarizeVariant(variant, perms)
% Summarize one permeability ensemble by component.

logPerms = log10(perms);
components = {'kxx', 'kyy', 'kzz'};
rows = cell(numel(components), numel(summaryVariableNames()));

for j = 1:numel(components)
    values = logPerms(:, j);
    qs = quantile(values, [0.01 0.05 0.25 0.5 0.75 0.95 0.99]);
    rows(j, :) = {variant.Label, variant.Description, variant.FWPattern, ...
                  variant.HWPattern, components{j}, numel(values), ...
                  mean(values), std(values), qs(1), qs(2), qs(3), ...
                  qs(4), qs(5), qs(6), qs(7), ...
                  mean(values < -3), mean(values < -1), mean(values > 0), ...
                  mean(values > 1)};
end
end


function names = summaryVariableNames()
% Column names for the summary CSV.

names = {'VariantLabel', 'Description', 'FWPattern', 'HWPattern', ...
         'Component', 'N', 'MeanLog10k', 'StdLog10k', ...
         'Q01Log10k', 'Q05Log10k', 'Q25Log10k', 'MedianLog10k', ...
         'Q75Log10k', 'Q95Log10k', 'Q99Log10k', ...
         'ProbLog10kLtMinus3', 'ProbLog10kLtMinus1', ...
         'ProbLog10kGt0', 'ProbLog10kGt1'};
end


function row = metadataRow(variant, opt, meta, finalFile)
% Build one metadata row for a completed diagnostic variant.

row = {variant.Label, variant.Description, variant.FWPattern, ...
       variant.HWPattern, mat2str(variant.FWThick, 8), ...
       mat2str(variant.HWThick, 8), mat2str(variant.FWZmax, 8), ...
       mat2str(variant.HWZmax, 8), opt.FaultingDepth, opt.SandVcl, ...
       opt.ClayVcl, opt.Nsim, meta.NumAttempts, meta.NumRejected, ...
       meta.AcceptanceRatio, meta.SeedBase, variant.CollapseNote, finalFile};
end


function names = metadataVariableNames()
% Column names for the metadata CSV.

names = {'VariantLabel', 'Description', 'FWPattern', 'HWPattern', ...
         'FWThickness', 'HWThickness', 'FWZmax', 'HWZmax', ...
         'FaultingDepth', 'SandVcl', 'ClayVcl', 'TargetN', ...
         'NumAttempts', 'NumRejected', 'AcceptanceRatio', 'SeedBase', ...
         'CollapseNote', 'CheckpointFile'};
end


function checkpointInfo = makeCheckpointInfo(variant, opt)
% Build the compatibility record used for final/progress checkpoints.

checkpointInfo = struct();
checkpointInfo.Kind = 'w6_collapsed_sand_sensitivity';
checkpointInfo.VariantLabel = variant.Label;
checkpointInfo.FWPattern = variant.FWPattern;
checkpointInfo.HWPattern = variant.HWPattern;
checkpointInfo.FWThick = variant.FWThick;
checkpointInfo.HWThick = variant.HWThick;
checkpointInfo.FWZmax = variant.FWZmax;
checkpointInfo.HWZmax = variant.HWZmax;
checkpointInfo.TargetN = opt.Nsim;
checkpointInfo.SeedBase = opt.BaseSeed + 1000000 * findVariantSeedOffset(variant.Label);
checkpointInfo.CorrCoef = opt.CorrCoef;
checkpointInfo.FaultingDepth = opt.FaultingDepth;
checkpointInfo.SandVcl = opt.SandVcl;
checkpointInfo.ClayVcl = opt.ClayVcl;
end


function offset = findVariantSeedOffset(label)
% Return deterministic seed offset for each diagnostic variant.

switch lower(char(label))
    case 'current_sssc_fixed_grid'
        offset = 1;
    case 'collapsed_sc'
        offset = 2;
    otherwise
        offset = 99;
end
end


function [perms, meta, loaded] = loadFinalRunCheckpoint(filePath, expected)
% Load a completed diagnostic checkpoint if it matches the request.

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
% Load a partial progress checkpoint if it matches the request.

state = initializeRunState(targetN, expected.SeedBase);
loaded = false;

if ~exist(filePath, 'file')
    return
end

try
    S = load(filePath, 'state', 'checkpointInfo');
catch ME
    warning('Could not load progress checkpoint %s (%s). Restarting that variant.', filePath, ME.message)
    return
end

if ~isfield(S, 'state') || ~isfield(S, 'checkpointInfo')
    warning('Progress checkpoint %s is missing required variables. Restarting that variant.', filePath)
    return
end

if ~isRunCheckpointCompatible(S.checkpointInfo, expected)
    return
end

requiredFields = {'Perms', 'NumValid', 'NumAttempts', 'NumRejected', 'BatchId', 'SeedBase'};
for i = 1:numel(requiredFields)
    if ~isfield(S.state, requiredFields{i})
        warning('Progress checkpoint %s is incomplete. Restarting that variant.', filePath)
        return
    end
end

if ~isequal(size(S.state.Perms), [targetN, 3])
    warning('Progress checkpoint %s has an unexpected size. Restarting that variant.', filePath)
    return
end

if S.state.NumValid < 0 || S.state.NumValid > targetN
    warning('Progress checkpoint %s has an invalid NumValid. Restarting that variant.', filePath)
    return
end

state = S.state;
loaded = true;
end


function tf = isRunCheckpointCompatible(checkpointInfo, expected)
% Check whether a checkpoint matches the diagnostic request.

requiredFields = {'Kind', 'VariantLabel', 'FWPattern', 'HWPattern', ...
                  'FWThick', 'HWThick', 'FWZmax', 'HWZmax', ...
                  'TargetN', 'SeedBase', 'CorrCoef', 'FaultingDepth', ...
                  'SandVcl', 'ClayVcl'};
for i = 1:numel(requiredFields)
    if ~isfield(checkpointInfo, requiredFields{i})
        tf = false;
        return
    end
end

tf = strcmpi(string(checkpointInfo.Kind), string(expected.Kind)) && ...
     strcmpi(string(checkpointInfo.VariantLabel), string(expected.VariantLabel)) && ...
     strcmpi(string(checkpointInfo.FWPattern), string(expected.FWPattern)) && ...
     strcmpi(string(checkpointInfo.HWPattern), string(expected.HWPattern)) && ...
     isequaln(checkpointInfo.FWThick, expected.FWThick) && ...
     isequaln(checkpointInfo.HWThick, expected.HWThick) && ...
     isequaln(checkpointInfo.FWZmax, expected.FWZmax) && ...
     isequaln(checkpointInfo.HWZmax, expected.HWZmax) && ...
     isequaln(checkpointInfo.TargetN, expected.TargetN) && ...
     isequaln(checkpointInfo.SeedBase, expected.SeedBase) && ...
     isequaln(checkpointInfo.CorrCoef, expected.CorrCoef) && ...
     isequaln(checkpointInfo.FaultingDepth, expected.FaultingDepth) && ...
     isequaln(checkpointInfo.SandVcl, expected.SandVcl) && ...
     isequaln(checkpointInfo.ClayVcl, expected.ClayVcl);
end


function saveFinalRunCheckpoint(filePath, perms, meta, checkpointInfo)
% Save one completed diagnostic checkpoint atomically.

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


function setupPredictPaths()
% Ensure repository folders used by the examples are on the MATLAB path.

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
