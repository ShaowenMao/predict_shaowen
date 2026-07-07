%RUN_KR_UPSCALING_CL_MEDIAN_EXAMPLES_FULL87 Capillary-limit Kr upscaling.
%
% This script mirrors the completed full-87 Pc workflow for the median
% sand-ratio nonuniform example, but upscales relative permeability using
% the original package's rigorous capillary-limit branch:
%
%   upscaleKrReg(..., opt.kr_mode = 'CL')
%
% The inputs are replayed PREDICT realizations, so no PREDICT simulation is
% rerun here. For each selected Level-3 case, slice, and throw window, one
% fine-scale fault-core map is converted into MRST-style G/rock/fluid
% structures and passed to the original Kr upscaler. The representative
% curve for each case-window is then selected as a medoid over the combined
% gas and wetting-phase relative-permeability curves.
%
% Smoke test:
%   setenv('KR_CL_MAX_ROWS','2')
%   setenv('KR_CL_USE_PARALLEL','0')
%   run_kr_upscaling_cl_median_examples_full87
%
% Full run:
%   setenv('KR_CL_MAX_ROWS','')
%   setenv('KR_CL_USE_PARALLEL','1')
%   setenv('KR_CL_NUM_WORKERS','16')
%   run_kr_upscaling_cl_median_examples_full87

clear; clc;

scriptDir = fileparts(mfilename('fullpath'));
examplesDir = fileparts(scriptDir);
repoRoot = fileparts(examplesDir);

cfg = struct();
cfg.geologyId = "s05_c012";
cfg.level3CaseIds = [1 3 4 7];
cfg.caseToken = "cases_01_03_04_07";
cfg.windows = ["famp1", "famp2", "famp3", "famp4", "famp5", "famp6"];
cfg.sgGrid = linspace(0.02, 0.68, 80);
cfg.sourceRoot = fullfile('D:', 'codex_gom', 'UQ_workflow', ...
    'pc_upscaling_median_examples_full87');
cfg.replaySummaryCsv = fullfile(cfg.sourceRoot, 'tables', ...
    'replay_summary_with_full87_context_s05_c012_cases_01_03_04_07.csv');
cfg.upscalingZip = fullfile(repoRoot, 'upscaling.zip');
cfg.upscalingRoot = fullfile('D:', 'codex_gom', 'tmp_upscaling_zip_inspect');
cfg.mrstRoot = fullfile('C:', 'Users', 'Shaow', 'OneDrive', 'MIT', ...
    'mrst-2025a', 'SINTEF-AppliedCompSci-MRST-75749fa');
cfg.outputRoot = fullfile('D:', 'codex_gom', 'UQ_workflow', ...
    'kr_upscaling_cl_median_examples_full87');
cfg.useStrikeCollapsedGrid = parseLogicalEnv("KR_CL_USE_STRIKE_COLLAPSED", false);
cfg.useFastWholeGridCl = parseLogicalEnv("KR_CL_USE_FAST_WHOLEGRID", true);
cfg.saveRawSteps = parseLogicalEnv("KR_CL_SAVE_RAW_STEPS", false);
cfg.useMonotoneSgClamp = parseLogicalEnv("KR_CL_USE_MONOTONE_SG_CLAMP", false);
cfg.zeroOffDiagonalPermForTpfa = parseLogicalEnv("KR_CL_ZERO_OFFDIAGONAL_PERM_FOR_TPFA", false);
cfg.onlyRows = parseIntegerListEnv("KR_CL_ONLY_ROWS");

maxRowsText = strtrim(string(getenv('KR_CL_MAX_ROWS')));
if maxRowsText ~= ""
    cfg.maxRows = str2double(maxRowsText);
else
    cfg.maxRows = inf;
end
outputTag = matlab.lang.makeValidName(strtrim(string(getenv('KR_CL_OUTPUT_TAG'))));
if outputTag ~= ""
    cfg.outputRoot = cfg.outputRoot + "_" + outputTag;
end
cfg.useParallel = parseLogicalEnv("KR_CL_USE_PARALLEL", ~isfinite(cfg.maxRows));
cfg.numWorkers = parseNumericEnv("KR_CL_NUM_WORKERS", 16);
if isfinite(cfg.maxRows)
    if cfg.useStrikeCollapsedGrid
        gridTag = "_strikeCollapsed";
    else
        gridTag = "_fullGrid";
    end
    cfg.outputRoot = cfg.outputRoot + "_smoke" + string(cfg.maxRows) + gridTag;
end

cfg.curveDir = fullfile(cfg.outputRoot, 'curves');
cfg.tableDir = fullfile(cfg.outputRoot, 'tables');
cfg.figureDir = fullfile(cfg.outputRoot, 'figures');
cfg.checkpointDir = fullfile(cfg.curveDir, 'curve_checkpoints');
ensureFolder(cfg.curveDir);
ensureFolder(cfg.tableDir);
ensureFolder(cfg.figureDir);
ensureFolder(cfg.checkpointDir);

initializeKrPaths(cfg);
cfg.originalDeckFile = resolveOriginalDeckFile(cfg);

fprintf('\n=== Load replay summary for full-87 CL Kr upscaling ===\n')
assert(exist(cfg.replaySummaryCsv, 'file') == 2, ...
    'Missing replay summary: %s', cfg.replaySummaryCsv);
replaySummaryAll = readtable(cfg.replaySummaryCsv, 'TextType', 'string');
caseMask = replaySummaryAll.GeologyId == cfg.geologyId & ...
    ismember(replaySummaryAll.Level3CaseId, cfg.level3CaseIds);
replaySummary = replaySummaryAll(caseMask, :);
replaySummary.WindowOrder = windowOrder(replaySummary.Window);
replaySummary = sortrows(replaySummary, ...
    {'Level3CaseId', 'SliceIndex', 'WindowOrder'});
replaySummary.ProductionCurveId = (1:height(replaySummary))';

expectedRows = 87 * numel(cfg.windows) * numel(cfg.level3CaseIds);
assert(height(replaySummary) == expectedRows, ...
    'Expected %d rows for cases %s, found %d.', ...
    expectedRows, cfg.caseToken, height(replaySummary));
if ~isempty(cfg.onlyRows)
    assert(all(cfg.onlyRows >= 1 & cfg.onlyRows <= height(replaySummary)), ...
        'KR_CL_ONLY_ROWS contains row ids outside 1:%d.', height(replaySummary));
    replaySummary = replaySummary(cfg.onlyRows, :);
elseif isfinite(cfg.maxRows)
    replaySummary = replaySummary(1:min(cfg.maxRows, height(replaySummary)), :);
end
fprintf('Using %d replayed rows from: %s\n', ...
    height(replaySummary), cfg.replaySummaryCsv);

fprintf('\n=== Prepare deterministic material Pc/Kr curves ===\n')
krOpt = krClOptions(cfg.originalDeckFile, cfg.sgGrid);
krOpt.useStrikeCollapsedGrid = cfg.useStrikeCollapsedGrid;
krOpt.useFastWholeGridCl = cfg.useFastWholeGridCl;
krOpt.useParallel = cfg.useParallel;
krOpt.numWorkers = cfg.numWorkers;
krOpt.checkpointDir = cfg.checkpointDir;
krOpt.saveRawSteps = cfg.saveRawSteps;
krOpt.useMonotoneSgClamp = cfg.useMonotoneSgClamp;
krOpt.zeroOffDiagonalPermForTpfa = cfg.zeroOffDiagonalPermForTpfa;
krOpt.rawStepDir = fullfile(cfg.outputRoot, 'diagnostics', 'raw_cl_steps');
if krOpt.saveRawSteps
    ensureFolder(krOpt.rawStepDir);
end
fprintf('Reference deck: %s\n', cfg.originalDeckFile);
fprintf('Kr mode = %s, direction = %s, Pc scaling component = %s.\n', ...
    krOpt.krMode, krOpt.direction, krOpt.scalingPermComponent);
fprintf('Use exact strike-collapsed grid optimization: %d\n', ...
    krOpt.useStrikeCollapsedGrid);
fprintf('Use exact parallel execution: %d', krOpt.useParallel);
if krOpt.useParallel
    fprintf(' (%d workers)', krOpt.numWorkers);
end
fprintf('\n');
fprintf('Use fast whole-grid single-block CL solver: %d\n', ...
    krOpt.useFastWholeGridCl);
fprintf('Save raw Pc-step CL diagnostics: %d\n', krOpt.saveRawSteps);
fprintf('Use monotone endpoint Sg clamp diagnostic: %d\n', ...
    krOpt.useMonotoneSgClamp);
fprintf('Zero off-diagonal permeability terms for TPFA diagnostic: %d\n', ...
    krOpt.zeroOffDiagonalPermForTpfa);

curveLongCsv = fullfile(cfg.curveDir, ...
    sprintf('kr_curve_points_s05_c012_%s_cl_full87.csv', cfg.caseToken));
curveSummaryCsv = fullfile(cfg.tableDir, ...
    sprintf('kr_curve_summary_s05_c012_%s_cl_full87.csv', cfg.caseToken));
curveMatFile = fullfile(cfg.curveDir, ...
    sprintf('kr_curves_s05_c012_%s_cl_full87.mat', cfg.caseToken));

fprintf('\n=== Compute capillary-limit Kr curves ===\n')
if exist(curveMatFile, 'file') == 2
    fprintf('Loading cached CL Kr curve MAT: %s\n', curveMatFile);
    cached = load(curveMatFile, 'curveMat');
    curveMat = cached.curveMat;
else
    [curveLong, curveSummary, curveMat] = computeKrClCurves( ...
        replaySummary, krOpt);
    writetable(curveLong, curveLongCsv);
    writetable(curveSummary, curveSummaryCsv);
    save(curveMatFile, 'curveMat', 'krOpt', 'cfg', '-v7.3');
    fprintf('Saved CL Kr curve points: %s\n', curveLongCsv);
    fprintf('Saved CL Kr curve summary: %s\n', curveSummaryCsv);
    fprintf('Saved CL Kr curve MAT: %s\n', curveMatFile);
end
if ~isempty(cfg.onlyRows)
    fprintf('\nDiagnostic row-only Kr run complete; skipping medoid analysis and figures.\n')
    fprintf('Output root: %s\n', cfg.outputRoot);
    return
end

fprintf('\n=== Select CL Kr medoid curves ===\n')
results = analyzeKrMedoids(curveMat, cfg);
medoidSummaryCsv = fullfile(cfg.tableDir, ...
    sprintf('kr_medoid_summary_s05_c012_%s_cl_full87.csv', cfg.caseToken));
distanceSummaryCsv = fullfile(cfg.tableDir, ...
    sprintf('kr_distance_summary_s05_c012_%s_cl_full87.csv', cfg.caseToken));
writetable(results.MedoidSummary, medoidSummaryCsv);
writetable(results.DistanceSummary, distanceSummaryCsv);
save(fullfile(cfg.tableDir, ...
    sprintf('kr_medoid_results_s05_c012_%s_cl_full87.mat', cfg.caseToken)), ...
    'results', '-v7.3');
fprintf('Saved CL Kr medoid summary: %s\n', medoidSummaryCsv);
fprintf('Saved CL Kr distance summary: %s\n', distanceSummaryCsv);

fprintf('\n=== Generate CL Kr figures ===\n')
makeKrFigures(curveMat, results, cfg);

fprintf('\nCapillary-limit Kr upscaling complete.\n')
fprintf('Output root: %s\n', cfg.outputRoot);


function initializeKrPaths(cfg)
% Add MRST and original upscaling helper paths used by upscaleKrReg.

assert(exist(cfg.upscalingRoot, 'dir') == 7, ...
    'Missing extracted upscaling root: %s', cfg.upscalingRoot);
if exist(fullfile(cfg.mrstRoot, 'startup.m'), 'file') == 2
    run(fullfile(cfg.mrstRoot, 'startup.m'));
else
    warning('MRST startup not found. Continuing with current MATLAB path: %s', ...
        cfg.mrstRoot);
end
mrstModule add mimetic upscaling incomp coarsegrid deckformat
addpath(cfg.upscalingRoot);
addpath(fullfile(cfg.upscalingRoot, 'upscaling'));
end


function krOpt = krClOptions(deckFile, sgGrid)
% Return options for original capillary-limit relative-perm upscaling.

krOpt = struct();
krOpt.sgGrid = sgGrid(:);
krOpt.minPoro = 1.0e-4;
krOpt.minPermMD = 1.0e-9;
krOpt.mDInM2 = 9.869233e-16;
krOpt.refPermSandSI = 7.60393535652603e-13;
krOpt.refPoroSand = 0.289875;
krOpt.clayPceRmse = 0.2953;
krOpt.clayPceUncertaintyQuantile = 0.5;
krOpt.contactAngleDeg = 30;
krOpt.scalingPermComponent = "kzz";
krOpt.direction = 'z';
krOpt.krMode = 'CL';
krOpt.sg = 'sandClay';
krOpt.deckFile = deckFile;
krOpt.useStrikeCollapsedGrid = false;
krOpt.useFastWholeGridCl = true;
krOpt.useParallel = false;
krOpt.numWorkers = 1;
krOpt.referenceCurves = readSgofReferenceCurves(deckFile);
end


function deckFile = resolveOriginalDeckFile(cfg)
% Locate the original Eclipse deck, extracting upscaling.zip if needed.

deckRelPath = fullfile('eclipse_data_files', ...
    'gom_forUps_theta30_PVDO_incompRock.DATA');
candidate = fullfile(cfg.upscalingRoot, deckRelPath);
if exist(candidate, 'file') == 2
    deckFile = candidate;
    return
end

extractRoot = fullfile(cfg.outputRoot, 'reference_upscaling_zip');
candidate = fullfile(extractRoot, deckRelPath);
if exist(candidate, 'file') ~= 2
    assert(exist(cfg.upscalingZip, 'file') == 2, ...
        'Missing original upscaling deck and upscaling.zip: %s', cfg.upscalingZip);
    ensureFolder(extractRoot);
    unzip(cfg.upscalingZip, extractRoot);
end

assert(exist(candidate, 'file') == 2, ...
    'Could not locate extracted deck file: %s', candidate);
deckFile = candidate;
end


function curves = readSgofReferenceCurves(deckFile)
% Read the two original SGOF Pc/Kr tables from the Eclipse deck.

assert(exist(deckFile, 'file') == 2, 'Missing deck file: %s', deckFile);
lines = string(readlines(deckFile));
sgofLine = find(strtrim(lines) == "SGOF", 1, 'first');
assert(~isempty(sgofLine), 'No SGOF keyword found in %s.', deckFile);

tables = cell(1, 2);
current = [];
tableIndex = 1;
for i = (sgofLine + 1):numel(lines)
    raw = strtrim(lines(i));
    if raw == "" || startsWith(raw, "--")
        continue
    end
    if raw == "/"
        if ~isempty(current)
            tables{tableIndex} = current;
            tableIndex = tableIndex + 1;
            current = [];
            if tableIndex > 2
                break
            end
        end
        continue
    end

    values = sscanf(erase(raw, "/"), '%f');
    if numel(values) == 4
        current(end+1, :) = values(:)'; %#ok<AGROW>
    elseif ~isempty(current)
        break
    end
end

assert(~isempty(tables{1}) && ~isempty(tables{2}), ...
    'Could not parse two SGOF tables from %s.', deckFile);
curves.sand = tableToCurve(tables{1});
curves.clay = tableToCurve(tables{2});
end


function curve = tableToCurve(T)
% Convert one SGOF numeric table to interpolation-ready Pc/Kr arrays.

curve.sg = T(:, 1);
curve.krg = T(:, 2);
curve.krog = T(:, 3);
curve.pcBar = T(:, 4);
curve.pcPa = curve.pcBar * 1.0e5;
end


function [curveLong, curveSummary, curveMat] = computeKrClCurves( ...
        replaySummary, krOpt)
% Compute original capillary-limit Kr curves for every replay row.

n = height(replaySummary);
sgGrid = krOpt.sgGrid(:)';
krg = nan(n, numel(sgGrid));
krw = nan(n, numel(sgGrid));
summaryRows = cell(n, 36);
longRows = cell(n * numel(sgGrid), 19);
longIdx = 0;
curveCells = cell(n, 1);
outputFiles = cellstr(replaySummary.OutputFile);
windowNames = replaySummary.Window;
caseIds = replaySummary.Level3CaseId;
sliceIds = replaySummary.SliceIndex;
curveIds = replaySummary.ProductionCurveId;

realizationKeys = string(replaySummary.Window) + "|" + ...
    string(replaySummary.SelectedSampleIndex) + "|" + ...
    string(replaySummary.ReplaySeed);
[uniqueKeys, firstUniqueIdx] = unique(realizationKeys, 'stable');
[~, rowToUnique] = ismember(realizationKeys, uniqueKeys);
numUnique = numel(uniqueKeys);

% Prefer a representative row that already has a checkpoint. This lets an
% interrupted row-wise run be resumed without recomputing completed curves.
representativeIdx = firstUniqueIdx(:);
for u = 1:numUnique
    matchingRows = find(rowToUnique == u);
    for r = matchingRows(:)'
        if exist(krCurveCheckpointFile(krOpt.checkpointDir, r), 'file') == 2
            representativeIdx(u) = r;
            break
        end
    end
end

fprintf('Unique replay realizations for Kr: %d / %d assignment rows.\n', ...
    numUnique, n);

if krOpt.useParallel && n > 1
    ensureParallelPool(krOpt.numWorkers);
    uniqueCurveCells = cell(numUnique, 1);
    parfor u = 1:numUnique
        i = representativeIdx(u);
        fprintf('CL Kr unique %4d/%4d (row %4d/%4d): case %02d slice %02d %s\n', ...
            u, numUnique, i, n, caseIds(i), sliceIds(i), char(windowNames(i)));
        uniqueCurveCells{u} = loadOrComputeKrCurveCheckpoint( ...
            curveIds(i), outputFiles{i}, krOpt, windowNames(i));
    end
else
    uniqueCurveCells = cell(numUnique, 1);
    for u = 1:numUnique
        i = representativeIdx(u);
        fprintf('CL Kr unique %4d/%4d (row %4d/%4d): case %02d slice %02d %s\n', ...
            u, numUnique, i, n, caseIds(i), sliceIds(i), char(windowNames(i)));
        uniqueCurveCells{u} = loadOrComputeKrCurveCheckpoint( ...
            curveIds(i), outputFiles{i}, krOpt, windowNames(i));
    end
end
curveCells(:) = uniqueCurveCells(rowToUnique);

for i = 1:n
    curve = curveCells{i};
    krg(i, :) = curve.krg;
    krw(i, :) = curve.krw;

    summaryRows(i, :) = { ...
        curveIds(i), replaySummary.SourceRow(i), replaySummary.GeologyId(i), ...
        replaySummary.ScenarioName(i), replaySummary.CaseLabel(i), ...
        replaySummary.Level3CaseId(i), replaySummary.Level3CaseName(i), ...
        replaySummary.Window(i), replaySummary.SliceIndex(i), ...
        replaySummary.AssignedState(i), replaySummary.SamplingPool(i), ...
        replaySummary.SelectedSampleIndex(i), replaySummary.ReplaySeed(i), ...
        curve.poreVolume, curve.numCells, curve.numRegions, ...
        curve.numClayRegions, curve.clayPoreVolumeFraction, ...
        curve.meanLog10KzzMD, curve.medianLog10KzzMD, ...
        curve.p05Log10KzzMD, curve.p95Log10KzzMD, ...
        curve.krgAtSg20, curve.krgAtSg50, curve.krgAtSg65, ...
        curve.krwAtSg20, curve.krwAtSg50, curve.krwAtSg65, ...
        curve.krgArea, curve.krwArea, curve.mobileGasThresholdSg, ...
        curve.maxKrg, curve.maxKrw, krOpt.krMode, ...
        krOpt.scalingPermComponent, krOpt.clayPceUncertaintyQuantile};

    for j = 1:numel(sgGrid)
        longIdx = longIdx + 1;
        longRows(longIdx, :) = { ...
            curveIds(i), replaySummary.SourceRow(i), replaySummary.GeologyId(i), ...
            replaySummary.Level3CaseId(i), replaySummary.Level3CaseName(i), ...
            replaySummary.Window(i), replaySummary.SliceIndex(i), ...
            replaySummary.AssignedState(i), replaySummary.SamplingPool(i), ...
            replaySummary.SelectedSampleIndex(i), replaySummary.ReplaySeed(i), ...
            sgGrid(j), krg(i, j), krw(i, j), 1.0 - sgGrid(j), ...
            curve.clayPoreVolumeFraction, curve.medianLog10KzzMD, ...
            curve.krgArea, curve.krwArea};
    end
end

curveSummary = cell2table(summaryRows, 'VariableNames', ...
    {'CurveId', 'ReplaySourceRow', 'GeologyId', 'ScenarioName', ...
     'CaseLabel', 'Level3CaseId', 'Level3CaseName', 'Window', ...
     'SliceIndex', 'AssignedState', 'SamplingPool', 'SelectedSampleIndex', ...
     'ReplaySeed', 'PoreVolume', 'NumCells', 'NumRegions', ...
     'NumClayRegions', 'ClayPoreVolumeFraction', 'MeanLog10KzzMD', ...
     'MedianLog10KzzMD', 'P05Log10KzzMD', 'P95Log10KzzMD', ...
     'KrgAtSg20', 'KrgAtSg50', 'KrgAtSg65', ...
     'KrwAtSg20', 'KrwAtSg50', 'KrwAtSg65', ...
     'KrgArea', 'KrwArea', 'MobileGasThresholdSg', ...
     'MaxKrg', 'MaxKrw', 'KrMode', 'ScalingPermComponent', ...
     'ClayPceUncertaintyQuantile'});

curveLong = cell2table(longRows(1:longIdx, :), 'VariableNames', ...
    {'CurveId', 'ReplaySourceRow', 'GeologyId', 'Level3CaseId', ...
     'Level3CaseName', 'Window', 'SliceIndex', 'AssignedState', ...
     'SamplingPool', 'SelectedSampleIndex', 'ReplaySeed', ...
     'GasSaturation', 'Krg', 'Krw', 'WaterSaturation', ...
     'ClayPoreVolumeFraction', 'MedianLog10KzzMD', ...
     'KrgArea', 'KrwArea'});

curveMat = struct();
curveMat.sgGrid = sgGrid;
curveMat.krg = krg;
curveMat.krw = krw;
curveMat.summary = curveSummary;
end


function curve = krClCurveFromReplay(replay, krOpt, windowName)
% Compute one original capillary-limit Kr curve from replay data.

[G, CG, rock, fluid, fault3D, pcForKr, sgmax, opt, diagnostics] = ...
    buildKrInputsFromReplay(replay, krOpt, windowName);

warnStates = [ ...
    warning('off', 'MATLAB:nearlySingularMatrix'), ...
    warning('off', 'MATLAB:singularMatrix')];
cleanupWarnings = onCleanup(@() restoreWarningStates(warnStates));
if krOpt.useFastWholeGridCl
    [krwOut, krgOut, rawClSteps] = upscaleKrRegClWholeGridFast(G, rock, fluid, ...
        fault3D, krOpt.sgGrid(:), pcForKr(:), sgmax(:), opt);
else
    rawClSteps = struct();
    [krwOut, krgOut] = upscaleKrReg(G, CG, rock, fluid, fault3D, ...
        krOpt.sgGrid(:), pcForKr(:), sgmax(:), [], [], replay.U, opt, false);
end
clear cleanupWarnings

krg = krgOut(:, 2);
krw = krwOut(:, 2);
assert(all(isfinite(krg)) && all(isfinite(krw)), ...
    'Kr upscaling returned non-finite values.');
krg = min(max(krg(:)', 0), 1);
krw = min(max(krw(:)', 0), 1);

curve = diagnostics;
curve.krg = krg;
curve.krw = krw;
if krOpt.saveRawSteps
    curve.rawClSteps = rawClSteps;
end
curve.krgAtSg20 = interp1(krOpt.sgGrid, krg, 0.20, 'linear', 'extrap');
curve.krgAtSg50 = interp1(krOpt.sgGrid, krg, 0.50, 'linear', 'extrap');
curve.krgAtSg65 = interp1(krOpt.sgGrid, krg, 0.65, 'linear', 'extrap');
curve.krwAtSg20 = interp1(krOpt.sgGrid, krw, 0.20, 'linear', 'extrap');
curve.krwAtSg50 = interp1(krOpt.sgGrid, krw, 0.50, 'linear', 'extrap');
curve.krwAtSg65 = interp1(krOpt.sgGrid, krw, 0.65, 'linear', 'extrap');
curve.krgArea = trapz(krOpt.sgGrid, krg);
curve.krwArea = trapz(krOpt.sgGrid, krw);
idx = find(krg >= 0.01, 1, 'first');
if isempty(idx)
    curve.mobileGasThresholdSg = NaN;
else
    curve.mobileGasThresholdSg = krOpt.sgGrid(idx);
end
curve.maxKrg = max(krg);
curve.maxKrw = max(krw);
end


function [krwOut, krnOut, rawClSteps] = upscaleKrRegClWholeGridFast(G, rock, fluid, ...
        fault3D, sg, pc, sgmax, opt)
% Exact CL branch of upscaleKrReg for a single whole-grid coarse block.
%
% The original helper extracts the same full subgrid and rebuilds boundary
% data for every capillary-pressure level. Here the coarse partition is one
% block, so we reuse the whole grid and cached boundary data while keeping
% the same TPFA flow calculation and capillary-limit equations.

if strcmp(opt.fault, 'test')
    if strcmp(opt.dir, 'z')
        idDim = 2;
    else
        error('Fast CL test mode currently expects opt.dir = z.');
    end
else
    if strcmp(opt.dir, 'x')
        idDim = 1;
    elseif strcmp(opt.dir, 'y')
        idDim = 2;
    elseif strcmp(opt.dir, 'z')
        idDim = 3;
    else
        error('Unknown upscaling direction: %s', opt.dir);
    end
end

isSmear = fault3D.Grid.isSmear;
reg = rock.regions.saturation;
idReg = unique(reg);
nreg = numel(idReg);
ncreg = zeros(nreg, 1);
for n = 1:nreg
    ncreg(n) = sum(reg == idReg(n));
end

nS = pow2(7);
pcVal = logspace(log10(pc(2)), log10(pc(end)), nS - 1);
pcVal = [0 0.98 * pcVal(1) pcVal];
volume = sum(G.cells.volumes .* rock.poro);
krw = nan(nS, 1);
krn = nan(nS, 1);
sw = nan(nS, 1);
cellSgDecreaseCount = zeros(nS, 1);
cellKrgDecreaseCount = zeros(nS, 1);
cellGasPermDecreaseCount = zeros(nS, 1);
maxCellSgDecrease = zeros(nS, 1);
maxCellKrgDecrease = zeros(nS, 1);
maxCellGasPermDecrease = zeros(nS, 1);
prevSgCells = [];
prevKrnv = [];
prevKnv = [];
tpfaCache = buildWholeGridTpfaCache(G, idDim);

for i = 1:numel(pcVal)
    sgCells = nan(G.cells.num, 1);
    krwv = nan(G.cells.num, 1);
    krnv = nan(G.cells.num, 1);
    for n = 1:nreg
        regId = idReg(n);
        idcell = reg == regId;
        if pcVal(i) >= fluid.pcOG{regId}(sgmax(n))
            endpointSg = 0.999 * sgmax(n);
            if isfield(opt, 'useMonotoneSgClamp') && opt.useMonotoneSgClamp
                if isempty(prevSgCells)
                    sgCells(idcell) = endpointSg;
                else
                    sgCells(idcell) = max(endpointSg, prevSgCells(idcell));
                end
            else
                sgCells(idcell) = endpointSg;
            end
        else
            pcv = pcVal(i);
            sgCells(idcell) = fluid.pcInv{regId}(pcv * ones(ncreg(n), 1));
            assert(all(sgCells(idcell) < sgmax(n)))
        end
    end
    if i > 1
        sgCells(sgCells < 1e-5) = 1e-5;
    end

    swCells = 1 - sgCells;
    sw(i) = sum(swCells .* G.cells.volumes .* rock.poro) / volume;

    krwv(~isSmear) = fluid.krOG{1}(swCells(~isSmear));
    krwv(isSmear) = fluid.krOG{2}(swCells(isSmear));
    krnv(~isSmear) = fluid.krG{1}(sgCells(~isSmear));
    krnv(isSmear) = fluid.krG{2}(sgCells(isSmear));
    krnv(krnv < 1e-3) = 1e-3;

    if isfield(opt, 'zeroOffDiagonalPermForTpfa') && ...
            opt.zeroOffDiagonalPermForTpfa && size(fault3D.Grid.perm, 2) >= 6
        tpfaPerm = fault3D.Grid.perm;
        tpfaPerm(:, [2 3 5]) = 0;
    else
        tpfaPerm = fault3D.Grid.perm;
    end
    kwv = krwv .* tpfaPerm;
    knv = krnv .* tpfaPerm;
    if ~isempty(prevSgCells)
        dSgCells = sgCells - prevSgCells;
        dKrnv = krnv - prevKrnv;
        dKnv = knv - prevKnv;
        cellSgDecreaseCount(i) = sum(dSgCells < -1e-12);
        cellKrgDecreaseCount(i) = sum(dKrnv < -1e-12);
        cellGasPermDecreaseCount(i) = sum(dKnv(:) < -1e-18);
        maxCellSgDecrease(i) = max(max(-dSgCells, 0), [], 'omitnan');
        maxCellKrgDecrease(i) = max(max(-dKrnv, 0), [], 'omitnan');
        maxCellGasPermDecrease(i) = max(max(-dKnv(:), 0), [], 'omitnan');
    end

    kw = upscalePermWholeGridTpfa(G, kwv, tpfaCache);
    if i == 1
        kn = 0;
    else
        kn = upscalePermWholeGridTpfa(G, knv, tpfaCache);
    end

    krw(i) = kw / fault3D.Perm(idDim);
    if krw(i) > 1
        krw(i) = 1;
    end
    krn(i) = kn / fault3D.Perm(idDim);
    prevSgCells = sgCells;
    prevKrnv = krnv;
    prevKnv = knv;
end

assert(~any(isnan(sw)))
assert(~any(isnan(krw)))
assert(~any(isnan(krn)))
assert(max(krw) <= 1)
assert(min(krw) >= 0)
assert(max(krn) <= 1)
assert(min(krn) >= 0)
assert(max(sw) <= 1)
assert(min(sw) >= min(1 - sgmax))

sgUps = 1 - sw;
krwOut = nan(numel(sg), 2);
krnOut = nan(numel(sg), 2);
krwOut(:, 2) = interp1(sgUps, krw, sg);
krnOut(:, 2) = interp1(sgUps, krn, sg);
rawClSteps = struct();
rawClSteps.pcVal = pcVal(:);
rawClSteps.sgUps = sgUps(:);
rawClSteps.krwRaw = krw(:);
rawClSteps.krgRaw = krn(:);
rawClSteps.swRaw = sw(:);
rawClSteps.sgGrid = sg(:);
rawClSteps.krwInterp = krwOut(:, 2);
rawClSteps.krgInterp = krnOut(:, 2);
rawClSteps.sgUpsDiff = diff(sgUps(:));
rawClSteps.cellSgDecreaseCount = cellSgDecreaseCount(:);
rawClSteps.cellKrgDecreaseCount = cellKrgDecreaseCount(:);
rawClSteps.cellGasPermDecreaseCount = cellGasPermDecreaseCount(:);
rawClSteps.maxCellSgDecrease = maxCellSgDecrease(:);
rawClSteps.maxCellKrgDecrease = maxCellKrgDecrease(:);
rawClSteps.maxCellGasPermDecrease = maxCellGasPermDecrease(:);
end


function cache = buildWholeGridTpfaCache(G, dim)
% Cache boundary-condition data for whole-grid TPFA upscaling.

fluid = initSingleFluid('mu', 1 * Pascal * second, ...
    'rho', 1 * kilogram / meter^3);
if dim == 1
    tags = [1 2];
elseif dim == 2
    tags = [3 4];
elseif dim == 3
    tags = [5 6];
else
    error('Unknown dimension id: %d', dim);
end

bndFaces = any(G.faces.neighbors == 0, 2);
ind = bndFaces(G.cells.faces(:, 1));
faceAndTag = G.cells.faces(ind, :);
faces1 = faceAndTag(faceAndTag(:, 2) == tags(1));
faces2 = faceAndTag(faceAndTag(:, 2) == tags(2));
bc = addBC([], faces1, 'pressure', 1 * barsa);
bc = addBC(bc, faces2, 'pressure', 0);

cache = struct();
cache.fluid = fluid;
cache.bc = bc;
cache.faces2 = faces2;
cache.area = sum(G.faces.areas(faces2, :));
cache.L = abs(G.faces.centroids(faces1(1), dim) - ...
    G.faces.centroids(faces2(1), dim));
end


function k = upscalePermWholeGridTpfa(G, perm, cache)
% Compute whole-grid effective permeability using the same TPFA solve.

rockUps = struct();
rockUps.perm = perm;
T = computeTrans(G, rockUps);
rSol = initResSol(G, 0.0);
rSol = incompTPFA(rSol, G, T, cache.fluid, 'bc', cache.bc, ...
    'LinSolve', @mldivide);
q = abs(sum(rSol.flux(cache.faces2)));
k = q * cache.L / (1 * barsa * cache.area);
end


function curve = loadOrComputeKrCurveCheckpoint(curveId, outputFile, krOpt, windowName)
% Load a completed curve checkpoint or compute and save it.

checkpointFile = krCurveCheckpointFile(krOpt.checkpointDir, curveId);
if exist(checkpointFile, 'file') == 2
    S = load(checkpointFile, 'curve');
    curve = S.curve;
    return
end

S = load(outputFile, 'replay');
curve = krClCurveFromReplay(S.replay, krOpt, windowName);
if krOpt.saveRawSteps && isfield(curve, 'rawClSteps')
    rawSteps = curve.rawClSteps;
    rawFile = krRawStepFile(krOpt.rawStepDir, curveId);
    save(rawFile, 'rawSteps', '-v7.3');
    fprintf('Saved raw CL Pc-step diagnostic: %s\n', rawFile);
end
tmpFile = checkpointFile + ".tmp_" + string(feature('getpid')) + ".mat";
save(tmpFile, 'curve', '-v7.3');
if exist(checkpointFile, 'file') ~= 2
    movefile(tmpFile, checkpointFile);
elseif exist(tmpFile, 'file') == 2
    delete(tmpFile);
end
end


function checkpointFile = krCurveCheckpointFile(checkpointDir, curveId)
% Build a stable checkpoint filename for a curve row.

checkpointFile = fullfile(checkpointDir, sprintf('kr_curve_%05d.mat', curveId));
end


function rawFile = krRawStepFile(rawStepDir, curveId)
% Build a stable filename for raw Pc-step CL diagnostics.

rawFile = fullfile(rawStepDir, sprintf('kr_raw_cl_steps_%05d.mat', curveId));
end


function restoreWarningStates(warnStates)
% Restore warning states changed during noisy MRST linear solves.

for i = 1:numel(warnStates)
    warning(warnStates(i).state, warnStates(i).identifier);
end
end


function ensureParallelPool(numWorkers)
% Start a process pool for independent Kr curve jobs if needed.

pool = gcp('nocreate');
if isempty(pool)
    parpool('Processes', numWorkers);
elseif pool.NumWorkers ~= numWorkers
    warning('Existing parallel pool has %d workers; requested %d.', ...
        pool.NumWorkers, numWorkers);
end
end


function [G, CG, rock, fluid, fault3D, pcForKr, sgmax, opt, diagnostics] = ...
        buildKrInputsFromReplay(replay, krOpt, windowName)
% Build MRST-style inputs required by original upscaleKrReg.

grid = replay.Grid;
if krOpt.useStrikeCollapsedGrid
    [G, CG, grid] = collapseReplayGridAlongStrike(replay, grid);
else
    G = replay.G;
    CG = replay.CG;
end
poroAll = max(grid.poro(:), krOpt.minPoro);
perm = grid.perm;
units = grid.units(:);
isSmear = logical(grid.isSmear(:));
volume = G.cells.volumes(:);
permComponentSI = selectPermComponent(perm, krOpt);
permComponentSI = max(permComponentSI(:), krOpt.minPermMD * krOpt.mDInM2);
log10KzzMD = log10(permComponentSI ./ krOpt.mDInM2);

valid = isfinite(poroAll) & isfinite(permComponentSI) & ...
    isfinite(volume) & volume > 0 & units > 0;
assert(all(valid), 'Kr runner expects valid replay cells for all fault-core cells.');

rock = struct();
rock.poro = poroAll;
rock.perm = perm;
rock.regions = struct();
rock.regions.saturation = units;
rock.regions.rocknum = ones(G.cells.num, 1);
rock.regions.rocknum(isSmear) = 2;

fault3D = struct();
fault3D.Grid = struct();
fault3D.Grid.isSmear = isSmear;
fault3D.Grid.perm = perm;
fault3D.Perm = replay.PermMD(:)' .* krOpt.mDInM2;

idReg = unique(units(:))';
fluid = struct();
fluid.krPts = struct();
fluid.krPts.g = [ ...
    krOpt.referenceCurves.sand.sg(1), NaN, krOpt.referenceCurves.sand.sg(end); ...
    krOpt.referenceCurves.clay.sg(1), NaN, krOpt.referenceCurves.clay.sg(end)];
fluid.pcOG = cell(1, max(idReg));
fluid.pcInv = cell(1, max(idReg));
fluid.krG = cell(1, 2);
fluid.krOG = cell(1, 2);
fluid.isclay = false(1, numel(idReg));

fluid.krG{1} = @(sg) interpBounded( ...
    krOpt.referenceCurves.sand.sg, krOpt.referenceCurves.sand.krg, sg);
fluid.krG{2} = @(sg) interpBounded( ...
    krOpt.referenceCurves.clay.sg, krOpt.referenceCurves.clay.krg, sg);
[swSand, iSand] = sort(1.0 - krOpt.referenceCurves.sand.sg);
[swClay, iClay] = sort(1.0 - krOpt.referenceCurves.clay.sg);
fluid.krOG{1} = @(sw) interpBounded( ...
    swSand, krOpt.referenceCurves.sand.krog(iSand), sw);
fluid.krOG{2} = @(sw) interpBounded( ...
    swClay, krOpt.referenceCurves.clay.krog(iClay), sw);

for n = 1:numel(idReg)
    id = units == idReg(n);
    smearFrac = mean(double(isSmear(id)), 'omitnan');
    isClay = smearFrac >= 0.5;
    fluid.isclay(n) = isClay;
    region = struct();
    region.isClay = isClay;
    region.permSI = mean(permComponentSI(id), 'omitnan');
    region.poro = mean(poroAll(id), 'omitnan');
    [sgRef, pcRefPa] = scaledRegionPc(region, krOpt);
    fluid.pcOG{idReg(n)} = @(sg) interp1(sgRef, pcRefPa, sg, ...
        'linear', 'extrap');
end

[fluid, pcForKr, sgmax] = preparePcInversesForKr(G, rock, fluid, krOpt);

opt = struct();
opt.kr_mode = krOpt.krMode;
opt.sg = krOpt.sg;
opt.window = char(windowName);
opt.fault = 'predict_3D';
opt.dir = krOpt.direction;
opt.theta = [krOpt.contactAngleDeg, krOpt.contactAngleDeg];
opt.useMonotoneSgClamp = krOpt.useMonotoneSgClamp;
opt.zeroOffDiagonalPermForTpfa = krOpt.zeroOffDiagonalPermForTpfa;

poreWeights = poroAll .* volume;
diagnostics = struct();
diagnostics.poreVolume = sum(poreWeights, 'omitnan');
diagnostics.numCells = G.cells.num;
diagnostics.numRegions = numel(idReg);
diagnostics.numClayRegions = sum(fluid.isclay);
diagnostics.clayPoreVolumeFraction = ...
    sum(poreWeights(isSmear), 'omitnan') ./ max(sum(poreWeights, 'omitnan'), realmin);
diagnostics.meanLog10KzzMD = mean(log10KzzMD, 'omitnan');
diagnostics.medianLog10KzzMD = median(log10KzzMD, 'omitnan');
diagnostics.p05Log10KzzMD = prctile(log10KzzMD, 5);
diagnostics.p95Log10KzzMD = prctile(log10KzzMD, 95);
end


function [Gout, CGout, gridOut] = collapseReplayGridAlongStrike(replay, gridIn)
% Collapse repeated along-strike cells into one exact equivalent y column.
%
% The replayed PREDICT maps are extruded along strike. For vertical Kr
% upscaling with uniform boundary conditions, a 100 x 10 x 100 repeated
% grid has the same solution as a 100 x 1 x 100 grid with the same total
% along-strike physical length. This preserves the original TPFA upscaling
% problem while greatly reducing the linear-system size.

Gin = replay.G;
dims = double(Gin.cartDims(:)');
if numel(dims) < 3 || dims(2) == 1
    Gout = Gin;
    CGout = replay.CG;
    gridOut = gridIn;
    return
end

fieldsToCheck = {'isSmear', 'units', 'vcl', 'poro'};
for i = 1:numel(fieldsToCheck)
    values = gridIn.(fieldsToCheck{i});
    assert(isStrikeInvariant(values, dims), ...
        'Cannot use strike-collapsed Kr grid: field %s varies along strike.', ...
        fieldsToCheck{i});
end
for j = 1:size(gridIn.perm, 2)
    assert(isStrikeInvariant(gridIn.perm(:, j), dims), ...
        'Cannot use strike-collapsed Kr grid: perm column %d varies along strike.', j);
end

physDim = max(Gin.nodes.coords, [], 1) - min(Gin.nodes.coords, [], 1);
Gout = computeGeometry(cartGrid([dims(1), 1, dims(3)], physDim));
p = partitionCartGrid(Gout.cartDims, [1, 1, 1]);
CGout = generateCoarseGrid(Gout, p);

gridOut = struct();
gridOut.isSmear = collapseVectorAlongStrike(gridIn.isSmear, dims);
gridOut.units = collapseVectorAlongStrike(gridIn.units, dims);
gridOut.vcl = collapseVectorAlongStrike(gridIn.vcl, dims);
gridOut.poro = collapseVectorAlongStrike(gridIn.poro, dims);
gridOut.perm = zeros(Gout.cells.num, size(gridIn.perm, 2));
for j = 1:size(gridIn.perm, 2)
    gridOut.perm(:, j) = collapseVectorAlongStrike(gridIn.perm(:, j), dims);
end
end


function tf = isStrikeInvariant(values, dims)
% Return true if every along-strike column contains the same x-z map.

arr = reshape(values(:), dims);
first = arr(:, 1, :);
diffVal = abs(arr - repmat(first, [1, dims(2), 1]));
tf = max(diffVal(:), [], 'omitnan') <= 1e-10;
end


function out = collapseVectorAlongStrike(values, dims)
% Extract one x-z map from a repeated x-y-z field.

arr = reshape(values(:), dims);
out = reshape(arr(:, 1, :), [], 1);
end


function [fluid, pcVal, sgmax] = preparePcInversesForKr(G, rock, fluid, krOpt)
% Populate fluid.pcInv and Pc range exactly as used by the original IP setup.

reg = rock.regions.saturation;
idReg = unique(reg);
nreg = numel(idReg);
pcv2 = inf;
pcMax = 0;
sgmax = zeros(1, nreg);
pcAtSgMax = zeros(1, max(idReg));

for n = 1:nreg
    regId = idReg(n);
    if strcmp(krOpt.sg, 'sandClay')
        if fluid.isclay(n)
            sgmin = fluid.krPts.g(2, 1);
            sgmax(n) = fluid.krPts.g(2, 3);
        else
            sgmin = fluid.krPts.g(1, 1);
            sgmax(n) = fluid.krPts.g(1, 3) - 0.01;
        end
    else
        sgmin = fluid.krPts.g(n, 1);
        sgmax(n) = fluid.krPts.g(n, 3);
    end

    sgvals = linspace(sgmin, sgmax(n), pow2(6)-1)';
    sgvals = [sgvals(1); sgvals(1)+1e-3; sgvals(2:end)];
    pcvals = fluid.pcOG{regId}(sgvals);
    pcvals = makeStrictlyIncreasing(pcvals);
    pcv2 = min(pcv2, pcvals(2));
    fluid.pcInv{regId} = @(pcOG) interp1(pcvals, sgvals, pcOG, ...
        'linear', 'extrap');
    pcAtSgMax(regId) = fluid.pcOG{regId}(sgmax(n));
    if fluid.isclay(n)
        pcMax = max(pcMax, pcAtSgMax(regId));
    end
end

if ~(isfinite(pcMax) && pcMax > pcv2)
    pcMax = max(pcAtSgMax(pcAtSgMax > 0));
end
assert(isfinite(pcv2) && isfinite(pcMax) && pcMax > pcv2, ...
    'Invalid Pc range for Kr upscaling.');
pcVal = logspace(log10(pcv2), log10(0.99 * pcMax), pow2(6)-2);
pcVal = [0, 0.98 * pcv2, pcVal];
end


function [sgRef, pcPa] = scaledRegionPc(region, krOpt)
% Return scaled material Pc curve for one material region.

if region.isClay
    ref = krOpt.referenceCurves.clay;
    log10KMD = log10(max(region.permSI / krOpt.mDInM2, krOpt.minPermMD));
    pceHgBar = 10.^(-0.1992 * log10KMD + 1.407 - krOpt.clayPceRmse + ...
        krOpt.clayPceUncertaintyQuantile * 2 * krOpt.clayPceRmse);
    pceCo2WaterPa = 1.0e5 * pceHgBar * ...
        abs(cosd(krOpt.contactAngleDeg) * 25 / (cosd(140) * 485));
    entryRefPa = interp1(ref.sg, ref.pcPa, 0.10, 'linear', 'extrap');
    pcPa = ref.pcPa .* (pceCo2WaterPa ./ entryRefPa);
else
    ref = krOpt.referenceCurves.sand;
    scale = sqrt((krOpt.refPermSandSI * region.poro) ./ ...
        (krOpt.refPoroSand * region.permSI));
    pcPa = ref.pcPa .* scale;
end
sgRef = ref.sg;
pcPa = makeStrictlyIncreasing(pcPa);
end


function y = interpBounded(x, v, xq)
% Interpolate and clip relative permeability values to [0, 1].

y = interp1(x(:), v(:), xq, 'linear', 'extrap');
y = min(max(y, 0), 1);
end


function pcPa = makeStrictlyIncreasing(pcPa)
% Add tiny monotonic increments so inverse interpolation remains stable.

pcPa = pcPa(:);
for i = 2:numel(pcPa)
    if pcPa(i) <= pcPa(i-1)
        pcPa(i) = pcPa(i-1) + max(abs(pcPa(i-1)) * 1e-9, 1e-6);
    end
end
end


function kSI = selectPermComponent(perm, krOpt)
% Select the permeability component used for Pc scaling.

switch lower(string(krOpt.scalingPermComponent))
    case "kxx"
        col = 1;
    case "kyy"
        if size(perm, 2) >= 4
            col = 4;
        elseif size(perm, 2) >= 2
            col = 2;
        else
            col = 1;
        end
    case "kzz"
        if size(perm, 2) >= 6
            col = 6;
        elseif size(perm, 2) >= 3
            col = 3;
        else
            col = 1;
        end
    otherwise
        error('Unknown permeability component: %s', krOpt.scalingPermComponent);
end
kSI = perm(:, col);
end


function results = analyzeKrMedoids(curveMat, cfg)
% Select one medoid Kr curve pair per Level-3 case and window.

summary = curveMat.summary;
windows = cfg.windows;
medoidRows = {};
distanceRows = {};

for c = 1:numel(cfg.level3CaseIds)
    caseId = cfg.level3CaseIds(c);
    for w = 1:numel(windows)
        windowName = windows(w);
        idx = find(summary.Level3CaseId == caseId & summary.Window == windowName);
        if isfinite(cfg.maxRows)
            if isempty(idx)
                continue
            end
        else
            assert(numel(idx) == 87, ...
                'Expected 87 curves for case %02d, %s; found %d.', ...
                caseId, windowName, numel(idx));
        end

        features = [curveMat.krg(idx, :), curveMat.krw(idx, :)];
        distances = pairwiseRmsDistance(features);
        meanDistance = mean(distances, 2, 'omitnan');
        [minMeanDistance, localMedoid] = min(meanDistance);
        medoidCurveId = idx(localMedoid);
        upper = distances(triu(true(size(distances)), 1));

        medoidRows(end+1, :) = { ...
            summary.GeologyId(idx(1)), caseId, ...
            summary.Level3CaseName(idx(1)), windowName, medoidCurveId, ...
            summary.SliceIndex(medoidCurveId), ...
            summary.SelectedSampleIndex(medoidCurveId), ...
            summary.AssignedState(medoidCurveId), ...
            summary.SamplingPool(medoidCurveId), ...
            summary.KrgAtSg20(medoidCurveId), ...
            summary.KrgAtSg50(medoidCurveId), ...
            summary.KrgAtSg65(medoidCurveId), ...
            summary.KrwAtSg20(medoidCurveId), ...
            summary.KrwAtSg50(medoidCurveId), ...
            summary.KrwAtSg65(medoidCurveId), ...
            summary.KrgArea(medoidCurveId), ...
            summary.KrwArea(medoidCurveId), ...
            minMeanDistance, median(meanDistance, 'omitnan'), ...
            max(meanDistance, [], 'omitnan')}; %#ok<AGROW>

        distanceRows(end+1, :) = { ...
            summary.GeologyId(idx(1)), caseId, ...
            summary.Level3CaseName(idx(1)), windowName, numel(idx), ...
            min(upper, [], 'omitnan'), median(upper, 'omitnan'), ...
            prctile(upper, 90), max(upper, [], 'omitnan'), ...
            minMeanDistance}; %#ok<AGROW>
    end
end

results = struct();
results.MedoidSummary = cell2table(medoidRows, 'VariableNames', ...
    {'GeologyId', 'Level3CaseId', 'Level3CaseName', 'Window', ...
     'MedoidCurveId', 'MedoidSliceIndex', 'MedoidSelectedSampleIndex', ...
     'AssignedState', 'SamplingPool', 'KrgAtSg20', 'KrgAtSg50', ...
     'KrgAtSg65', 'KrwAtSg20', 'KrwAtSg50', 'KrwAtSg65', ...
     'KrgArea', 'KrwArea', 'MedoidMeanDistance', ...
     'MedianMeanDistance', 'MaxMeanDistance'});
results.DistanceSummary = cell2table(distanceRows, 'VariableNames', ...
    {'GeologyId', 'Level3CaseId', 'Level3CaseName', 'Window', ...
     'NumCurves', 'PairDistanceMin', 'PairDistanceMedian', ...
     'PairDistanceP90', 'PairDistanceMax', 'MedoidMeanDistance'});
end


function distances = pairwiseRmsDistance(curves)
% Pairwise RMS distance between rows of a curve-feature matrix.

n = size(curves, 1);
distances = zeros(n, n);
for i = 1:n
    diffs = curves - curves(i, :);
    distances(:, i) = sqrt(mean(diffs.^2, 2, 'omitnan'));
end
end


function makeKrFigures(curveMat, results, cfg)
% Plot full-87 Kr curve ensembles and selected medoids.

summary = curveMat.summary;
windows = cfg.windows;
for c = 1:numel(cfg.level3CaseIds)
    caseId = cfg.level3CaseIds(c);
    caseRows = find(summary.Level3CaseId == caseId);
    if isempty(caseRows)
        continue
    end
    caseName = displayCaseName(summary.Level3CaseName(caseRows(1)));
    nCurvesByWindow = zeros(1, numel(windows));
    for ww = 1:numel(windows)
        nCurvesByWindow(ww) = sum(summary.Level3CaseId == caseId & ...
            summary.Window == windows(ww));
    end
    maxCurves = max(nCurvesByWindow);
    if all(nCurvesByWindow(nCurvesByWindow > 0) == 87) && maxCurves == 87
        runLabel = 'Full-87 capillary-limit relative-permeability curves by window';
        ensembleLabel = 'Grey = 87 slices; red = medoid curve';
    else
        runLabel = 'Partial capillary-limit relative-permeability smoke test';
        ensembleLabel = sprintf('Grey = available slices, max n = %d; red = medoid curve', maxCurves);
    end

    fig = figure('Color', 'w', 'Position', [50, 50, 1900, 760]);
    tiledlayout(2, numel(windows), 'TileSpacing', 'compact', 'Padding', 'compact');
    for w = 1:numel(windows)
        windowName = windows(w);
        idx = find(summary.Level3CaseId == caseId & summary.Window == windowName);
        medoidIdx = results.MedoidSummary.MedoidCurveId( ...
            results.MedoidSummary.Level3CaseId == caseId & ...
            results.MedoidSummary.Window == windowName);

        nexttile(w)
        if isempty(idx)
            title(sprintf('W%d', w), 'FontSize', 18, 'FontWeight', 'bold');
            ylim([0 1]); xlim([min(curveMat.sgGrid), max(curveMat.sgGrid)]);
            grid on
            if w == 1
                ylabel('KRG', 'FontSize', 18);
            end
            set(gca, 'FontSize', 14);
            nexttile(numel(windows) + w)
            ylim([0 1]); xlim([min(curveMat.sgGrid), max(curveMat.sgGrid)]);
            grid on
            xlabel('Gas saturation, S_g', 'FontSize', 18);
            if w == 1
                ylabel('KRWG', 'FontSize', 18);
            end
            set(gca, 'FontSize', 14);
            continue
        end
        hold on
        plot(curveMat.sgGrid(:), curveMat.krg(idx, :)', '-', ...
            'Color', [0.68 0.68 0.68], 'LineWidth', 0.55);
        if ~isempty(medoidIdx)
            plot(curveMat.sgGrid(:), curveMat.krg(medoidIdx, :)', 'r-', ...
                'LineWidth', 2.5);
        end
        hold off
        grid on
        ylim([0 1]); xlim([min(curveMat.sgGrid), max(curveMat.sgGrid)]);
        title(sprintf('W%d', w), 'FontSize', 18, 'FontWeight', 'bold');
        if w == 1
            ylabel('KRG', 'FontSize', 18);
        end
        set(gca, 'FontSize', 14);

        nexttile(numel(windows) + w)
        hold on
        plot(curveMat.sgGrid(:), curveMat.krw(idx, :)', '-', ...
            'Color', [0.68 0.68 0.68], 'LineWidth', 0.55);
        if ~isempty(medoidIdx)
            plot(curveMat.sgGrid(:), curveMat.krw(medoidIdx, :)', 'r-', ...
                'LineWidth', 2.5);
        end
        hold off
        grid on
        ylim([0 1]); xlim([min(curveMat.sgGrid), max(curveMat.sgGrid)]);
        xlabel('Gas saturation, S_g', 'FontSize', 18);
        if w == 1
            ylabel('KRWG', 'FontSize', 18);
        end
        set(gca, 'FontSize', 14);
    end
    sgtitle({sprintf('Case %02d: %s', caseId, caseName), ...
        runLabel, ensembleLabel}, ...
        'FontSize', 22, 'FontWeight', 'bold', 'Interpreter', 'none');
    saveFigureBoth(fig, cfg.figureDir, ...
        sprintf('s05_c012_case%02d_cl_full87_kr_curves_with_medoids', caseId));
    close(fig);
end

fig = figure('Color', 'w', 'Position', [100, 100, 1500, 680]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
metrics = ["KrgAtSg50", "KrwAtSg50"];
titles = ["Medoid KRG at S_g = 0.50", "Medoid KRWG at S_g = 0.50"];
for m = 1:numel(metrics)
    nexttile
    M = nan(numel(cfg.level3CaseIds), numel(windows));
    for c = 1:numel(cfg.level3CaseIds)
        caseId = cfg.level3CaseIds(c);
        for w = 1:numel(windows)
            row = results.MedoidSummary.Level3CaseId == caseId & ...
                results.MedoidSummary.Window == windows(w);
            if any(row)
                M(c, w) = results.MedoidSummary.(metrics(m))(row);
            end
        end
    end
    imagesc(M);
    colormap(gca, parula);
    clim([0 1]);
    colorbar;
    xticks(1:numel(windows));
    xticklabels("W" + string(1:numel(windows)));
    yticks(1:numel(cfg.level3CaseIds));
    yticklabels("Case " + compose("%02d", cfg.level3CaseIds));
    title(titles(m), 'FontSize', 18, 'FontWeight', 'bold');
    set(gca, 'FontSize', 14);
    for c = 1:size(M, 1)
        for w = 1:size(M, 2)
            if isfinite(M(c, w))
                text(w, c, sprintf('%.2f', M(c, w)), ...
                    'HorizontalAlignment', 'center', ...
                    'FontSize', 13, 'FontWeight', 'bold', 'Color', 'w');
            end
        end
    end
end
sgtitle('Selected medoid relative-permeability levels', ...
    'FontSize', 22, 'FontWeight', 'bold');
saveFigureBoth(fig, cfg.figureDir, ...
    's05_c012_cases_01_03_04_07_cl_medoid_kr_levels');
close(fig);
end


function label = displayCaseName(rawName)
% Convert internal Level-3 case names to presentation-friendly labels.

name = lower(strtrim(string(rawName)));
switch name
    case "independent_draw_1"
        label = 'Independent draw 1';
    case "strong_fault_wide_low"
        label = 'Strong fault-wide low';
    case "strong_fault_wide_high"
        label = 'Strong fault-wide high';
    case "strong_grouped_g3_low_g4_high"
        label = 'Strong grouped G3 low / G4 high';
    otherwise
        label = char(strrep(string(rawName), "_", " "));
end
end


function order = windowOrder(windowNames)
% Convert famp1..famp6 labels to numeric order.

windowNames = string(windowNames);
order = nan(size(windowNames));
for i = 1:numel(windowNames)
    token = regexp(char(windowNames(i)), '\d+', 'match', 'once');
    order(i) = str2double(token);
end
end


function value = parseLogicalEnv(name, defaultValue)
% Parse a boolean environment variable with a safe default.

text = lower(strtrim(string(getenv(char(name)))));
if text == ""
    value = defaultValue;
elseif any(text == ["1", "true", "yes", "on"])
    value = true;
elseif any(text == ["0", "false", "no", "off"])
    value = false;
else
    error('Environment variable %s must be true/false or 1/0.', char(name));
end
end


function value = parseNumericEnv(name, defaultValue)
% Parse a numeric environment variable with a safe default.

text = strtrim(string(getenv(char(name))));
if text == ""
    value = defaultValue;
else
    value = str2double(text);
    assert(isfinite(value) && value > 0, ...
        'Environment variable %s must be a positive number.', char(name));
end
end


function values = parseIntegerListEnv(name)
% Parse a comma/space/semicolon-delimited positive integer list.

text = strtrim(string(getenv(char(name))));
if text == ""
    values = [];
    return
end
tokens = regexp(char(text), '[,;\s]+', 'split');
tokens = tokens(~cellfun(@isempty, tokens));
values = zeros(numel(tokens), 1);
for i = 1:numel(tokens)
    values(i) = str2double(tokens{i});
end
assert(all(isfinite(values)) && all(values == round(values)) && all(values > 0), ...
    'Environment variable %s must contain positive integer row ids.', char(name));
values = unique(values(:), 'stable');
end


function saveFigureBoth(fig, folderPath, baseName)
% Save a figure as PNG and PDF.

ensureFolder(folderPath);
pngPath = fullfile(folderPath, baseName + ".png");
pdfPath = fullfile(folderPath, baseName + ".pdf");
exportgraphics(fig, pngPath, 'Resolution', 220);
exportgraphics(fig, pdfPath, 'ContentType', 'vector');
fprintf('Saved figure: %s\n', pngPath);
fprintf('Saved figure: %s\n', pdfPath);
end


function ensureFolder(folderPath)
% Create a folder if it does not already exist.

if exist(folderPath, 'dir') ~= 7
    mkdir(folderPath);
end
end
