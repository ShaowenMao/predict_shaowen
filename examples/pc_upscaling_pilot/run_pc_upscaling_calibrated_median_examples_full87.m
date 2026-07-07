%RUN_PC_UPSCALING_CALIBRATED_MEDIAN_EXAMPLES_FULL87 Calibrated Pc examples.
%
% This script reuses the exact replayed PREDICT realizations from the
% full-87 median-sand-ratio Pc pilot and computes a calibrated Pc curve for
% each slice/window realization. It is designed as an apples-to-apples
% comparison against the current fast Pc pilot for four representative
% Level-3 permeability cases:
%
%   geology: s05_c012
%   scenario: medium sand, nonuniform
%   geologic case: case_012_zf0500_svcl010_cvcl060
%   Level-3 cases: 01, 03, 04, 07
%
% The implementation follows the calibrated ordinary-upscaling branch of the
% original MRST workflow:
%
%   1. read sand and clay SGOF reference Pc curves from the original deck;
%   2. assign replayed fine cells to material units and smear/sand type;
%   3. scale sand curves using Leverett-style scaling;
%   4. scale clay curves using the GoM clay/mudrock Pce(log10(k)) model;
%   5. pore-volume-average fine/unit saturations at trial capillary pressures;
%   6. choose one medoid Pc curve per window from the 87 slice curves.
%
% This script intentionally does not use the original invasion-percolation
% branch.  That branch is useful as a later connectivity sensitivity test,
% but the ordinary branch gives the cleanest direct comparison to the pilot.

clear; clc;

scriptDir = fileparts(mfilename('fullpath'));
examplesDir = fileparts(scriptDir);
repoRoot = fileparts(examplesDir);

cfg = struct();
cfg.geologyId = "s05_c012";
cfg.level3CaseIds = [1, 3, 4, 7];
cfg.windows = ["famp1", "famp2", "famp3", "famp4", "famp5", "famp6"];
cfg.sourceRoot = fullfile('D:', 'codex_gom', 'UQ_workflow', ...
    'pc_upscaling_median_examples_full87');
cfg.replaySummaryCsv = fullfile(cfg.sourceRoot, 'tables', ...
    'replay_summary_with_full87_context_s05_c012_cases_01_03_04_07.csv');
cfg.pilotCurveMat = fullfile(cfg.sourceRoot, 'curves', ...
    'pc_curves_s05_c012_cases_01_03_04_07_full87.mat');
cfg.upscalingZip = fullfile(repoRoot, 'upscaling.zip');
cfg.outputRoot = fullfile('D:', 'codex_gom', 'UQ_workflow', ...
    'pc_upscaling_calibrated_median_examples_full87');
cfg.curveDir = fullfile(cfg.outputRoot, 'curves');
cfg.tableDir = fullfile(cfg.outputRoot, 'tables');
cfg.figureDir = fullfile(cfg.outputRoot, 'figures');

ensureFolder(cfg.curveDir);
ensureFolder(cfg.tableDir);
ensureFolder(cfg.figureDir);
cfg.originalDeckFile = resolveOriginalDeckFile(cfg);

caseTag = sprintf('%02d_', cfg.level3CaseIds);
caseTag = char("cases_" + string(caseTag(1:end-1)));

fprintf('\n=== Load replay summary for calibrated full-87 Pc upscaling ===\n')
assert(exist(cfg.replaySummaryCsv, 'file') == 2, ...
    'Missing replay summary: %s', cfg.replaySummaryCsv);
replaySummaryAll = readtable(cfg.replaySummaryCsv, 'TextType', 'string');
caseMask = replaySummaryAll.GeologyId == cfg.geologyId & ...
    ismember(replaySummaryAll.Level3CaseId, cfg.level3CaseIds);
replaySummary = replaySummaryAll(caseMask, :);
replaySummary.WindowOrder = windowOrder(replaySummary.Window);
replaySummary = sortrows(replaySummary, ...
    {'Level3CaseId', 'SliceIndex', 'WindowOrder'});

expectedRows = numel(cfg.level3CaseIds) * 87 * numel(cfg.windows);
assert(height(replaySummary) == expectedRows, ...
    'Expected %d replay rows, found %d.', expectedRows, height(replaySummary));
fprintf('Using %d replayed rows for Level-3 cases [%s] from: %s\n', ...
    height(replaySummary), num2str(cfg.level3CaseIds), cfg.replaySummaryCsv);

fprintf('\n=== Read original sand/clay SGOF reference curves ===\n')
pcOpt = calibratedPcOptions(cfg.originalDeckFile);
fprintf('Reference deck: %s\n', cfg.originalDeckFile);
fprintf('Pc scaling uses %s permeability and clay Pce uncertainty quantile %.2f.\n', ...
    pcOpt.scalingPermComponent, pcOpt.clayPceUncertaintyQuantile);

curveLongCsv = fullfile(cfg.curveDir, ...
    sprintf('pc_curve_points_s05_c012_%s_calibrated_full87.csv', caseTag));
curveSummaryCsv = fullfile(cfg.tableDir, ...
    sprintf('pc_curve_summary_s05_c012_%s_calibrated_full87.csv', caseTag));
curveMatFile = fullfile(cfg.curveDir, ...
    sprintf('pc_curves_s05_c012_%s_calibrated_full87.mat', caseTag));

fprintf('\n=== Compute calibrated ordinary-upscaled Pc curves ===\n')
if exist(curveMatFile, 'file') == 2
    fprintf('Loading cached calibrated curve MAT: %s\n', curveMatFile);
    cached = load(curveMatFile, 'curveMat');
    curveMat = cached.curveMat;
else
    [curveLong, curveSummary, curveMat] = computeCalibratedPcCurves( ...
        replaySummary, pcOpt);
    writetable(curveLong, curveLongCsv);
    writetable(curveSummary, curveSummaryCsv);
    save(curveMatFile, 'curveMat', 'pcOpt', 'cfg', '-v7.3');
    fprintf('Saved calibrated curve points: %s\n', curveLongCsv);
    fprintf('Saved calibrated curve summary: %s\n', curveSummaryCsv);
    fprintf('Saved calibrated curve MAT: %s\n', curveMatFile);
end

fprintf('\n=== Select calibrated medoid Pc curves ===\n')
medoidSummaryCsv = fullfile(cfg.tableDir, ...
    sprintf('pc_medoid_summary_s05_c012_%s_calibrated_full87.csv', caseTag));
distanceSummaryCsv = fullfile(cfg.tableDir, ...
    sprintf('pc_distance_summary_s05_c012_%s_calibrated_full87.csv', caseTag));
resultsMatFile = fullfile(cfg.tableDir, ...
    sprintf('pc_medoid_results_s05_c012_%s_calibrated_full87.mat', caseTag));
if exist(resultsMatFile, 'file') == 2
    fprintf('Loading cached medoid results: %s\n', resultsMatFile);
    cached = load(resultsMatFile, 'results');
    results = cached.results;
else
    results = analyzeCalibratedMedoids(curveMat, cfg);
    writetable(results.MedoidSummary, medoidSummaryCsv);
    writetable(results.DistanceSummary, distanceSummaryCsv);
    save(resultsMatFile, 'results', '-v7.3');
    fprintf('Saved medoid summary: %s\n', medoidSummaryCsv);
    fprintf('Saved distance summary: %s\n', distanceSummaryCsv);
end

if exist(cfg.pilotCurveMat, 'file') == 2
    comparisonTable = comparePilotAndCalibratedMedoids(curveMat, results, cfg);
    comparisonCsv = fullfile(cfg.tableDir, ...
        sprintf('pc_medoid_comparison_pilot_vs_calibrated_s05_c012_%s_full87.csv', caseTag));
    writetable(comparisonTable, comparisonCsv);
    fprintf('Saved pilot-vs-calibrated medoid comparison: %s\n', comparisonCsv);
else
    warning('Pilot curve MAT not found, skipping medoid comparison table: %s', ...
        cfg.pilotCurveMat);
end

fprintf('\n=== Generate calibrated and pilot-comparison figures ===\n')
makeCalibratedFigures(curveMat, results, cfg);
if exist(cfg.pilotCurveMat, 'file') == 2
    makePilotComparisonFigures(curveMat, results, cfg);
else
    warning('Pilot curve MAT not found, skipping pilot comparison figure: %s', ...
        cfg.pilotCurveMat);
end

fprintf('\nCalibrated full-87 Pc upscaling complete.\n')
fprintf('Output root: %s\n', cfg.outputRoot);


function pcOpt = calibratedPcOptions(deckFile)
% Return calibrated Pc options and original sand/clay reference curves.

pcOpt = struct();
pcOpt.sgGrid = linspace(0.02, 0.68, 80);
pcOpt.numPressureTrials = 500;
pcOpt.minPoro = 1.0e-4;
pcOpt.minPermMD = 1.0e-9;
pcOpt.mDInM2 = 9.869233e-16;
pcOpt.refPermSandSI = 7.60393535652603e-13;
pcOpt.refPoroSand = 0.289875;
pcOpt.clayPceRmse = 0.2953;
pcOpt.clayPceUncertaintyQuantile = 0.5;
pcOpt.contactAngleDeg = 30;
pcOpt.scalingPermComponent = "kzz";
pcOpt.deckFile = deckFile;
pcOpt.referenceCurves = readSgofReferenceCurves(deckFile);
pcOpt.greyCurveColor = [0.68 0.71 0.76];
pcOpt.medoidColor = [0.86 0.22 0.16];
pcOpt.pilotColor = [0.13 0.38 0.67];
end


function deckFile = resolveOriginalDeckFile(cfg)
% Locate the original upscaling deck, extracting upscaling.zip if needed.

deckRelPath = fullfile('eclipse_data_files', ...
    'gom_forUps_theta30_PVDO_incompRock.DATA');

candidate = fullfile('D:', 'codex_gom', 'tmp_upscaling_zip_inspect', deckRelPath);
if exist(candidate, 'file') == 2
    deckFile = candidate;
    return
end

extractRoot = fullfile(cfg.outputRoot, 'reference_upscaling_zip');
candidate = fullfile(extractRoot, deckRelPath);
if exist(candidate, 'file') ~= 2
    assert(exist(cfg.upscalingZip, 'file') == 2, ...
        ['Missing original upscaling deck and missing upscaling.zip.\n' ...
         'Expected zip file: %s'], cfg.upscalingZip);
    ensureFolder(extractRoot);
    unzip(cfg.upscalingZip, extractRoot);
end

assert(exist(candidate, 'file') == 2, ...
    'Could not locate extracted deck file: %s', candidate);
deckFile = candidate;
end


function curves = readSgofReferenceCurves(deckFile)
% Read the two original SGOF Pc tables from the Eclipse deck.

assert(exist(deckFile, 'file') == 2, 'Missing deck file: %s', deckFile);
lines = string(readlines(deckFile));
sgofLine = find(strtrim(lines) == "SGOF", 1, 'first');
assert(~isempty(sgofLine), 'No SGOF keyword found in %s.', deckFile);

tables = cell(1, 2);
current = [];
tableIndex = 1;
for i = (sgofLine + 1):numel(lines)
    raw = strtrim(lines(i));
    if raw == ""
        continue
    end
    if startsWith(raw, "--")
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

    numericText = erase(raw, "/");
    values = sscanf(numericText, '%f');
    if numel(values) == 4
        current(end+1, :) = values(:)'; %#ok<AGROW>
    elseif ~isempty(current)
        break
    end
end

assert(~isempty(tables{1}) && ~isempty(tables{2}), ...
    'Expected two SGOF tables in %s.', deckFile);

curves = struct();
curves.sand = tableToCurve(tables{1});
curves.clay = tableToCurve(tables{2});
end


function curve = tableToCurve(T)
% Convert one SGOF numeric table to an interpolation-ready Pc curve.

curve.sg = T(:, 1);
curve.krg = T(:, 2);
curve.krog = T(:, 3);
curve.pcBar = T(:, 4);
curve.pcPa = curve.pcBar * 1.0e5;
[curve.pcPaUnique, ia] = unique(curve.pcPa, 'stable');
curve.sgAtPcUnique = curve.sg(ia);
end


function [curveLong, curveSummary, curveMat] = computeCalibratedPcCurves( ...
        replaySummary, pcOpt)
% Compute calibrated ordinary-upscaled Pc curves for every replay row.

n = height(replaySummary);
sg = pcOpt.sgGrid(:)';
pcPa = nan(n, numel(sg));
summaryRows = cell(n, 30);
longRows = cell(n * numel(sg), 17);
longIdx = 0;

for i = 1:n
    outputFile = char(replaySummary.OutputFile(i));
    fprintf('Calibrated Pc curve %3d/%3d: slice %02d %s\n', ...
        i, n, replaySummary.SliceIndex(i), char(replaySummary.Window(i)));
    S = load(outputFile, 'replay');
    curve = calibratedPcCurveFromReplay(S.replay, pcOpt);
    pcPa(i, :) = curve.pcPa;

    summaryRows(i, :) = { ...
        i, replaySummary.SourceRow(i), replaySummary.GeologyId(i), ...
        replaySummary.ScenarioName(i), replaySummary.CaseLabel(i), ...
        replaySummary.Level3CaseId(i), replaySummary.Level3CaseName(i), ...
        replaySummary.Window(i), replaySummary.SliceIndex(i), ...
        replaySummary.AssignedState(i), replaySummary.SamplingPool(i), ...
        replaySummary.SelectedSampleIndex(i), replaySummary.ReplaySeed(i), ...
        curve.poreVolume, curve.numCells, curve.numRegions, ...
        curve.numClayRegions, curve.clayPoreVolumeFraction, ...
        curve.meanLog10KzzMD, curve.medianLog10KzzMD, ...
        curve.p05Log10KzzMD, curve.p95Log10KzzMD, ...
        curve.pcAtSg20Pa, curve.pcAtSg50Pa, curve.pcAtSg65Pa, ...
        curve.bulkSgMax, curve.pcMinPa, curve.pcMaxPa, ...
        pcOpt.scalingPermComponent, pcOpt.clayPceUncertaintyQuantile};

    for j = 1:numel(sg)
        longIdx = longIdx + 1;
        longRows(longIdx, :) = { ...
            i, replaySummary.SourceRow(i), replaySummary.GeologyId(i), ...
            replaySummary.Level3CaseId(i), replaySummary.Level3CaseName(i), ...
            replaySummary.Window(i), replaySummary.SliceIndex(i), ...
            replaySummary.AssignedState(i), replaySummary.SamplingPool(i), ...
            replaySummary.SelectedSampleIndex(i), replaySummary.ReplaySeed(i), ...
            sg(j), curve.pcPa(j), curve.pcPa(j) / 1.0e5, ...
            log10(max(curve.pcPa(j), realmin)), curve.clayPoreVolumeFraction, ...
            curve.medianLog10KzzMD};
    end
end

curveSummary = cell2table(summaryRows, 'VariableNames', ...
    {'CurveId', 'ReplaySourceRow', 'GeologyId', 'ScenarioName', ...
     'CaseLabel', 'Level3CaseId', 'Level3CaseName', 'Window', ...
     'SliceIndex', 'AssignedState', 'SamplingPool', 'SelectedSampleIndex', ...
     'ReplaySeed', 'PoreVolume', 'NumCells', 'NumRegions', ...
     'NumClayRegions', 'ClayPoreVolumeFraction', 'MeanLog10KzzMD', ...
     'MedianLog10KzzMD', 'P05Log10KzzMD', 'P95Log10KzzMD', ...
     'PcAtSg20Pa', 'PcAtSg50Pa', 'PcAtSg65Pa', 'BulkSgMax', ...
     'PcMinPa', 'PcMaxPa', 'ScalingPermComponent', ...
     'ClayPceUncertaintyQuantile'});

curveLong = cell2table(longRows(1:longIdx, :), 'VariableNames', ...
    {'CurveId', 'ReplaySourceRow', 'GeologyId', 'Level3CaseId', ...
     'Level3CaseName', 'Window', 'SliceIndex', 'AssignedState', ...
     'SamplingPool', 'SelectedSampleIndex', 'ReplaySeed', ...
     'GasSaturation', 'PcPa', 'PcBar', 'Log10PcPa', ...
     'ClayPoreVolumeFraction', 'MedianLog10KzzMD'});

curveMat = struct();
curveMat.sgGrid = sg;
curveMat.pcPa = pcPa;
curveMat.pcBar = pcPa / 1.0e5;
curveMat.summary = curveSummary;
end


function curve = calibratedPcCurveFromReplay(replay, pcOpt)
% Compute one ordinary-upscaled calibrated Pc curve from a replay structure.

poroAll = max(replay.Grid.poro(:), pcOpt.minPoro);
perm = replay.Grid.perm;
units = replay.Grid.units(:);
isSmear = logical(replay.Grid.isSmear(:));
volume = cellVolumes(replay, numel(poroAll));

permComponentSI = selectPermComponent(perm, pcOpt);
permComponentSI = max(permComponentSI(:), pcOpt.minPermMD * pcOpt.mDInM2);
log10KzzMD = log10(permComponentSI ./ pcOpt.mDInM2);

valid = isfinite(poroAll) & isfinite(permComponentSI) & ...
    isfinite(volume) & volume > 0 & units > 0;
poro = poroAll(valid);
permComponentSI = permComponentSI(valid);
units = units(valid);
isSmear = isSmear(valid);
volume = volume(valid);
log10KzzMD = log10KzzMD(valid);
poreWeightsCell = poro .* volume;

regionIds = unique(units(:))';
region = struct([]);
for r = 1:numel(regionIds)
    id = units == regionIds(r);
    smearFrac = mean(double(isSmear(id)), 'omitnan');
    region(r).id = regionIds(r);
    region(r).isClay = smearFrac >= 0.5;
    region(r).smearFraction = smearFrac;
    region(r).weight = sum(poreWeightsCell(id), 'omitnan');
    region(r).permSI = mean(permComponentSI(id), 'omitnan');
    region(r).poro = mean(poro(id), 'omitnan');
    region(r).sgRef = regionSgReference(region(r).isClay, pcOpt);
    region(r).pcRefPa = scaledRegionPc(region(r), pcOpt);
    region(r).pcRefPa = makeStrictlyIncreasing(region(r).pcRefPa);
end

regionWeights = [region.weight];
regionWeights = regionWeights ./ sum(regionWeights);
pcMin = minPositiveRegionPc(region);
pcMax = maxRegionPc(region);
pcTrials = logspace(log10(pcMin), log10(0.99 * pcMax), ...
    pcOpt.numPressureTrials);
pcTrials = [0, 0.98 * pcMin, pcTrials];

bulkSg = zeros(size(pcTrials));
for p = 1:numel(pcTrials)
    sgRegion = zeros(1, numel(region));
    for r = 1:numel(region)
        sgRegion(r) = interp1(region(r).pcRefPa, region(r).sgRef, ...
            pcTrials(p), 'linear', 'extrap');
        sgRegion(r) = min(max(sgRegion(r), min(region(r).sgRef)), ...
            max(region(r).sgRef));
    end
    bulkSg(p) = sum(regionWeights .* sgRegion);
end

[bulkSgUnique, ia] = unique(bulkSg, 'stable');
pcUnique = pcTrials(ia);
pcAtSg = interp1(bulkSgUnique, pcUnique, pcOpt.sgGrid, ...
    'linear', 'extrap');
pcAtSg = max(pcAtSg, realmin);

curve = struct();
curve.pcPa = pcAtSg;
curve.poreVolume = sum(poreWeightsCell, 'omitnan');
curve.numCells = numel(poreWeightsCell);
curve.numRegions = numel(region);
curve.numClayRegions = sum([region.isClay]);
curve.clayPoreVolumeFraction = sum(regionWeights([region.isClay]));
curve.meanLog10KzzMD = mean(log10KzzMD, 'omitnan');
curve.medianLog10KzzMD = median(log10KzzMD, 'omitnan');
curve.p05Log10KzzMD = prctile(log10KzzMD, 5);
curve.p95Log10KzzMD = prctile(log10KzzMD, 95);
curve.pcAtSg20Pa = interp1(pcOpt.sgGrid, pcAtSg, 0.20, 'linear', 'extrap');
curve.pcAtSg50Pa = interp1(pcOpt.sgGrid, pcAtSg, 0.50, 'linear', 'extrap');
curve.pcAtSg65Pa = interp1(pcOpt.sgGrid, pcAtSg, 0.65, 'linear', 'extrap');
curve.bulkSgMax = max(bulkSgUnique);
curve.pcMinPa = pcMin;
curve.pcMaxPa = pcMax;
end


function volume = cellVolumes(replay, nCells)
% Return MRST cell volumes when present; otherwise use unit volumes.

if isfield(replay, 'G') && isfield(replay.G, 'cells') && ...
        isfield(replay.G.cells, 'volumes') && ...
        numel(replay.G.cells.volumes) == nCells
    volume = replay.G.cells.volumes(:);
else
    volume = ones(nCells, 1);
end
end


function kSI = selectPermComponent(perm, pcOpt)
% Select the permeability component used by the original Pc scaling.

switch lower(string(pcOpt.scalingPermComponent))
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
        error('Unknown permeability component: %s', pcOpt.scalingPermComponent);
end
kSI = perm(:, col);
end


function sg = regionSgReference(isClay, pcOpt)
% Return the reference gas-saturation grid for sand or clay.

if isClay
    sg = pcOpt.referenceCurves.clay.sg;
else
    sg = pcOpt.referenceCurves.sand.sg;
end
end


function pcPa = scaledRegionPc(region, pcOpt)
% Scale one sand or clay reference Pc curve following the original workflow.

if region.isClay
    ref = pcOpt.referenceCurves.clay;
    log10KMD = log10(max(region.permSI / pcOpt.mDInM2, pcOpt.minPermMD));
    pceHgBar = 10.^(-0.1992 * log10KMD + 1.407 - pcOpt.clayPceRmse + ...
        pcOpt.clayPceUncertaintyQuantile * 2 * pcOpt.clayPceRmse);
    pceCo2WaterPa = 1.0e5 * pceHgBar * ...
        abs(cosd(pcOpt.contactAngleDeg) * 25 / (cosd(140) * 485));
    refPcAtSg10 = interp1(ref.sg, ref.pcPa, 0.10, 'linear', 'extrap');
    pcPa = ref.pcPa .* (pceCo2WaterPa / refPcAtSg10);
else
    ref = pcOpt.referenceCurves.sand;
    scale = sqrt((pcOpt.refPermSandSI * region.poro) ./ ...
        (pcOpt.refPoroSand * region.permSI));
    pcPa = ref.pcPa .* scale;
end
end


function pc = makeStrictlyIncreasing(pc)
% Small monotonicity repair so inverse interpolation is stable.

pc = pc(:)';
for i = 2:numel(pc)
    if pc(i) <= pc(i-1)
        pc(i) = pc(i-1) + max(abs(pc(i-1)), 1.0) * 1.0e-12;
    end
end
end


function pcMin = minPositiveRegionPc(region)
% Return the smallest positive reference Pc across regions.

mins = nan(1, numel(region));
for r = 1:numel(region)
    p = region(r).pcRefPa(region(r).pcRefPa > 0);
    mins(r) = min(p);
end
pcMin = min(mins, [], 'omitnan');
pcMin = max(pcMin, realmin);
end


function pcMax = maxRegionPc(region)
% Return the largest Pc across regions.

maxes = nan(1, numel(region));
for r = 1:numel(region)
    maxes(r) = max(region(r).pcRefPa);
end
pcMax = max(maxes, [], 'omitnan');
pcMax = max(pcMax, 10 * minPositiveRegionPc(region));
end


function results = analyzeCalibratedMedoids(curveMat, cfg)
% Select one medoid curve per window using RMS log10(Pc) distance.

summary = curveMat.summary;
windows = cfg.windows;
caseIds = cfg.level3CaseIds;
medoidRows = {};
distanceRows = {};

for c = 1:numel(caseIds)
    caseId = caseIds(c);
    for w = 1:numel(windows)
        windowName = windows(w);
        idx = find(summary.Level3CaseId == caseId & summary.Window == windowName);
        assert(numel(idx) == 87, ...
            'Expected 87 curves for case %02d %s, found %d.', ...
            caseId, windowName, numel(idx));

        curves = curveMat.pcPa(idx, :);
        logCurves = log10(max(curves, realmin));
        distances = pairwiseRmsDistance(logCurves);
        meanDistance = mean(distances, 2, 'omitnan');
        [minMeanDistance, localMedoid] = min(meanDistance);
        medoidCurveId = idx(localMedoid);
        upper = distances(triu(true(size(distances)), 1));

        medoidRows(end+1, :) = { ...
            summary.GeologyId(idx(1)), caseId, ...
            summary.Level3CaseName(idx(1)), windowName, medoidCurveId, ...
            summary.SliceIndex(medoidCurveId), ...
            summary.SelectedSampleIndex(medoidCurveId), ...
            summary.ReplaySeed(medoidCurveId), ...
            minMeanDistance, median(upper, 'omitnan'), ...
            prctile(upper, 95), summary.PcAtSg20Pa(medoidCurveId), ...
            summary.PcAtSg50Pa(medoidCurveId), ...
            summary.PcAtSg65Pa(medoidCurveId), ...
            summary.ClayPoreVolumeFraction(medoidCurveId), ...
            summary.MedianLog10KzzMD(medoidCurveId)};

        for j = 1:numel(idx)
            distanceRows(end+1, :) = { ...
                summary.GeologyId(idx(j)), caseId, ...
                summary.Level3CaseName(idx(j)), windowName, ...
                summary.SliceIndex(idx(j)), summary.SelectedSampleIndex(idx(j)), ...
                summary.ReplaySeed(idx(j)), idx(j), medoidCurveId, ...
                distances(j, localMedoid), meanDistance(j)};
        end
    end
end

results = struct();
results.MedoidSummary = cell2table(medoidRows, 'VariableNames', ...
    {'GeologyId', 'Level3CaseId', 'Level3CaseName', 'Window', ...
     'MedoidCurveId', 'MedoidSliceIndex', 'MedoidSelectedSampleIndex', ...
     'MedoidReplaySeed', 'MedoidMeanRmsLog10PcDistance', ...
     'MedianPairwiseRmsLog10PcDistance', 'P95PairwiseRmsLog10PcDistance', ...
     'MedoidPcAtSg20Pa', 'MedoidPcAtSg50Pa', 'MedoidPcAtSg65Pa', ...
     'MedoidClayPoreVolumeFraction', 'MedoidMedianLog10KzzMD'});
results.DistanceSummary = cell2table(distanceRows, 'VariableNames', ...
    {'GeologyId', 'Level3CaseId', 'Level3CaseName', 'Window', ...
     'SliceIndex', 'SelectedSampleIndex', 'ReplaySeed', 'CurveId', ...
     'MedoidCurveId', 'RmsLog10PcDistanceToMedoid', ...
     'MeanRmsLog10PcDistanceToAllCurves'});
end


function D = pairwiseRmsDistance(logCurves)
% Pairwise RMS distance between rows of a log-curve matrix.

n = size(logCurves, 1);
D = zeros(n, n);
for i = 1:n
    for j = (i+1):n
        diffv = logCurves(i, :) - logCurves(j, :);
        d = sqrt(mean(diffv.^2, 'omitnan'));
        D(i, j) = d;
        D(j, i) = d;
    end
end
end


function makeCalibratedFigures(curveMat, results, cfg)
% Plot calibrated full-87 curves and medoids for each Level-3 case.

figureDir = cfg.figureDir;
summary = curveMat.summary;
windows = cfg.windows;
caseIds = cfg.level3CaseIds;
sg = curveMat.sgGrid;

for c = 1:numel(caseIds)
    caseId = caseIds(c);
    caseMask = summary.Level3CaseId == caseId;
    caseName = summary.Level3CaseName(find(caseMask, 1, 'first'));

    fig = figure('Color', 'w', 'Position', [80 80 1700 900]);
    tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    for w = 1:numel(windows)
        nexttile;
        windowName = windows(w);
        idx = find(caseMask & summary.Window == windowName);
        medoidMask = results.MedoidSummary.Level3CaseId == caseId & ...
            results.MedoidSummary.Window == windowName;
        medoidId = results.MedoidSummary.MedoidCurveId(medoidMask);
        semilogy(sg, curveMat.pcBar(idx, :)', '-', ...
            'Color', [0.72 0.74 0.78], 'LineWidth', 0.8);
        hold on;
        semilogy(sg, curveMat.pcBar(medoidId, :), '-', ...
            'Color', [0.86 0.22 0.16], 'LineWidth', 2.8);
        grid on;
        title(sprintf('%s | calibrated ordinary', upper(char(windowName))), ...
            'FontSize', 16, 'FontWeight', 'bold');
        xlabel('Gas saturation');
        ylabel('Pc [bar]');
        xlim([min(sg), max(sg)]);
        set(gca, 'FontSize', 13, 'LineWidth', 1.0);
    end
    sgtitle({sprintf('Case %02d full-87 calibrated Pc curves by window', caseId), ...
        sprintf('%s | Grey = 87 slices; red = medoid curve', caseName)}, ...
        'FontSize', 22, 'FontWeight', 'bold', 'Interpreter', 'none');
    saveFigureBoth(fig, figureDir, ...
        sprintf('s05_c012_case%02d_calibrated_full87_pc_curves_with_medoids', caseId));

    medoidCase = results.MedoidSummary(results.MedoidSummary.Level3CaseId == caseId, :);
    fig = figure('Color', 'w', 'Position', [120 120 1400 720]);
    barData = [medoidCase.MedoidPcAtSg20Pa, ...
               medoidCase.MedoidPcAtSg50Pa, ...
               medoidCase.MedoidPcAtSg65Pa] ./ 1.0e5;
    bar(categorical(medoidCase.Window), barData);
    grid on;
    ylabel('Medoid Pc [bar]');
    legend({'Sg = 0.20', 'Sg = 0.50', 'Sg = 0.65'}, ...
        'Location', 'northwest');
    title(sprintf('Case %02d calibrated medoid Pc levels by window', caseId), ...
        'FontSize', 20, 'FontWeight', 'bold');
    subtitle(caseName, 'Interpreter', 'none');
    set(gca, 'FontSize', 15, 'LineWidth', 1.0);
    saveFigureBoth(fig, figureDir, ...
        sprintf('s05_c012_case%02d_calibrated_medoid_pc_levels', caseId));
end
end


function comparisonTable = comparePilotAndCalibratedMedoids(calCurveMat, calResults, cfg)
% Build a per-window table comparing pilot and calibrated medoid curves.

P = load(cfg.pilotCurveMat, 'curveMat');
pilotMat = P.curveMat;
pilotSummary = pilotMat.summary;
pilotMask = ismember(pilotSummary.Level3CaseId, cfg.level3CaseIds);
pilotSummary = pilotSummary(pilotMask, :);
pilotPcPa = pilotMat.pcPa(pilotMask, :);
pilotSg = pilotMat.sgGrid;

rows = cell(numel(cfg.level3CaseIds) * numel(cfg.windows), 17);
row = 0;
for c = 1:numel(cfg.level3CaseIds)
    caseId = cfg.level3CaseIds(c);
    for w = 1:numel(cfg.windows)
        windowName = cfg.windows(w);

        calMask = calResults.MedoidSummary.Level3CaseId == caseId & ...
            calResults.MedoidSummary.Window == windowName;
        calMedoidId = calResults.MedoidSummary.MedoidCurveId(calMask);
        calSummary = calCurveMat.summary(calMedoidId, :);
        calCurve = calCurveMat.pcPa(calMedoidId, :);
        calSg = calCurveMat.sgGrid;
        calNorm = normalizeAtSg(calSg, calCurve, 0.50);

        pidx = find(pilotSummary.Level3CaseId == caseId & ...
            pilotSummary.Window == windowName);
        pilotLog = log10(max(pilotPcPa(pidx, :), realmin));
        D = pairwiseRmsDistance(pilotLog);
        [~, localMedoid] = min(mean(D, 2, 'omitnan'));
        pilotGlobalId = pidx(localMedoid);
        pilotCurve = pilotPcPa(pilotGlobalId, :);
        pilotInterp = interp1(pilotSg, pilotCurve, calSg, 'linear', 'extrap');
        pilotNorm = normalizeAtSg(calSg, pilotInterp, 0.50);

        shapeDiff = sqrt(mean((log10(max(calNorm, realmin)) - ...
            log10(max(pilotNorm, realmin))).^2, 'omitnan'));
        sameSlice = pilotSummary.SliceIndex(pilotGlobalId) == calSummary.SliceIndex;
        sameSample = pilotSummary.SelectedSampleIndex(pilotGlobalId) == ...
            calSummary.SelectedSampleIndex;

        row = row + 1;
        rows(row, :) = { ...
            caseId, calSummary.Level3CaseName, windowName, ...
            calSummary.SliceIndex, calSummary.SelectedSampleIndex, ...
            calSummary.ReplaySeed, calSummary.PcAtSg20Pa / 1.0e5, ...
            calSummary.PcAtSg50Pa / 1.0e5, calSummary.PcAtSg65Pa / 1.0e5, ...
            calSummary.ClayPoreVolumeFraction, calSummary.MedianLog10KzzMD, ...
            pilotSummary.SliceIndex(pilotGlobalId), ...
            pilotSummary.SelectedSampleIndex(pilotGlobalId), ...
            pilotSummary.ReplaySeed(pilotGlobalId), ...
            shapeDiff, sameSlice, sameSample};
    end
end

comparisonTable = cell2table(rows, 'VariableNames', ...
    {'Level3CaseId', 'Level3CaseName', 'Window', 'CalibratedMedoidSliceIndex', ...
     'CalibratedMedoidSelectedSampleIndex', 'CalibratedMedoidReplaySeed', ...
     'CalibratedMedoidPcSg20Bar', 'CalibratedMedoidPcSg50Bar', ...
     'CalibratedMedoidPcSg65Bar', 'CalibratedMedoidClayPoreVolumeFraction', ...
     'CalibratedMedoidMedianLog10KzzMD', 'PilotMedoidSliceIndex', ...
     'PilotMedoidSelectedSampleIndex', 'PilotMedoidReplaySeed', ...
     'RmsLog10NormalizedPcShapeDifference', 'SameMedoidSlice', ...
     'SameMedoidPredictSample'});
end


function makePilotComparisonFigures(calCurveMat, calResults, cfg)
% Compare calibrated medoid curves with current pilot medoid curves.

P = load(cfg.pilotCurveMat, 'curveMat');
pilotMat = P.curveMat;
pilotSummary = pilotMat.summary;
pilotMask = ismember(pilotSummary.Level3CaseId, cfg.level3CaseIds);
pilotSummary = pilotSummary(pilotMask, :);
pilotPcPa = pilotMat.pcPa(pilotMask, :);
pilotSg = pilotMat.sgGrid;

for c = 1:numel(cfg.level3CaseIds)
    caseId = cfg.level3CaseIds(c);
    caseMask = calCurveMat.summary.Level3CaseId == caseId;
    caseName = calCurveMat.summary.Level3CaseName(find(caseMask, 1, 'first'));

    fig = figure('Color', 'w', 'Position', [80 80 1700 900]);
    tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    for w = 1:numel(cfg.windows)
        nexttile;
        windowName = cfg.windows(w);
        calMask = calResults.MedoidSummary.Level3CaseId == caseId & ...
            calResults.MedoidSummary.Window == windowName;
        calMedoidId = calResults.MedoidSummary.MedoidCurveId(calMask);
        calCurve = calCurveMat.pcPa(calMedoidId, :);
        calSg = calCurveMat.sgGrid;
        calNorm = normalizeAtSg(calSg, calCurve, 0.50);

        pidx = find(pilotSummary.Level3CaseId == caseId & ...
            pilotSummary.Window == windowName);
        pilotLog = log10(max(pilotPcPa(pidx, :), realmin));
        D = pairwiseRmsDistance(pilotLog);
        [~, localMedoid] = min(mean(D, 2, 'omitnan'));
        pilotCurve = pilotPcPa(pidx(localMedoid), :);
        pilotInterp = interp1(pilotSg, pilotCurve, calSg, 'linear', 'extrap');
        pilotNorm = normalizeAtSg(calSg, pilotInterp, 0.50);

        semilogy(calSg, pilotNorm, '--', 'Color', [0.13 0.38 0.67], ...
            'LineWidth', 2.4);
        hold on;
        semilogy(calSg, calNorm, '-', 'Color', [0.86 0.22 0.16], ...
            'LineWidth', 2.6);
        grid on;
        title(upper(char(windowName)), 'FontSize', 16, 'FontWeight', 'bold');
        xlabel('Gas saturation');
        ylabel('Pc / Pc(Sg=0.50)');
        xlim([min(calSg), max(calSg)]);
        set(gca, 'FontSize', 13, 'LineWidth', 1.0);
        if w == 1
            legend({'current pilot', 'calibrated'}, 'Location', 'best');
        end
    end
    sgtitle({sprintf('Case %02d medoid Pc curve shape comparison', caseId), ...
        sprintf('%s | normalized by Pc at Sg = 0.50; pilot absolute Pc is not calibrated', caseName)}, ...
        'FontSize', 21, 'FontWeight', 'bold', 'Interpreter', 'none');
    saveFigureBoth(fig, cfg.figureDir, ...
        sprintf('s05_c012_case%02d_pilot_vs_calibrated_medoid_pc_shape', caseId));
end
end


function y = normalizeAtSg(sg, pc, sgRef)
% Normalize a Pc curve by its value at a reference saturation.

ref = interp1(sg, pc, sgRef, 'linear', 'extrap');
y = pc ./ max(ref, realmin);
y = max(y, realmin);
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
% Create folder if it does not already exist.

if exist(folderPath, 'dir') ~= 7
    mkdir(folderPath);
end
end


function saveFigureBoth(fig, folderPath, baseName)
% Save one figure as PNG and PDF.

ensureFolder(folderPath);
try
    set(fig, 'DefaultAxesToolbarVisible', 'off');
catch
end
axesList = findall(fig, 'Type', 'axes');
for i = 1:numel(axesList)
    try
        disableDefaultInteractivity(axesList(i));
    catch
    end
    try
        axtoolbar(axesList(i), 'none');
    catch
    end
end
try
    delete(findall(fig, 'Type', 'AxesToolbar'));
catch
end
try
    delete(findall(fig, 'Tag', 'AxesToolbar'));
catch
end
drawnow;
pngFile = fullfile(folderPath, baseName + ".png");
pdfFile = fullfile(folderPath, baseName + ".pdf");
exportgraphics(fig, pngFile, 'Resolution', 220);
exportgraphics(fig, pdfFile, 'ContentType', 'vector');
fprintf('Saved figure: %s\n', pngFile);
fprintf('Saved figure: %s\n', pdfFile);
end
