%RUN_PC_UPSCALING_PILOT_S05_C012_CASE07 Pc-medoid pilot for one Level-3 case.
%
% This script tests two candidate workflows for choosing representative
% capillary-pressure curves from the sampled Texas offshore fault
% permeability fields:
%
%   1. Rigorous W6 benchmark:
%      replay all 87 selected W6 slice realizations, compute one effective
%      Pc curve per slice, and choose the W6 Pc medoid curve.
%
%   2. Reduced six-window test:
%      replay 10 common along-strike slices for each of W1-W6, compute one
%      effective Pc curve per replayed realization, and choose one medoid
%      curve per window.
%
% The selected case is the previously preferred grouped mixed low/high
% example:
%
%   s05_c012, Level-3 case 7
%   medium sand, nonuniform | fault depth 500 m | sand Vcl 0.1 | clay Vcl 0.6
%
% The effective Pc curve is a screening capillary-equilibrium upscaling:
% fine cells receive Brooks-Corey/Leverett-style entry pressures scaled by
% sqrt(phi/k), the bulk saturation is pore-volume averaged at each trial
% capillary pressure, and the result is interpolated to a common gas
% saturation grid. The absolute pressure scale is configurable; medoid
% selection uses normalized curve shape in log space.

clear; clc;

repoRoot = fileparts(fileparts(mfilename('fullpath')));
scriptDir = fileparts(mfilename('fullpath'));
examplesDir = fileparts(scriptDir);
repoRoot = fileparts(examplesDir);
addpath(examplesDir);

setupMrstIfNeeded();

outputRoot = fullfile('D:', 'codex_gom', 'UQ_workflow', 'pc_upscaling_pilot');
inputDir = fullfile(outputRoot, 'inputs');
replayDir = fullfile(outputRoot, 'replay_s05_c012_case07_unique137');
curveDir = fullfile(outputRoot, 'curves');
figureDir = fullfile(outputRoot, 'figures');
tableDir = fullfile(outputRoot, 'tables');
ensureFolder(replayDir);
ensureFolder(curveDir);
ensureFolder(figureDir);
ensureFolder(tableDir);

dataRoot = fullfile(examplesDir, 'thickness_scenario_data');
selectionCsv = fullfile(inputDir, 's05_c012_case07_unique_replay_rows_137.csv');
selectionCsv = buildUniqueReplaySelection(inputDir, selectionCsv);

fprintf('\n=== Replay exact selected PREDICT realizations ===\n')
replaySummaryCsv = fullfile(replayDir, 'replay_summary.csv');
if exist(replaySummaryCsv, 'file') == 2
    fprintf('Using existing replay summary: %s\n', replaySummaryCsv);
    replaySummary = readtable(replaySummaryCsv, 'TextType', 'string');
else
    replaySummary = replay_selected_predict_realizations( ...
        selectionCsv, replayDir, ...
        'DataRoot', dataRoot, ...
        'MaxRows', inf);
end

selectionTable = readtable(selectionCsv, 'TextType', 'string');
replaySummary = attachSelectionContext(replaySummary, selectionTable);

fprintf('\n=== Compute effective Pc curves ===\n')
pcOpt = defaultPcOptions();
[curveLong, curveSummary, curveMat] = computePcCurves(replaySummary, pcOpt);

curveLongCsv = fullfile(curveDir, 'pc_curve_points_s05_c012_case07.csv');
curveSummaryCsv = fullfile(tableDir, 'pc_curve_summary_s05_c012_case07.csv');
curveMatFile = fullfile(curveDir, 'pc_curves_s05_c012_case07.mat');
writetable(curveLong, curveLongCsv);
writetable(curveSummary, curveSummaryCsv);
save(curveMatFile, 'curveMat', 'pcOpt', '-v7.3');

fprintf('Saved Pc curve points: %s\n', curveLongCsv);
fprintf('Saved Pc curve summary: %s\n', curveSummaryCsv);
fprintf('Saved Pc curve MAT: %s\n', curveMatFile);

fprintf('\n=== Select medoid Pc curves ===\n')
results = analyzePcMedoids(curveMat, pcOpt);
medoidSummaryCsv = fullfile(tableDir, 'pc_medoid_summary_s05_c012_case07.csv');
distanceSummaryCsv = fullfile(tableDir, 'pc_distance_summary_s05_c012_case07.csv');
writetable(results.MedoidSummary, medoidSummaryCsv);
writetable(results.DistanceSummary, distanceSummaryCsv);
save(fullfile(tableDir, 'pc_medoid_results_s05_c012_case07.mat'), 'results', '-v7.3');

fprintf('Saved medoid summary: %s\n', medoidSummaryCsv);
fprintf('Saved distance summary: %s\n', distanceSummaryCsv);

fprintf('\n=== Generate review figures ===\n')
makePcPilotFigures(curveMat, results, figureDir);

fprintf('\nPc pilot complete.\n')
fprintf('Output root: %s\n', outputRoot);


function setupMrstIfNeeded()
% Start MRST if it is available locally and not already on the path.

if exist('mrstModule', 'file') == 2
    return
end

candidateRoots = { ...
    fullfile('C:', 'Users', 'Shaow', 'OneDrive', 'MIT', 'mrst-2025a', ...
             'SINTEF-AppliedCompSci-MRST-75749fa'), ...
    fullfile('C:', 'Users', 'Shaow', 'OneDrive', 'MIT', 'mrst-2025a')};

for i = 1:numel(candidateRoots)
    startupFile = fullfile(candidateRoots{i}, 'startup.m');
    if exist(startupFile, 'file') == 2
        run(startupFile);
        return
    end
end

warning(['MRST startup.m was not found automatically. The replay step ' ...
         'will fail unless MRST is already on the MATLAB path.']);
end


function selectionCsv = buildUniqueReplaySelection(inputDir, selectionCsv)
% Combine the W6 full benchmark and common-10 test into one unique replay set.

w6File = fullfile(inputDir, 's05_c012_case07_w6_all87.csv');
commonFile = fullfile(inputDir, 's05_c012_case07_all_windows_common10slices.csv');
assert(exist(w6File, 'file') == 2, 'Missing W6 selection file: %s', w6File);
assert(exist(commonFile, 'file') == 2, 'Missing common-10 selection file: %s', commonFile);

w6Rows = readtable(w6File, 'TextType', 'string');
w6Rows.PilotSet = repmat("w6_all87", height(w6Rows), 1);
commonRows = readtable(commonFile, 'TextType', 'string');
commonRows.PilotSet = repmat("common10", height(commonRows), 1);

allRows = [w6Rows; commonRows];
keys = strcat(allRows.geology_id, "|", string(allRows.case_id), "|", ...
              string(allRows.slice_index), "|", allRows.window, "|", ...
              string(allRows.selected_sample_index));
[~, keep] = unique(keys, 'stable');
uniqueRows = allRows(keep, :);

% Keep an explicit flag for membership in either test, since the W6 common
% slices are intentionally shared by both pilots.
uniqueRows.InW6All87 = uniqueRows.window == "famp6";
commonKeys = strcat(commonRows.geology_id, "|", string(commonRows.case_id), "|", ...
                    string(commonRows.slice_index), "|", commonRows.window, "|", ...
                    string(commonRows.selected_sample_index));
uniqueRows.InCommon10 = ismember(keys(keep), commonKeys);

writetable(uniqueRows, selectionCsv);
fprintf('Prepared unique replay selection: %s (%d rows)\n', ...
        selectionCsv, height(uniqueRows));
end


function T = attachSelectionContext(T, selectionTable)
% Add slice/window context from the pilot selection table to replay summary.

assert(height(T) == height(selectionTable), ...
       'Replay summary and selection table have different row counts.');

T.GeologyId = selectionTable.geology_id;
T.ScenarioLabel = selectionTable.scenario_label;
T.ScenarioName = selectionTable.scenario_name;
T.CaseLabel = selectionTable.case_label;
T.Level3CaseId = str2double(string(selectionTable.case_id));
T.Level3CaseName = selectionTable.case_name;
T.Window = selectionTable.window;
T.SliceIndex = str2double(string(selectionTable.slice_index));
T.DrawGroupIndex = str2double(string(selectionTable.draw_group_index));
T.AssignedState = selectionTable.assigned_state;
T.SamplingMode = selectionTable.sampling_mode;
T.SamplingPool = selectionTable.sampling_pool;
T.SelectedSampleIndex = str2double(string(selectionTable.selected_sample_index));
T.InW6All87 = tableColumnToLogical(selectionTable.InW6All87);
T.InCommon10 = tableColumnToLogical(selectionTable.InCommon10);
end


function values = tableColumnToLogical(column)
% Convert a CSV/table column containing logical flags to a logical vector.

if islogical(column)
    values = column;
elseif isnumeric(column)
    values = column ~= 0;
else
    text = lower(strtrim(string(column)));
    values = text == "true" | text == "1";
end
end


function pcOpt = defaultPcOptions()
% Default screening Pc model parameters.

pcOpt.saturationName = "gas_saturation";
pcOpt.sgGrid = linspace(0.02, 0.98, 80);
pcOpt.numPressureTrials = 500;
pcOpt.brooksCoreyLambda = 2.0;
pcOpt.pcRefPa = 1.0e4;
pcOpt.minPoro = 1.0e-4;
pcOpt.minPermMD = 1.0e-9;
pcOpt.mDInM2 = 9.869233e-16;
pcOpt.curveDistanceSpace = "log10_normalized_pc";
pcOpt.common10Color = [0.13 0.40 0.74];
pcOpt.fullMedoidColor = [0.02 0.02 0.02];
pcOpt.reducedMedoidColor = [0.92 0.43 0.08];
end


function [curveLong, curveSummary, curveMat] = computePcCurves(replaySummary, pcOpt)
% Compute effective Pc curves for every replayed realization.

n = height(replaySummary);
sg = pcOpt.sgGrid(:)';
pcNorm = nan(n, numel(sg));
pcPa = nan(n, numel(sg));

summaryRows = cell(n, 19);
longRows = cell(n * numel(sg), 13);
longIdx = 0;

for i = 1:n
    outputFile = char(replaySummary.OutputFile(i));
    fprintf('Pc curve %3d/%3d: %s slice %d window %s\n', ...
            i, n, char(replaySummary.GeologyId(i)), ...
            replaySummary.SliceIndex(i), char(replaySummary.Window(i)));

    S = load(outputFile, 'replay', 'verification');
    replay = S.replay;
    curve = effectivePcCurveFromReplay(replay, pcOpt);
    pcNorm(i, :) = curve.pcNormalized;
    pcPa(i, :) = curve.pcPa;

    summaryRows(i, :) = { ...
        i, replaySummary.SourceRow(i), replaySummary.GeologyId(i), ...
        replaySummary.ScenarioName(i), replaySummary.CaseLabel(i), ...
        replaySummary.Level3CaseId(i), replaySummary.Level3CaseName(i), ...
        replaySummary.Window(i), replaySummary.SliceIndex(i), ...
        replaySummary.AssignedState(i), replaySummary.SamplingPool(i), ...
        replaySummary.SelectedSampleIndex(i), replaySummary.ReplaySeed(i), ...
        replaySummary.InW6All87(i), replaySummary.InCommon10(i), ...
        curve.poreVolume, curve.medianEntryPressure, ...
        curve.p05EntryPressure, curve.p95EntryPressure};

    for j = 1:numel(sg)
        longIdx = longIdx + 1;
        longRows(longIdx, :) = { ...
            i, replaySummary.GeologyId(i), replaySummary.Level3CaseId(i), ...
            replaySummary.Window(i), replaySummary.SliceIndex(i), ...
            replaySummary.AssignedState(i), replaySummary.SamplingPool(i), ...
            replaySummary.InW6All87(i), replaySummary.InCommon10(i), ...
            sg(j), pcNorm(i, j), pcPa(i, j), log10(pcNorm(i, j))};
    end
end

curveSummary = cell2table(summaryRows, 'VariableNames', ...
    {'CurveId', 'ReplaySourceRow', 'GeologyId', 'ScenarioName', ...
     'CaseLabel', 'Level3CaseId', 'Level3CaseName', 'Window', ...
     'SliceIndex', 'AssignedState', 'SamplingPool', 'SelectedSampleIndex', ...
     'ReplaySeed', 'InW6All87', 'InCommon10', 'PoreVolume', ...
     'MedianEntryPressureNorm', 'P05EntryPressureNorm', 'P95EntryPressureNorm'});

curveLong = cell2table(longRows, 'VariableNames', ...
    {'CurveId', 'GeologyId', 'Level3CaseId', 'Window', 'SliceIndex', ...
     'AssignedState', 'SamplingPool', 'InW6All87', 'InCommon10', ...
     'GasSaturation', 'PcNormalized', 'PcPa', 'Log10PcNormalized'});

curveMat = struct();
curveMat.sgGrid = sg;
curveMat.pcNormalized = pcNorm;
curveMat.pcPa = pcPa;
curveMat.summary = curveSummary;
end


function curve = effectivePcCurveFromReplay(replay, pcOpt)
% Upscale fine-cell Pc by pore-volume averaging at trial pressures.

poro = replay.Grid.poro(:);
perm = replay.Grid.perm;

if size(perm, 2) >= 6
    kSI = perm(:, [1 4 6]);
elseif size(perm, 2) >= 3
    kSI = perm(:, 1:3);
else
    kSI = repmat(perm(:, 1), 1, 3);
end

kSI = max(kSI, pcOpt.minPermMD * pcOpt.mDInM2);
kCharMD = exp(mean(log(kSI ./ pcOpt.mDInM2), 2));
kCharMD = max(kCharMD, pcOpt.minPermMD);
poro = max(poro, pcOpt.minPoro);

if isfield(replay, 'G') && isfield(replay.G, 'cells') && ...
        isfield(replay.G.cells, 'volumes') && numel(replay.G.cells.volumes) == numel(poro)
    bulkVolume = replay.G.cells.volumes(:);
else
    bulkVolume = ones(size(poro));
end

poreWeights = poro .* bulkVolume;
valid = isfinite(poreWeights) & poreWeights > 0 & all(isfinite(kSI), 2);
poreWeights = poreWeights(valid);
poro = poro(valid);
kCharMD = kCharMD(valid);

entry = sqrt(poro ./ kCharMD);
entry = entry ./ median(entry, 'omitnan');
entry = max(entry, 1.0e-12);
poreWeights = poreWeights ./ sum(poreWeights);

lambda = pcOpt.brooksCoreyLambda;
pcMin = max(min(entry) * 0.5, 1.0e-12);
pcMax = max(entry) / (1 - max(pcOpt.sgGrid))^(1 / lambda) * 2.0;
pcTrials = logspace(log10(pcMin), log10(pcMax), pcOpt.numPressureTrials);

bulkSg = zeros(size(pcTrials));
for i = 1:numel(pcTrials)
    sgCell = 1 - (entry ./ pcTrials(i)).^lambda;
    sgCell = min(max(sgCell, 0), 0.999);
    bulkSg(i) = sum(poreWeights .* sgCell);
end

[bulkSgUnique, ia] = unique(bulkSg, 'stable');
pcUnique = pcTrials(ia);
if bulkSgUnique(1) > 0
    bulkSgUnique = [0, bulkSgUnique];
    pcUnique = [pcTrials(1), pcUnique];
end
if bulkSgUnique(end) < max(pcOpt.sgGrid)
    bulkSgUnique = [bulkSgUnique, max(pcOpt.sgGrid)];
    pcUnique = [pcUnique, pcTrials(end)];
end

pcNormalized = interp1(bulkSgUnique, pcUnique, pcOpt.sgGrid, 'linear', 'extrap');
pcNormalized = max(pcNormalized, 1.0e-12);

curve = struct();
curve.pcNormalized = pcNormalized;
curve.pcPa = pcNormalized .* pcOpt.pcRefPa;
curve.poreVolume = sum(poro .* bulkVolume(valid));
curve.medianEntryPressure = median(entry, 'omitnan');
curve.p05EntryPressure = prctile(entry, 5);
curve.p95EntryPressure = prctile(entry, 95);
end


function results = analyzePcMedoids(curveMat, pcOpt)
% Select medoid curves for the W6 full and common-10 tests.

summary = curveMat.summary;
windows = ["famp1", "famp2", "famp3", "famp4", "famp5", "famp6"];

medoidRows = {};
distanceRows = {};

w6FullMask = summary.Window == "famp6" & summary.InW6All87;
[w6FullMedoid, w6FullStats, w6FullDistances] = medoidForMask(curveMat, w6FullMask);
medoidRows(end+1, :) = medoidRow("w6_all87", "famp6", w6FullMedoid, w6FullStats, summary); %#ok<AGROW>
distanceRows(end+1, :) = distanceRow("w6_all87", "famp6", w6FullStats); %#ok<AGROW>

commonMedoids = containers.Map();
for i = 1:numel(windows)
    w = windows(i);
    mask = summary.Window == w & summary.InCommon10;
    [m, stats] = medoidForMask(curveMat, mask);
    commonMedoids(char(w)) = m;
    medoidRows(end+1, :) = medoidRow("common10", w, m, stats, summary); %#ok<AGROW>
    distanceRows(end+1, :) = distanceRow("common10", w, stats); %#ok<AGROW>
end

w6CommonMedoid = commonMedoids('famp6');
w6CommonToFullDistance = curveDistance(curveMat.pcNormalized(w6FullMedoid, :), ...
                                       curveMat.pcNormalized(w6CommonMedoid, :));
w6FullMeanDistances = mean(w6FullDistances, 2);
[~, rankOrder] = sort(w6FullMeanDistances, 'ascend');
w6CommonRankInFull = find(rankOrder == find(find(w6FullMask) == w6CommonMedoid), 1);
if isempty(w6CommonRankInFull)
    w6CommonRankInFull = NaN;
end

comparisonRows = { ...
    "w6_common10_medoid_vs_w6_all87_medoid", ...
    w6FullMedoid, w6CommonMedoid, ...
    summary.SliceIndex(w6FullMedoid), summary.SliceIndex(w6CommonMedoid), ...
    w6CommonToFullDistance, w6CommonRankInFull};

comparison = cell2table(comparisonRows, 'VariableNames', ...
    {'Comparison', 'FullMedoidCurveId', 'ReducedMedoidCurveId', ...
     'FullMedoidSliceIndex', 'ReducedMedoidSliceIndex', ...
     'CurveDistanceRmsLog10Pc', 'ReducedMedoidRankWithinFull87'});

results = struct();
results.MedoidSummary = cell2table(medoidRows, 'VariableNames', ...
    {'TestName', 'Window', 'MedoidCurveId', 'MedoidSliceIndex', ...
     'NumCurves', 'MeanDistance', 'MedianDistance', 'P90Distance', ...
     'MaxDistance', 'AssignedState', 'SamplingPool'});
results.DistanceSummary = cell2table(distanceRows, 'VariableNames', ...
    {'TestName', 'Window', 'NumCurves', 'MeanPairDistance', ...
     'MedianPairDistance', 'P90PairDistance', 'MaxPairDistance'});
results.W6MedoidComparison = comparison;
results.W6FullMedoidCurveId = w6FullMedoid;
results.W6Common10MedoidCurveId = w6CommonMedoid;
results.Common10MedoidCurveIds = commonMedoids;

% Append comparison to disk-friendly distance table.
results.DistanceSummary.FullMedoidCurveId = nan(height(results.DistanceSummary), 1);
results.DistanceSummary.ReducedMedoidCurveId = nan(height(results.DistanceSummary), 1);
results.DistanceSummary.CurveDistanceRmsLog10Pc = nan(height(results.DistanceSummary), 1);
results.DistanceSummary.ReducedMedoidRankWithinFull87 = nan(height(results.DistanceSummary), 1);
newRow = results.DistanceSummary(1, :);
newRow.TestName = "w6_reduced_vs_full";
newRow.Window = "famp6";
newRow.NumCurves = 87;
newRow.MeanPairDistance = NaN;
newRow.MedianPairDistance = NaN;
newRow.P90PairDistance = NaN;
newRow.MaxPairDistance = NaN;
newRow.FullMedoidCurveId = w6FullMedoid;
newRow.ReducedMedoidCurveId = w6CommonMedoid;
newRow.CurveDistanceRmsLog10Pc = w6CommonToFullDistance;
newRow.ReducedMedoidRankWithinFull87 = w6CommonRankInFull;
results.DistanceSummary = [results.DistanceSummary; newRow];
end


function [medoidCurveId, stats, distances] = medoidForMask(curveMat, mask)
% Return the global curve id with smallest average distance in the subset.

ids = find(mask);
assert(~isempty(ids), 'Cannot select medoid from an empty curve subset.');
curves = curveMat.pcNormalized(ids, :);
distances = pairwiseCurveDistances(curves);
meanDist = mean(distances, 2, 'omitnan');
[~, localIdx] = min(meanDist);
medoidCurveId = ids(localIdx);

upper = distances(triu(true(size(distances)), 1));
upper = upper(isfinite(upper));
stats = struct();
stats.NumCurves = numel(ids);
stats.MeanPairDistance = mean(upper, 'omitnan');
stats.MedianPairDistance = median(upper, 'omitnan');
stats.P90PairDistance = prctile(upper, 90);
stats.MaxPairDistance = max(upper);
stats.MeanDistance = meanDist(localIdx);
stats.MedianDistance = median(distances(localIdx, :), 'omitnan');
stats.P90Distance = prctile(distances(localIdx, :), 90);
stats.MaxDistance = max(distances(localIdx, :));
end


function D = pairwiseCurveDistances(curves)
% Pairwise RMS distance between log10-normalized Pc curves.

n = size(curves, 1);
D = zeros(n, n);
for i = 1:n
    for j = i+1:n
        d = curveDistance(curves(i, :), curves(j, :));
        D(i, j) = d;
        D(j, i) = d;
    end
end
end


function d = curveDistance(a, b)
% RMS distance in log10 Pc over the common saturation grid.

d = sqrt(mean((log10(a(:)) - log10(b(:))).^2, 'omitnan'));
end


function row = medoidRow(testName, windowName, medoidCurveId, stats, summary)
% One medoid-summary table row.

row = {testName, windowName, medoidCurveId, summary.SliceIndex(medoidCurveId), ...
       stats.NumCurves, stats.MeanDistance, stats.MedianDistance, ...
       stats.P90Distance, stats.MaxDistance, ...
       summary.AssignedState(medoidCurveId), summary.SamplingPool(medoidCurveId)};
end


function row = distanceRow(testName, windowName, stats)
% One distance-summary table row.

row = {testName, windowName, stats.NumCurves, stats.MeanPairDistance, ...
       stats.MedianPairDistance, stats.P90PairDistance, stats.MaxPairDistance};
end


function makePcPilotFigures(curveMat, results, figureDir)
% Create review figures for W6 full and common-10 Pc pilots.

ensureFolder(figureDir);
makeW6All87Figure(curveMat, results, figureDir);
makeCommon10Figure(curveMat, results, figureDir);
makeW6MedoidComparisonFigure(curveMat, results, figureDir);
end


function makeW6All87Figure(curveMat, results, figureDir)
% Plot all 87 W6 curves and highlight full/reduced medoids.

summary = curveMat.summary;
sg = curveMat.sgGrid;
mask = summary.Window == "famp6" & summary.InW6All87;
commonMask = summary.Window == "famp6" & summary.InCommon10;

fig = figure('Color', 'w', 'Position', [100, 100, 1100, 760]);
hold on
plot(sg, curveMat.pcNormalized(mask, :)', 'Color', [0.72 0.74 0.78], 'LineWidth', 0.7);
plot(sg, curveMat.pcNormalized(commonMask, :)', 'Color', [0.45 0.62 0.84], 'LineWidth', 1.0);
plot(sg, curveMat.pcNormalized(results.W6FullMedoidCurveId, :), ...
     'Color', [0.02 0.02 0.02], 'LineWidth', 3.2);
plot(sg, curveMat.pcNormalized(results.W6Common10MedoidCurveId, :), ...
     '--', 'Color', [0.92 0.43 0.08], 'LineWidth', 3.2);
set(gca, 'YScale', 'log', 'FontSize', 18, 'LineWidth', 1.1);
grid on; box on
xlabel('Gas saturation');
ylabel('Normalized capillary pressure');
title({'W6 Pc pilot: all 87 slices', ...
       'Grey = all W6 curves, blue = common-10 subset, black/orange = medoids'}, ...
       'FontSize', 22, 'FontWeight', 'bold');
legend({'W6 all 87', 'W6 common 10', 'Full-87 medoid', 'Common-10 medoid'}, ...
       'Location', 'northwest');
saveFigureBoth(fig, figureDir, 'w6_all87_pc_curves_with_medoids');
close(fig);
end


function makeCommon10Figure(curveMat, results, figureDir)
% Plot 10 curves per window and highlight common-10 medoids.

summary = curveMat.summary;
sg = curveMat.sgGrid;
windows = ["famp1", "famp2", "famp3", "famp4", "famp5", "famp6"];

fig = figure('Color', 'w', 'Position', [100, 100, 1500, 900]);
tiledlayout(2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

for i = 1:numel(windows)
    nexttile
    w = windows(i);
    mask = summary.Window == w & summary.InCommon10;
    curves = curveMat.pcNormalized(mask, :);
    ids = find(mask);
    medoidTable = results.MedoidSummary(results.MedoidSummary.TestName == "common10" & ...
        results.MedoidSummary.Window == w, :);
    medoidId = medoidTable.MedoidCurveId(1);
    plot(sg, curves', 'Color', [0.68 0.71 0.76], 'LineWidth', 0.9);
    hold on
    plot(sg, curveMat.pcNormalized(medoidId, :), ...
         'Color', [0.86 0.22 0.16], 'LineWidth', 3.0);
    set(gca, 'YScale', 'log', 'FontSize', 16, 'LineWidth', 1.0);
    grid on; box on
    title(sprintf('W%d | medoid slice %d', i, summary.SliceIndex(medoidId)), ...
          'FontSize', 18, 'FontWeight', 'bold');
    xlabel('Gas saturation');
    ylabel('Normalized Pc');
    if isempty(ids)
        text(0.5, 0.5, 'No curves', 'HorizontalAlignment', 'center');
    end
end

sgtitle({'Reduced Pc pilot: 10 common slices per window', ...
         'Grey = 10 selected slice curves, red = window medoid'}, ...
        'FontSize', 24, 'FontWeight', 'bold');
saveFigureBoth(fig, figureDir, 'common10_all_windows_pc_curves_with_medoids');
close(fig);
end


function makeW6MedoidComparisonFigure(curveMat, results, figureDir)
% Direct comparison of the W6 full-87 and common-10 medoids.

summary = curveMat.summary;
sg = curveMat.sgGrid;
fullId = results.W6FullMedoidCurveId;
reducedId = results.W6Common10MedoidCurveId;

fig = figure('Color', 'w', 'Position', [100, 100, 950, 700]);
plot(sg, curveMat.pcNormalized(fullId, :), ...
     'Color', [0.02 0.02 0.02], 'LineWidth', 3.2);
hold on
plot(sg, curveMat.pcNormalized(reducedId, :), ...
     '--', 'Color', [0.92 0.43 0.08], 'LineWidth', 3.2);
set(gca, 'YScale', 'log', 'FontSize', 18, 'LineWidth', 1.1);
grid on; box on
xlabel('Gas saturation');
ylabel('Normalized capillary pressure');
title({'W6 medoid comparison', ...
       sprintf('Full-87 slice %d vs common-10 slice %d', ...
               summary.SliceIndex(fullId), summary.SliceIndex(reducedId))}, ...
      'FontSize', 22, 'FontWeight', 'bold');
legend({'Full-87 medoid', 'Common-10 medoid'}, 'Location', 'northwest');
saveFigureBoth(fig, figureDir, 'w6_full87_vs_common10_medoid_pc_curve');
close(fig);
end


function saveFigureBoth(fig, outputDir, baseName)
% Save a MATLAB figure as PNG and PDF.

pngPath = fullfile(outputDir, [baseName '.png']);
pdfPath = fullfile(outputDir, [baseName '.pdf']);
exportgraphics(fig, pngPath, 'Resolution', 220);
exportgraphics(fig, pdfPath, 'ContentType', 'vector');
fprintf('Saved figure: %s\n', pngPath);
fprintf('Saved figure: %s\n', pdfPath);
end


function ensureFolder(folderPath)
% Create a folder if needed.

if exist(folderPath, 'dir') ~= 7
    mkdir(folderPath);
end
end
