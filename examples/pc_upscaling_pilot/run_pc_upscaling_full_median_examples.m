%RUN_PC_UPSCALING_FULL_MEDIAN_EXAMPLES Full Pc pilot for four field examples.
%
% This script applies the rigorous Pc-screening workflow to the four
% representative median-sand-ratio Texas offshore examples:
%
%   geology: s05_c012
%   scenario: medium sand, nonuniform
%   geologic case: case_012_zf0500_svcl010_cvcl060
%   Level-3 cases: 1, 3, 4, 7
%
% For each Level-3 case, each throw window, and each of 87 along-strike
% slices, the script replays the exact selected PREDICT realization,
% computes an effective Pc curve from the replayed fine-grid properties,
% and chooses one medoid Pc curve per case-window pair.
%
% The Pc curve is a screening capillary-equilibrium upscaling: fine cells
% receive Brooks-Corey/Leverett-style entry pressures scaled by sqrt(phi/k),
% bulk gas saturation is pore-volume averaged at each trial Pc, and medoid
% selection uses normalized Pc-curve shape in log space.

clear; clc;

scriptDir = fileparts(mfilename('fullpath'));
examplesDir = fileparts(scriptDir);
addpath(examplesDir);

setupMrstIfNeeded();

cfg = struct();
cfg.outputRoot = fullfile('D:', 'codex_gom', 'UQ_workflow', ...
    'pc_upscaling_median_examples_full87');
cfg.fieldSamplingCsv = fullfile('D:', 'codex_gom', 'UQ_workflow', ...
    'texas_offshore_field_sampling', 'texas_field_slice_window_values.csv');
cfg.dataRoot = fullfile(examplesDir, 'thickness_scenario_data');
cfg.replayPredictCodeRoot = fullfile('D:', 'codex_gom', ...
    'predict_shaowen_replay_2647b6d');
% Production-code replay agrees to better than 5e-4 log10(k) for this data.
% A 1e-3 tolerance avoids unnecessary reruns while still catching real drift.
cfg.verifyToleranceLog10 = 1.0e-3;
cfg.geologyId = "s05_c012";
cfg.caseIds = [1 3 4 7];
cfg.windows = ["famp1", "famp2", "famp3", "famp4", "famp5", "famp6"];

cfg.inputDir = fullfile(cfg.outputRoot, 'inputs');
cfg.replayDir = fullfile(cfg.outputRoot, 'replay_unique');
cfg.curveDir = fullfile(cfg.outputRoot, 'curves');
cfg.tableDir = fullfile(cfg.outputRoot, 'tables');
cfg.figureDir = fullfile(cfg.outputRoot, 'figures');
cfg.logDir = fullfile(cfg.outputRoot, 'logs');
ensureFolder(cfg.inputDir);
ensureFolder(cfg.replayDir);
ensureFolder(cfg.curveDir);
ensureFolder(cfg.tableDir);
ensureFolder(cfg.figureDir);
ensureFolder(cfg.logDir);

fprintf('\n=== Build full-87 replay selection ===\n')
selectionCsv = buildFull87Selection(cfg);
selectionTable = readtable(selectionCsv, 'TextType', 'string');

fprintf('\n=== Replay exact selected PREDICT realizations ===\n')
replaySummaryCsv = fullfile(cfg.replayDir, 'replay_summary.csv');
if exist(replaySummaryCsv, 'file') == 2
    replaySummary = readtable(replaySummaryCsv, 'TextType', 'string');
    if all(ismember({'MaxAbsLog10Diff', 'VerificationStatus'}, ...
                    replaySummary.Properties.VariableNames))
        replayDiffs = str2double(string(replaySummary.MaxAbsLog10Diff));
        maxReplayDiff = max(replayDiffs, [], 'omitnan');
        numBadReplayRows = sum(replayDiffs > cfg.verifyToleranceLog10 | ...
            string(replaySummary.VerificationStatus) == "replay_failed");
    else
        maxReplayDiff = inf;
        numBadReplayRows = height(replaySummary);
    end
    if height(replaySummary) == height(selectionTable) && numBadReplayRows == 0
        fprintf('Using existing replay summary: %s\n', replaySummaryCsv);
    else
        fprintf(['Existing replay summary has %d rows, selection has %d rows, ' ...
            'max diff is %.3g, and %d rows exceed tolerance; rerunning replay.\n'], ...
            height(replaySummary), height(selectionTable), maxReplayDiff, numBadReplayRows);
        replaySummary = replay_selected_predict_realizations( ...
            selectionCsv, cfg.replayDir, ...
            'DataRoot', cfg.dataRoot, ...
            'MaxRows', inf, ...
            'PredictCodeRoot', cfg.replayPredictCodeRoot, ...
            'VerifyToleranceLog10', cfg.verifyToleranceLog10);
    end
else
    replaySummary = replay_selected_predict_realizations( ...
        selectionCsv, cfg.replayDir, ...
        'DataRoot', cfg.dataRoot, ...
        'MaxRows', inf, ...
        'PredictCodeRoot', cfg.replayPredictCodeRoot, ...
        'VerifyToleranceLog10', cfg.verifyToleranceLog10);
end

replaySummary = attachSelectionContext(replaySummary, selectionTable);
replaySummaryContextCsv = fullfile(cfg.tableDir, ...
    'replay_summary_with_full87_context_s05_c012_cases_01_03_04_07.csv');
writetable(replaySummary, replaySummaryContextCsv);
fprintf('Saved replay summary with context: %s\n', replaySummaryContextCsv);

fprintf('\n=== Compute effective Pc curves ===\n')
pcOpt = defaultPcOptions();
[curveLong, curveSummary, curveMat] = computePcCurves(replaySummary, pcOpt);

curveLongCsv = fullfile(cfg.curveDir, ...
    'pc_curve_points_s05_c012_cases_01_03_04_07_full87.csv');
curveSummaryCsv = fullfile(cfg.tableDir, ...
    'pc_curve_summary_s05_c012_cases_01_03_04_07_full87.csv');
curveMatFile = fullfile(cfg.curveDir, ...
    'pc_curves_s05_c012_cases_01_03_04_07_full87.mat');
writetable(curveLong, curveLongCsv);
writetable(curveSummary, curveSummaryCsv);
save(curveMatFile, 'curveMat', 'pcOpt', 'cfg', '-v7.3');
fprintf('Saved Pc curve points: %s\n', curveLongCsv);
fprintf('Saved Pc curve summary: %s\n', curveSummaryCsv);
fprintf('Saved Pc curve MAT: %s\n', curveMatFile);

fprintf('\n=== Select full-87 medoid Pc curves ===\n')
results = analyzeFull87Medoids(curveMat, cfg);
medoidSummaryCsv = fullfile(cfg.tableDir, ...
    'pc_medoid_summary_s05_c012_cases_01_03_04_07_full87.csv');
distanceSummaryCsv = fullfile(cfg.tableDir, ...
    'pc_distance_summary_s05_c012_cases_01_03_04_07_full87.csv');
caseSummaryCsv = fullfile(cfg.tableDir, ...
    'pc_case_summary_s05_c012_cases_01_03_04_07_full87.csv');
writetable(results.MedoidSummary, medoidSummaryCsv);
writetable(results.DistanceSummary, distanceSummaryCsv);
writetable(results.CaseSummary, caseSummaryCsv);
save(fullfile(cfg.tableDir, ...
    'pc_medoid_results_s05_c012_cases_01_03_04_07_full87.mat'), ...
    'results', '-v7.3');
fprintf('Saved medoid summary: %s\n', medoidSummaryCsv);
fprintf('Saved distance summary: %s\n', distanceSummaryCsv);
fprintf('Saved case summary: %s\n', caseSummaryCsv);

fprintf('\n=== Generate full-87 review figures ===\n')
makeFull87Figures(curveMat, results, cfg.figureDir);

fprintf('\nFull median-example Pc pilot complete.\n')
fprintf('Output root: %s\n', cfg.outputRoot);


function setupMrstIfNeeded()
% Start MRST if available locally and not already on the MATLAB path.

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


function selectionCsv = buildFull87Selection(cfg)
% Extract one replay row for every requested case/slice/window combination.

assert(exist(cfg.fieldSamplingCsv, 'file') == 2, ...
    'Missing field sampling table: %s', cfg.fieldSamplingCsv);

T = readtable(cfg.fieldSamplingCsv, 'TextType', 'string');
caseId = str2double(string(T.case_id));
mask = T.geology_id == cfg.geologyId & ismember(caseId, cfg.caseIds);
S = T(mask, :);
assert(height(S) > 0, 'No rows found for geology %s.', cfg.geologyId);

S.Level3PcPilot = repmat("full87_four_examples", height(S), 1);
S.InFull87PcPilot = true(height(S), 1);
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

selectionCsv = fullfile(cfg.inputDir, ...
    's05_c012_cases_01_03_04_07_full87_replay_rows.csv');
writetable(S, selectionCsv);
fprintf('Prepared full-87 replay selection: %s (%d rows)\n', ...
    selectionCsv, height(S));
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


function T = attachSelectionContext(T, selectionTable)
% Attach field-sampling context to the replay summary table.

assert(height(T) == height(selectionTable), ...
    'Replay summary and selection table have different row counts.');

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


function pcOpt = defaultPcOptions()
% Default screening Pc model parameters.

pcOpt.sgGrid = linspace(0.02, 0.98, 80);
pcOpt.numPressureTrials = 500;
pcOpt.brooksCoreyLambda = 2.0;
pcOpt.pcRefPa = 1.0e4;
pcOpt.minPoro = 1.0e-4;
pcOpt.minPermMD = 1.0e-9;
pcOpt.mDInM2 = 9.869233e-16;
pcOpt.medoidColor = [0.86 0.22 0.16];
pcOpt.greyCurveColor = [0.68 0.71 0.76];
end


function [curveLong, curveSummary, curveMat] = computePcCurves(replaySummary, pcOpt)
% Compute effective Pc curves for every replayed realization.

n = height(replaySummary);
sg = pcOpt.sgGrid(:)';
pcNorm = nan(n, numel(sg));
pcPa = nan(n, numel(sg));

summaryRows = cell(n, 25);
longRows = cell(n * numel(sg), 15);
longIdx = 0;

for i = 1:n
    outputFile = char(replaySummary.OutputFile(i));
    fprintf('Pc curve %4d/%4d: case %02d slice %02d window %s\n', ...
        i, n, replaySummary.Level3CaseId(i), ...
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
        replaySummary.CaseCategory(i), replaySummary.CaseStrength(i), ...
        replaySummary.PatternName(i), replaySummary.Window(i), ...
        replaySummary.SliceIndex(i), replaySummary.AssignedState(i), ...
        replaySummary.SamplingMode(i), replaySummary.SamplingPool(i), ...
        replaySummary.SelectedSampleIndex(i), replaySummary.ReplaySeed(i), ...
        replaySummary.LogKxx(i), replaySummary.LogKyy(i), replaySummary.LogKzz(i), ...
        curve.poreVolume, curve.medianEntryPressure, ...
        curve.p05EntryPressure, curve.p95EntryPressure, outputFile};

    for j = 1:numel(sg)
        longIdx = longIdx + 1;
        longRows(longIdx, :) = { ...
            i, replaySummary.GeologyId(i), replaySummary.Level3CaseId(i), ...
            replaySummary.Level3CaseName(i), replaySummary.Window(i), ...
            replaySummary.SliceIndex(i), replaySummary.AssignedState(i), ...
            replaySummary.SamplingMode(i), replaySummary.SamplingPool(i), ...
            sg(j), pcNorm(i, j), pcPa(i, j), log10(pcNorm(i, j)), ...
            replaySummary.LogKxx(i), replaySummary.LogKzz(i)};
    end
end

curveSummary = cell2table(summaryRows, 'VariableNames', ...
    {'CurveId', 'ReplaySourceRow', 'GeologyId', 'ScenarioName', ...
     'CaseLabel', 'Level3CaseId', 'Level3CaseName', 'CaseCategory', ...
     'CaseStrength', 'PatternName', 'Window', 'SliceIndex', ...
     'AssignedState', 'SamplingMode', 'SamplingPool', ...
     'SelectedSampleIndex', 'ReplaySeed', 'LogKxx', 'LogKyy', 'LogKzz', ...
     'PoreVolume', 'MedianEntryPressureNorm', ...
     'P05EntryPressureNorm', 'P95EntryPressureNorm', 'ReplayOutputFile'});

curveLong = cell2table(longRows, 'VariableNames', ...
    {'CurveId', 'GeologyId', 'Level3CaseId', 'Level3CaseName', ...
     'Window', 'SliceIndex', 'AssignedState', 'SamplingMode', ...
     'SamplingPool', 'GasSaturation', 'PcNormalized', 'PcPa', ...
     'Log10PcNormalized', 'LogKxx', 'LogKzz'});

curveMat = struct();
curveMat.sgGrid = sg;
curveMat.pcNormalized = pcNorm;
curveMat.pcPa = pcPa;
curveMat.summary = curveSummary;
end


function curve = effectivePcCurveFromReplay(replay, pcOpt)
% Upscale fine-cell Pc by pore-volume averaging at trial pressures.

poroAll = replay.Grid.poro(:);
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
poroAll = max(poroAll, pcOpt.minPoro);

if isfield(replay, 'G') && isfield(replay.G, 'cells') && ...
        isfield(replay.G.cells, 'volumes') && numel(replay.G.cells.volumes) == numel(poroAll)
    bulkVolumeAll = replay.G.cells.volumes(:);
else
    bulkVolumeAll = ones(size(poroAll));
end

poreWeights = poroAll .* bulkVolumeAll;
valid = isfinite(poreWeights) & poreWeights > 0 & all(isfinite(kSI), 2);
poreWeights = poreWeights(valid);
poro = poroAll(valid);
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
curve.poreVolume = sum(poroAll(valid) .* bulkVolumeAll(valid));
curve.medianEntryPressure = median(entry, 'omitnan');
curve.p05EntryPressure = prctile(entry, 5);
curve.p95EntryPressure = prctile(entry, 95);
end


function results = analyzeFull87Medoids(curveMat, cfg)
% Select medoid curves for every case-window pair.

summary = curveMat.summary;
caseIds = cfg.caseIds(:)';
windows = cfg.windows;
medoidRows = {};
distanceRows = {};
caseRows = {};

for c = caseIds
    caseMask = summary.Level3CaseId == c;
    caseRows(end+1, :) = { ...
        summary.GeologyId(find(caseMask, 1)), c, ...
        summary.Level3CaseName(find(caseMask, 1)), ...
        summary.CaseCategory(find(caseMask, 1)), ...
        summary.CaseStrength(find(caseMask, 1)), ...
        summary.PatternName(find(caseMask, 1)), ...
        sum(caseMask)}; %#ok<AGROW>

    for iw = 1:numel(windows)
        w = windows(iw);
        mask = caseMask & summary.Window == w;
        [m, stats] = medoidForMask(curveMat, mask);
        medoidRows(end+1, :) = medoidRow(c, w, m, stats, summary); %#ok<AGROW>
        distanceRows(end+1, :) = distanceRow(c, w, stats, summary); %#ok<AGROW>
    end
end

results = struct();
results.MedoidSummary = cell2table(medoidRows, 'VariableNames', ...
    {'GeologyId', 'Level3CaseId', 'Level3CaseName', 'CaseCategory', ...
     'CaseStrength', 'PatternName', 'Window', 'MedoidCurveId', ...
     'MedoidSliceIndex', 'NumCurves', 'MeanDistance', 'MedianDistance', ...
     'P90Distance', 'MaxDistance', 'AssignedState', 'SamplingMode', ...
     'SamplingPool', 'SelectedSampleIndex', 'LogKxx', 'LogKyy', 'LogKzz'});
results.DistanceSummary = cell2table(distanceRows, 'VariableNames', ...
    {'GeologyId', 'Level3CaseId', 'Level3CaseName', 'Window', ...
     'NumCurves', 'MeanPairDistance', 'MedianPairDistance', ...
     'P90PairDistance', 'MaxPairDistance'});
results.CaseSummary = cell2table(caseRows, 'VariableNames', ...
    {'GeologyId', 'Level3CaseId', 'Level3CaseName', 'CaseCategory', ...
     'CaseStrength', 'PatternName', 'NumReplayRows'});
end


function [medoidCurveId, stats, distances] = medoidForMask(curveMat, mask)
% Return the global curve id with smallest average distance in the subset.

ids = find(mask);
assert(numel(ids) == 87, ...
    'Expected 87 curves for each full case-window pair, found %d.', numel(ids));
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


function row = medoidRow(caseId, windowName, medoidCurveId, stats, summary)
% One medoid-summary table row.

row = {summary.GeologyId(medoidCurveId), caseId, ...
       summary.Level3CaseName(medoidCurveId), ...
       summary.CaseCategory(medoidCurveId), ...
       summary.CaseStrength(medoidCurveId), ...
       summary.PatternName(medoidCurveId), ...
       windowName, medoidCurveId, summary.SliceIndex(medoidCurveId), ...
       stats.NumCurves, stats.MeanDistance, stats.MedianDistance, ...
       stats.P90Distance, stats.MaxDistance, ...
       summary.AssignedState(medoidCurveId), ...
       summary.SamplingMode(medoidCurveId), ...
       summary.SamplingPool(medoidCurveId), ...
       summary.SelectedSampleIndex(medoidCurveId), ...
       summary.LogKxx(medoidCurveId), summary.LogKyy(medoidCurveId), ...
       summary.LogKzz(medoidCurveId)};
end


function row = distanceRow(caseId, windowName, stats, summary)
% One distance-summary table row.

caseIdx = find(summary.Level3CaseId == caseId, 1);
row = {summary.GeologyId(caseIdx), caseId, ...
       summary.Level3CaseName(caseIdx), windowName, ...
       stats.NumCurves, stats.MeanPairDistance, ...
       stats.MedianPairDistance, stats.P90PairDistance, stats.MaxPairDistance};
end


function makeFull87Figures(curveMat, results, figureDir)
% Create one 2-by-3 Pc-curve figure per representative Level-3 case.

ensureFolder(figureDir);
caseIds = unique(results.MedoidSummary.Level3CaseId, 'stable');
for i = 1:numel(caseIds)
    makeCaseFull87Figure(curveMat, results, figureDir, caseIds(i));
end
makeMedoidDistanceOverview(results, figureDir);
end


function makeCaseFull87Figure(curveMat, results, figureDir, caseId)
% Plot all 87 curves for each window in one Level-3 case.

summary = curveMat.summary;
sg = curveMat.sgGrid;
windows = ["famp1", "famp2", "famp3", "famp4", "famp5", "famp6"];
caseName = prettyLabel(summary.Level3CaseName( ...
    find(summary.Level3CaseId == caseId, 1)));
caseCategory = prettyLabel(summary.CaseCategory( ...
    find(summary.Level3CaseId == caseId, 1)));

fig = figure('Color', 'w', 'Position', [80, 80, 1550, 930]);
tiledlayout(2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

for iw = 1:numel(windows)
    nexttile
    w = windows(iw);
    mask = summary.Level3CaseId == caseId & summary.Window == w;
    medoidRow = results.MedoidSummary( ...
        results.MedoidSummary.Level3CaseId == caseId & ...
        results.MedoidSummary.Window == w, :);
    medoidId = medoidRow.MedoidCurveId(1);

    plot(sg, curveMat.pcNormalized(mask, :)', ...
        'Color', [0.68 0.71 0.76], 'LineWidth', 0.75);
    hold on
    plot(sg, curveMat.pcNormalized(medoidId, :), ...
        'Color', [0.86 0.22 0.16], 'LineWidth', 3.0);
    set(gca, 'YScale', 'log', 'FontSize', 15, 'LineWidth', 1.0);
    grid on; box on
    title(sprintf('W%d | %s | medoid slice %d', iw, ...
        char(prettyLabel(medoidRow.AssignedState(1))), ...
        medoidRow.MedoidSliceIndex(1)), ...
        'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');
    xlabel('Gas saturation');
    ylabel('Normalized Pc');
end

sgtitle({sprintf('Full-87 Pc curves | s05_c012 case %02d: %s', ...
                 caseId, char(caseName)), ...
         sprintf('%s | grey = 87 slices, red = medoid Pc curve', ...
                 char(caseCategory))}, ...
        'FontSize', 22, 'FontWeight', 'bold', 'Interpreter', 'none');

baseName = sprintf('s05_c012_case%02d_full87_pc_curves_with_medoids', caseId);
saveFigureBoth(fig, figureDir, baseName);
close(fig);
end


function label = prettyLabel(value)
% Convert workflow identifiers into readable figure labels.

label = strrep(string(value), '_', ' ');
end


function makeMedoidDistanceOverview(results, figureDir)
% Plot medoid centrality distance for all case-window pairs.

T = results.MedoidSummary;
caseIds = unique(T.Level3CaseId, 'stable');
windows = ["famp1", "famp2", "famp3", "famp4", "famp5", "famp6"];
M = nan(numel(caseIds), numel(windows));
labels = strings(numel(caseIds), 1);
for i = 1:numel(caseIds)
    c = caseIds(i);
    labels(i) = sprintf('case %02d', c);
    for j = 1:numel(windows)
        mask = T.Level3CaseId == c & T.Window == windows(j);
        M(i, j) = T.MeanDistance(mask);
    end
end

fig = figure('Color', 'w', 'Position', [100, 100, 980, 520]);
imagesc(M);
axis tight
colormap(parula);
cb = colorbar;
cb.Label.String = 'Medoid mean distance';
set(gca, 'XTick', 1:numel(windows), ...
         'XTickLabel', {'W1','W2','W3','W4','W5','W6'}, ...
         'YTick', 1:numel(caseIds), ...
         'YTickLabel', labels, ...
         'FontSize', 16, 'LineWidth', 1.0);
xlabel('Throw window');
ylabel('Representative example');
title('Pc medoid centrality across four full-87 examples', ...
    'FontSize', 20, 'FontWeight', 'bold');
for i = 1:size(M, 1)
    for j = 1:size(M, 2)
        text(j, i, sprintf('%.2f', M(i, j)), ...
            'HorizontalAlignment', 'center', ...
            'Color', chooseTextColor(M(i, j), M), ...
            'FontSize', 13, 'FontWeight', 'bold');
    end
end
saveFigureBoth(fig, figureDir, 's05_c012_cases_01_03_04_07_pc_medoid_centrality');
close(fig);
end


function color = chooseTextColor(value, M)
% Choose black or white text based on relative color intensity.

lo = min(M(:), [], 'omitnan');
hi = max(M(:), [], 'omitnan');
if hi <= lo
    scaled = 0.5;
else
    scaled = (value - lo) / (hi - lo);
end
if scaled > 0.55
    color = [1 1 1];
else
    color = [0.05 0.10 0.18];
end
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
