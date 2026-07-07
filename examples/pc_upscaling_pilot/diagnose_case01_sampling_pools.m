%DIAGNOSE_CASE01_SAMPLING_POOLS Compare case01 sampling pools and Pc spread.
%
% This diagnostic focuses on the median-sand nonuniform example used in the
% full-87 Pc pilot:
%
%   geology: s05_c012
%   scenario: scenario_05_medium_sand_nonuniform
%   geologic case: case_012_zf0500_svcl010_cvcl060
%   Level-3 case: 01, independent draw 1
%
% For every throw window, the script compares the full 2000-realization
% independent sampling pool against the 87 realizations selected for the 87
% along-strike slices. It also checks whether selected effective
% permeability explains the Pc-curve distance from the medoid.

clear; clc;

repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
outputRoot = fullfile('D:', 'codex_gom', 'UQ_workflow', ...
    'pc_upscaling_median_examples_full87');
figureDir = fullfile(outputRoot, 'figures', 'case01_sampling_pool_diagnostics');
tableDir = fullfile(outputRoot, 'tables');
ensureFolder(figureDir);

scenarioName = "scenario_05_medium_sand_nonuniform";
caseLabel = "case_012_zf0500_svcl010_cvcl060";
caseId = 1;
windows = ["famp1", "famp2", "famp3", "famp4", "famp5", "famp6"];
componentNames = ["kxx", "kyy", "kzz"];
windowLabels = ["W1", "W2", "W3", "W4", "W5", "W6"];

dataRoot = fullfile(repoRoot, 'examples', 'thickness_scenario_data', ...
    'data', scenarioName);
selectionFile = fullfile(tableDir, ...
    'replay_summary_with_full87_context_s05_c012_cases_01_03_04_07.csv');
curveMatFile = fullfile(outputRoot, 'curves', ...
    'pc_curves_s05_c012_cases_01_03_04_07_full87.mat');
resultMatFile = fullfile(outputRoot, 'tables', ...
    'pc_medoid_results_s05_c012_cases_01_03_04_07_full87.mat');

assert(exist(selectionFile, 'file') == 2, 'Missing selection table: %s', selectionFile);
assert(exist(curveMatFile, 'file') == 2, 'Missing Pc curve MAT: %s', curveMatFile);
assert(exist(resultMatFile, 'file') == 2, 'Missing Pc result MAT: %s', resultMatFile);

selection = readtable(selectionFile, 'TextType', 'string');
selection = selection(selection.Level3CaseId == caseId, :);
curveData = load(curveMatFile, 'curveMat');
resultData = load(resultMatFile, 'results');
curveMat = curveData.curveMat;
results = resultData.results;

pool = struct();
for iw = 1:numel(windows)
    matFile = fullfile(dataRoot, windows(iw), caseLabel, 'predict_runs.mat');
    assert(exist(matFile, 'file') == 2, 'Missing PREDICT result: %s', matFile);
    S = load(matFile, 'perms');
    pool(iw).window = windows(iw);
    pool(iw).perms = log10(max(S.perms, realmin));
end

summary = buildSummaryTable(pool, selection, windows, componentNames);
writetable(summary, fullfile(tableDir, ...
    'case01_sampling_pool_perm_summary.csv'));

makeMarginalPoolFigure(pool, selection, results, windows, windowLabels, ...
    componentNames, figureDir);
makePcDistancePermFigure(selection, curveMat, results, windows, ...
    windowLabels, figureDir);

fprintf('Saved case01 sampling-pool diagnostics to: %s\n', figureDir);


function summary = buildSummaryTable(pool, selection, windows, componentNames)
% Summarize full-pool and selected-slice permeability distributions.

rows = {};
for iw = 1:numel(windows)
    w = windows(iw);
    selectedMask = selection.Window == w;
    selectedPerms = [selection.LogKxx(selectedMask), ...
                     selection.LogKyy(selectedMask), ...
                     selection.LogKzz(selectedMask)];
    poolPerms = pool(iw).perms;
    for ic = 1:numel(componentNames)
        p = poolPerms(:, ic);
        s = selectedPerms(:, ic);
        rows(end+1, :) = { ...
            w, componentNames(ic), numel(p), numel(s), ...
            prctile(p, 5), median(p, 'omitnan'), prctile(p, 95), ...
            prctile(p, 95) - prctile(p, 5), ...
            prctile(s, 5), median(s, 'omitnan'), prctile(s, 95), ...
            prctile(s, 95) - prctile(s, 5)};
    end
end

summary = cell2table(rows, 'VariableNames', ...
    {'Window', 'Component', 'PoolN', 'SelectedN', ...
     'PoolP05', 'PoolMedian', 'PoolP95', 'PoolP90Spread', ...
     'SelectedP05', 'SelectedMedian', 'SelectedP95', ...
     'SelectedP90Spread'});
end


function makeMarginalPoolFigure(pool, selection, results, windows, ...
        windowLabels, componentNames, figureDir)
% Show full independent pool, selected 87 samples, and Pc medoid sample.

fig = figure('Color', 'w', 'Position', [60, 60, 1700, 900]);
tiledlayout(3, 6, 'Padding', 'compact', 'TileSpacing', 'compact');
rowEdges = {-7:0.25:2, -7:0.25:2, -7:0.25:2};
xticksValue = [-7, -4, -1, 2];

for ic = 1:numel(componentNames)
    edges = rowEdges{ic};
    for iw = 1:numel(windows)
        nexttile
        w = windows(iw);
        poolValues = pool(iw).perms(:, ic);
        selectedMask = selection.Window == w;
        selectedValues = [selection.LogKxx(selectedMask), ...
                          selection.LogKyy(selectedMask), ...
                          selection.LogKzz(selectedMask)];
        selectedValues = selectedValues(:, ic);
        medoidRow = results.MedoidSummary( ...
            results.MedoidSummary.Level3CaseId == 1 & ...
            results.MedoidSummary.Window == w, :);
        medoidValues = [medoidRow.LogKxx, medoidRow.LogKyy, medoidRow.LogKzz];
        medoidValue = medoidValues(ic);

        histogram(poolValues, edges, 'Normalization', 'probability', ...
            'FaceColor', [0.73 0.75 0.79], 'EdgeColor', 'none', ...
            'FaceAlpha', 0.75);
        hold on
        histogram(selectedValues, edges, 'Normalization', 'probability', ...
            'DisplayStyle', 'stairs', 'EdgeColor', [0.08 0.29 0.56], ...
            'LineWidth', 2.2);
        xline(medoidValue, '-', 'Color', [0.86 0.22 0.16], ...
            'LineWidth', 2.2);
        xline(median(poolValues, 'omitnan'), ':', ...
            'Color', [0.25 0.25 0.25], 'LineWidth', 1.4);

        set(gca, 'FontSize', 14, 'LineWidth', 0.9, ...
            'XLim', [-7 2], 'XTick', xticksValue);
        grid on; box on
        if ic == 1
            title(windowLabels(iw), 'FontWeight', 'bold', ...
                'FontSize', 17, 'Interpreter', 'none');
        end
        if iw == 1
            ylabel(sprintf('log10(%s) probability', componentNames(ic)), ...
                'FontSize', 15, 'Interpreter', 'none');
        end
        if ic == numel(componentNames)
            xlabel('log10(k) [mD]', 'FontSize', 15);
        end
        if ic == 1 && iw == 6
            legend({'Full pool, n=2000', 'Selected slices, n=87', ...
                'Pc medoid sample', 'Pool median'}, ...
                'Location', 'northeast', 'FontSize', 10);
        end
    end
end

sgtitle(['Case01 independent sampling pool vs selected 87 slices | ', ...
    's05 c012 case 012'], 'FontSize', 24, 'FontWeight', 'bold', ...
    'Interpreter', 'none');
saveFigureBoth(fig, figureDir, 'case01_pool_vs_selected_marginal_permeability');
close(fig);
end


function makePcDistancePermFigure(selection, curveMat, results, windows, ...
        windowLabels, figureDir)
% Relate selected-slice permeability to Pc curve distance from medoid.

caseId = 1;
summary = curveMat.summary;
fig = figure('Color', 'w', 'Position', [80, 80, 1600, 820]);
tiledlayout(2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

for iw = 1:numel(windows)
    nexttile
    w = windows(iw);
    mask = summary.Level3CaseId == caseId & summary.Window == w;
    ids = find(mask);
    medoidRow = results.MedoidSummary( ...
        results.MedoidSummary.Level3CaseId == caseId & ...
        results.MedoidSummary.Window == w, :);
    medoidId = medoidRow.MedoidCurveId(1);
    d = nan(numel(ids), 1);
    for i = 1:numel(ids)
        d(i) = curveDistance(curveMat.pcNormalized(ids(i), :), ...
            curveMat.pcNormalized(medoidId, :));
    end

    selectedRows = selection(selection.Window == w, :);
    logGeomK = mean([selectedRows.LogKxx, selectedRows.LogKyy, ...
                     selectedRows.LogKzz], 2, 'omitnan');
    scatter(logGeomK, d, 56, selectedRows.LogKzz, 'filled', ...
        'MarkerFaceAlpha', 0.78, 'MarkerEdgeColor', [0.18 0.18 0.18]);
    hold on
    scatter(mean([medoidRow.LogKxx, medoidRow.LogKyy, medoidRow.LogKzz]), ...
        0, 110, 'p', 'filled', 'MarkerFaceColor', [0.86 0.22 0.16], ...
        'MarkerEdgeColor', 'k');
    set(gca, 'FontSize', 14, 'LineWidth', 0.9);
    grid on; box on
    xlabel('Selected sample mean log10(k)');
    ylabel('Pc distance from medoid');
    title(sprintf('%s | medoid slice %d', windowLabels(iw), ...
        medoidRow.MedoidSliceIndex(1)), 'FontWeight', 'bold', ...
        'Interpreter', 'none');
    cb = colorbar;
    cb.Label.String = 'selected log10(kzz)';
end

sgtitle('Case01 selected permeability vs Pc-curve distance from medoid', ...
    'FontSize', 23, 'FontWeight', 'bold', 'Interpreter', 'none');
saveFigureBoth(fig, figureDir, 'case01_selected_perm_vs_pc_distance');
close(fig);
end


function d = curveDistance(a, b)
% RMS distance in log10 Pc over the common saturation grid.

d = sqrt(mean((log10(a(:)) - log10(b(:))).^2, 'omitnan'));
end


function saveFigureBoth(fig, outputDir, baseName)
% Save a figure as PNG and PDF.

ensureFolder(outputDir);
exportgraphics(fig, fullfile(outputDir, baseName + ".png"), ...
    'Resolution', 220);
print(fig, fullfile(outputDir, char(baseName + ".pdf")), ...
    '-dpdf', '-painters', '-bestfit');
end


function ensureFolder(folderPath)
% Create a directory when needed.

if exist(folderPath, 'dir') ~= 7
    mkdir(folderPath);
end
end
