%REGENERATE_PC_FULL87_FIGURES Rebuild Pc review figures from saved results.
%
% This lightweight script does not replay PREDICT and does not recompute Pc
% curves. It only reloads the saved full-87 Pc curve and medoid result MAT
% files from run_pc_upscaling_full_median_examples.m, then regenerates the
% review figures. Use it after changing labels or figure formatting.

clear; clc;

outputRoot = fullfile('D:', 'codex_gom', 'UQ_workflow', ...
    'pc_upscaling_median_examples_full87');
curveMatFile = fullfile(outputRoot, 'curves', ...
    'pc_curves_s05_c012_cases_01_03_04_07_full87.mat');
resultMatFile = fullfile(outputRoot, 'tables', ...
    'pc_medoid_results_s05_c012_cases_01_03_04_07_full87.mat');
figureDir = fullfile(outputRoot, 'figures');

assert(exist(curveMatFile, 'file') == 2, 'Missing curve MAT: %s', curveMatFile);
assert(exist(resultMatFile, 'file') == 2, 'Missing result MAT: %s', resultMatFile);
ensureFolder(figureDir);

curveData = load(curveMatFile, 'curveMat');
resultData = load(resultMatFile, 'results');
curveMat = curveData.curveMat;
results = resultData.results;

makeFull87Figures(curveMat, results, figureDir);
fprintf('Regenerated full-87 Pc figures in: %s\n', figureDir);


function makeFull87Figures(curveMat, results, figureDir)
% Create one 2-by-3 Pc-curve figure per representative Level-3 case.

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
caseIdx = find(summary.Level3CaseId == caseId, 1);
caseName = prettyLabel(summary.Level3CaseName(caseIdx));
caseCategory = prettyLabel(summary.CaseCategory(caseIdx));

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
    'FontSize', 20, 'FontWeight', 'bold', 'Interpreter', 'none');
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


function label = prettyLabel(value)
% Convert workflow identifiers into readable figure labels.

label = strrep(string(value), '_', ' ');
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
% Save a figure as PNG and PDF with stable filenames.

ensureFolder(outputDir);
pngFile = fullfile(outputDir, baseName + ".png");
pdfFile = fullfile(outputDir, baseName + ".pdf");
exportgraphics(fig, pngFile, 'Resolution', 220);
exportgraphics(fig, pdfFile, 'ContentType', 'vector');
end


function ensureFolder(folderPath)
% Create a directory if needed.

if exist(folderPath, 'dir') ~= 7
    mkdir(folderPath);
end
end
