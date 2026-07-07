%PLOT_CASE01_ENTRY_PRESSURE_PDFS Plot fine-cell entry-pressure PDFs by window.
%
% This diagnostic uses the exact replayed realizations from the full-87 Pc
% pilot for case01. For every selected slice/window, it computes the same
% normalized fine-cell entry-pressure proxy used in the screening Pc
% upscaling:
%
%   entry = sqrt(phi / kChar), kChar = geometric mean(kxx, kyy, kzz)
%
% Each slice is normalized by its own median entry pressure. The plotted
% variable is log10(entry / median(entry)), so the figure emphasizes the
% within-slice entry-pressure contrast and high-entry-pressure tails.

clear; clc;

outputRoot = fullfile('D:', 'codex_gom', 'UQ_workflow', ...
    'pc_upscaling_median_examples_full87');
tableDir = fullfile(outputRoot, 'tables');
figureDir = fullfile(outputRoot, 'figures', 'case01_entry_pressure_diagnostics');
ensureFolder(figureDir);

replayContextFile = fullfile(tableDir, ...
    'replay_summary_with_full87_context_s05_c012_cases_01_03_04_07.csv');
assert(exist(replayContextFile, 'file') == 2, ...
    'Missing replay context table: %s', replayContextFile);

T = readtable(replayContextFile, 'TextType', 'string');
T = T(T.Level3CaseId == 1, :);
windows = ["famp1", "famp2", "famp3", "famp4", "famp5", "famp6"];
windowLabels = ["W1", "W2", "W3", "W4", "W5", "W6"];

edges = -5:0.10:5;
centers = 0.5 * (edges(1:end-1) + edges(2:end));
binWidth = edges(2) - edges(1);

allWindowStats = cell(numel(windows), 1);
allWindowPdfs = cell(numel(windows), 1);

for iw = 1:numel(windows)
    rows = T(T.Window == windows(iw), :);
    assert(height(rows) == 87, 'Expected 87 slices for %s.', windows(iw));
    pdfs = nan(height(rows), numel(centers));
    statRows = cell(height(rows), 9);
    nValid = 0;

    for ir = 1:height(rows)
        try
            S = load(char(rows.OutputFile(ir)), 'replay');
        catch ME
            warning('Skipping unreadable replay file for %s slice %d: %s', ...
                rows.Window(ir), rows.SliceIndex(ir), ME.message);
            continue
        end
        [logEntry, weights] = computeLogEntryPressure(S.replay);
        binMass = weightedHistogram(logEntry, weights, edges);
        nValid = nValid + 1;
        pdfs(nValid, :) = binMass ./ binWidth;

        q = weightedQuantile(logEntry, weights, [0.05, 0.50, 0.95, 0.99]);
        statRows(nValid, :) = { ...
            rows.Window(ir), rows.SliceIndex(ir), rows.SelectedSampleIndex(ir), ...
            q(1), q(2), q(3), q(4), ...
            sum(weights(logEntry > 1)), sum(weights(logEntry > 2))};
    end

    assert(nValid > 0, 'No readable replay files for %s.', windows(iw));
    pdfs = pdfs(1:nValid, :);
    allWindowPdfs{iw} = pdfs;
    allWindowStats{iw} = cell2table(statRows(1:nValid, :), 'VariableNames', ...
        {'Window', 'SliceIndex', 'SelectedSampleIndex', ...
         'LogEntryP05', 'LogEntryP50', 'LogEntryP95', 'LogEntryP99', ...
         'TailMassAbove10x', 'TailMassAbove100x'});
end

sliceStats = vertcat(allWindowStats{:});
writetable(sliceStats, fullfile(tableDir, ...
    'case01_entry_pressure_slice_summary.csv'));

windowStats = summarizeWindows(sliceStats, windows);
writetable(windowStats, fullfile(tableDir, ...
    'case01_entry_pressure_window_summary.csv'));

makePdfFigure(allWindowPdfs, windowStats, windows, windowLabels, centers, figureDir);
makeTailFigure(windowStats, windowLabels, figureDir);

fprintf('Saved case01 entry-pressure diagnostics to: %s\n', figureDir);


function binMass = weightedHistogram(x, weights, edges)
% Compute weighted histogram bin masses for MATLAB versions without weights.

binId = discretize(x(:), edges);
valid = isfinite(binId);
binMass = accumarray(binId(valid), weights(valid), ...
    [numel(edges) - 1, 1], @sum, 0);
binMass = binMass(:)';
end


function [logEntry, weights] = computeLogEntryPressure(replay)
% Compute pore-volume-weighted log10 normalized entry pressure.

pcOpt.minPoro = 1.0e-4;
pcOpt.minPermMD = 1.0e-9;
pcOpt.mDInM2 = 9.869233e-16;

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
        isfield(replay.G.cells, 'volumes') && ...
        numel(replay.G.cells.volumes) == numel(poroAll)
    bulkVolumeAll = replay.G.cells.volumes(:);
else
    bulkVolumeAll = ones(size(poroAll));
end

poreWeights = poroAll .* bulkVolumeAll;
valid = isfinite(poreWeights) & poreWeights > 0 & ...
    isfinite(kCharMD) & isfinite(poroAll);
entry = sqrt(poroAll(valid) ./ kCharMD(valid));
weights = poreWeights(valid);

entry = entry ./ median(entry, 'omitnan');
entry = max(entry, 1.0e-12);
weights = weights ./ sum(weights);
logEntry = log10(entry);
end


function q = weightedQuantile(x, w, probs)
% Weighted quantiles for finite values with nonnegative weights.

x = x(:);
w = w(:);
mask = isfinite(x) & isfinite(w) & w > 0;
x = x(mask);
w = w(mask);
[x, order] = sort(x);
w = w(order);
cw = cumsum(w) ./ sum(w);
q = interp1(cw, x, probs, 'linear', 'extrap');
end


function windowStats = summarizeWindows(sliceStats, windows)
% Summarize tail metrics across the 87 slices for each window.

rows = cell(numel(windows), 9);
for iw = 1:numel(windows)
    mask = sliceStats.Window == windows(iw);
    rows(iw, :) = { ...
        windows(iw), sum(mask), ...
        median(sliceStats.LogEntryP95(mask), 'omitnan'), ...
        median(sliceStats.LogEntryP99(mask), 'omitnan'), ...
        max(sliceStats.LogEntryP99(mask), [], 'omitnan'), ...
        median(sliceStats.TailMassAbove10x(mask), 'omitnan'), ...
        median(sliceStats.TailMassAbove100x(mask), 'omitnan'), ...
        max(sliceStats.TailMassAbove10x(mask), [], 'omitnan'), ...
        max(sliceStats.TailMassAbove100x(mask), [], 'omitnan')};
end

windowStats = cell2table(rows, 'VariableNames', ...
    {'Window', 'NumSlices', 'MedianLogEntryP95', 'MedianLogEntryP99', ...
     'MaxLogEntryP99', 'MedianTailMassAbove10x', ...
     'MedianTailMassAbove100x', 'MaxTailMassAbove10x', ...
     'MaxTailMassAbove100x'});
end


function makePdfFigure(allWindowPdfs, windowStats, windows, windowLabels, ...
        centers, figureDir)
% Plot slice-level and mean entry-pressure PDFs for every window.

fig = figure('Color', 'w', 'Position', [70, 70, 1550, 850]);
tiledlayout(2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

for iw = 1:numel(windows)
    nexttile
    pdfs = allWindowPdfs{iw};
    plot(centers, max(pdfs, 1.0e-8)', ...
        'Color', [0.72 0.75 0.80], 'LineWidth', 0.45);
    hold on
    meanPdf = mean(pdfs, 1, 'omitnan');
    plot(centers, max(meanPdf, 1.0e-8), ...
        'Color', [0.86 0.22 0.16], 'LineWidth', 3.0);
    xline(0, ':', 'Color', [0.10 0.10 0.10], 'LineWidth', 1.4);

    stats = windowStats(windowStats.Window == windows(iw), :);
    xline(stats.MedianLogEntryP95, '--', ...
        'Color', [0.08 0.29 0.56], 'LineWidth', 1.8);
    xline(stats.MedianLogEntryP99, '-.', ...
        'Color', [0.08 0.45 0.24], 'LineWidth', 1.8);

    set(gca, 'YScale', 'log', 'FontSize', 15, 'LineWidth', 1.0, ...
        'XLim', [-5 5], 'YLim', [1.0e-5 1.0e2]);
    grid on; box on
    title(sprintf('%s | median P95=10^{%.2f}, P99=10^{%.2f}', ...
        windowLabels(iw), stats.MedianLogEntryP95, ...
        stats.MedianLogEntryP99), ...
        'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'tex');
    xlabel('log_{10}(entry pressure / slice median)');
    ylabel('Probability density');
    if iw == 1
        legend({'87 slice PDFs', 'Mean PDF', 'Slice median', ...
            'Median P95', 'Median P99'}, ...
            'Location', 'southwest', 'FontSize', 10);
    end
end

sgtitle(['Case01 fine-cell entry-pressure PDFs by window ', ...
    '(normalized within each slice)'], ...
    'FontSize', 24, 'FontWeight', 'bold');
saveFigureBoth(fig, figureDir, 'case01_entry_pressure_pdfs_by_window');
close(fig);
end


function makeTailFigure(windowStats, windowLabels, figureDir)
% Plot compact tail metrics for presentation.

fig = figure('Color', 'w', 'Position', [100, 100, 980, 500]);
x = 1:height(windowStats);
bar(x - 0.18, windowStats.MedianLogEntryP99, 0.34, ...
    'FaceColor', [0.22 0.45 0.70], 'EdgeColor', 'none');
hold on
bar(x + 0.18, windowStats.MaxLogEntryP99, 0.34, ...
    'FaceColor', [0.86 0.45 0.13], 'EdgeColor', 'none');
set(gca, 'XTick', x, 'XTickLabel', cellstr(windowLabels), ...
    'FontSize', 16, 'LineWidth', 1.0);
grid on; box on
ylabel('P99 of log_{10}(entry pressure / slice median)');
xlabel('Throw window');
title('Case01 entry-pressure tail strength by window', ...
    'FontSize', 22, 'FontWeight', 'bold');
legend({'Median across slices', 'Maximum across slices'}, 'Location', 'northwest', ...
    'FontSize', 13);
saveFigureBoth(fig, figureDir, 'case01_entry_pressure_tail_summary');
close(fig);
end


function saveFigureBoth(fig, outputDir, baseName)
% Save a figure as PNG and PDF.

ensureFolder(outputDir);
axesList = findall(fig, 'Type', 'axes');
for ia = 1:numel(axesList)
    try
        axesList(ia).Toolbar.Visible = 'off';
    catch
    end
end
drawnow
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
