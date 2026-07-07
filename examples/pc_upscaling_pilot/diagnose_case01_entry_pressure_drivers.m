%DIAGNOSE_CASE01_ENTRY_PRESSURE_DRIVERS Quantify why W1/W6 Pc tails vary.
%
% The entry-pressure proxy used in the Pc screening is
%
%   entry = sqrt(phi / kChar)
%
% where kChar is the geometric mean of kxx, kyy, and kzz.  This script
% decomposes log10(entry / slice median) into permeability and porosity
% terms for each selected case01 slice, then summarizes which windows have
% rare high-entry-pressure tails and what fine-cell properties drive them.

clear; clc;

outputRoot = fullfile('D:', 'codex_gom', 'UQ_workflow', ...
    'pc_upscaling_median_examples_full87');
tableDir = fullfile(outputRoot, 'tables');
figureDir = fullfile(outputRoot, 'figures', 'case01_entry_pressure_diagnostics');
ensureFolder(tableDir);
ensureFolder(figureDir);

contextFile = fullfile(tableDir, ...
    'replay_summary_with_full87_context_s05_c012_cases_01_03_04_07.csv');
assert(exist(contextFile, 'file') == 2, 'Missing context file: %s', contextFile)

T = readtable(contextFile, 'TextType', 'string');
T = T(T.Level3CaseId == 1, :);
windows = ["famp1", "famp2", "famp3", "famp4", "famp5", "famp6"];
windowLabels = ["W1", "W2", "W3", "W4", "W5", "W6"];

sliceRows = {};
topTailRows = {};

for i = 1:height(T)
    row = T(i, :);
    S = load(char(row.OutputFile), 'replay');
    D = computeEntryDriverVectors(S.replay);

    qEntry = weightedQuantile(D.logEntry, D.weights, [0.01 0.05 0.50 0.95 0.99]);
    qK = weightedQuantile(D.logKChar, D.weights, [0.01 0.05 0.50 0.95 0.99]);
    qPhi = weightedQuantile(D.logPhi, D.weights, [0.01 0.50 0.99]);

    tail10 = D.logEntry > 1;
    tail100 = D.logEntry > 2;
    tailForDrivers = D.logEntry >= qEntry(5);

    sliceRows(end+1, :) = { ...
        row.Window, row.SliceIndex, row.SelectedSampleIndex, ...
        qEntry(1), qEntry(2), qEntry(3), qEntry(4), qEntry(5), ...
        qK(1), qK(2), qK(3), qK(4), qK(5), ...
        qPhi(1), qPhi(2), qPhi(3), ...
        sum(D.weights(tail10)), sum(D.weights(tail100)), ...
        weightedMean(D.logEntry(tailForDrivers), D.weights(tailForDrivers)), ...
        weightedMean(D.kTerm(tailForDrivers), D.weights(tailForDrivers)), ...
        weightedMean(D.phiTerm(tailForDrivers), D.weights(tailForDrivers)), ...
        weightedMean(D.residualTerm(tailForDrivers), D.weights(tailForDrivers)), ...
        weightedMean(double(D.isSmear(tailForDrivers)), D.weights(tailForDrivers)), ...
        weightedMean(D.vcl(tailForDrivers), D.weights(tailForDrivers))}; %#ok<AGROW>

    if any(tail100)
        topTailRows(end+1, :) = { ...
            row.Window, row.SliceIndex, row.SelectedSampleIndex, ...
            sum(D.weights(tail100)), ...
            weightedMean(D.logEntry(tail100), D.weights(tail100)), ...
            weightedMean(D.kTerm(tail100), D.weights(tail100)), ...
            weightedMean(D.phiTerm(tail100), D.weights(tail100)), ...
            weightedMean(D.residualTerm(tail100), D.weights(tail100)), ...
            weightedMean(double(D.isSmear(tail100)), D.weights(tail100)), ...
            weightedMean(D.vcl(tail100), D.weights(tail100)), ...
            weightedQuantile(D.logKChar(tail100), D.weights(tail100), 0.50), ...
            weightedQuantile(D.logPhi(tail100), D.weights(tail100), 0.50)}; %#ok<AGROW>
    end
end

sliceStats = cell2table(sliceRows, 'VariableNames', ...
    {'Window', 'SliceIndex', 'SelectedSampleIndex', ...
     'LogEntryP01', 'LogEntryP05', 'LogEntryP50', 'LogEntryP95', 'LogEntryP99', ...
     'LogKCharP01', 'LogKCharP05', 'LogKCharP50', 'LogKCharP95', 'LogKCharP99', ...
     'LogPhiP01', 'LogPhiP50', 'LogPhiP99', ...
     'TailMassAbove10x', 'TailMassAbove100x', ...
     'Top1MeanLogEntry', 'Top1MeanKTerm', 'Top1MeanPhiTerm', 'Top1MeanResidual', ...
     'Top1SmearFraction', 'Top1MeanVcl'});

topTailStats = cell2table(topTailRows, 'VariableNames', ...
    {'Window', 'SliceIndex', 'SelectedSampleIndex', 'TailMassAbove100x', ...
     'Tail100MeanLogEntry', 'Tail100MeanKTerm', 'Tail100MeanPhiTerm', ...
     'Tail100MeanResidual', 'Tail100SmearFraction', 'Tail100MeanVcl', ...
     'Tail100MedianLogKChar', 'Tail100MedianLogPhi'});

windowStats = summarizeWindows(sliceStats, topTailStats, windows);

writetable(sliceStats, fullfile(tableDir, ...
    'case01_entry_pressure_driver_slice_summary.csv'));
writetable(topTailStats, fullfile(tableDir, ...
    'case01_entry_pressure_driver_gt100x_tail_slices.csv'));
writetable(windowStats, fullfile(tableDir, ...
    'case01_entry_pressure_driver_window_summary.csv'));

makeDriverSummaryFigure(windowStats, windowLabels, figureDir);

fprintf('Saved entry-pressure driver diagnostics to: %s\n', tableDir);


function D = computeEntryDriverVectors(replay)
% Compute normalized entry pressure and its permeability/porosity drivers.

minPoro = 1.0e-4;
minPermMD = 1.0e-9;
mDInM2 = 9.869233e-16;

poro = max(replay.Grid.poro(:), minPoro);
perm = replay.Grid.perm;
if size(perm, 2) >= 6
    kSI = perm(:, [1 4 6]);
elseif size(perm, 2) >= 3
    kSI = perm(:, 1:3);
else
    kSI = repmat(perm(:, 1), 1, 3);
end
kSI = max(kSI, minPermMD * mDInM2);
kCharMD = exp(mean(log(kSI ./ mDInM2), 2));
kCharMD = max(kCharMD, minPermMD);

if isfield(replay, 'G') && isfield(replay.G, 'cells') && ...
        isfield(replay.G.cells, 'volumes') && ...
        numel(replay.G.cells.volumes) == numel(poro)
    bulkVolume = replay.G.cells.volumes(:);
else
    bulkVolume = ones(size(poro));
end

weights = poro .* bulkVolume;
valid = isfinite(weights) & weights > 0 & ...
    isfinite(kCharMD) & isfinite(poro);

logPhi = log10(poro(valid));
logKChar = log10(kCharMD(valid));
rawLogEntry = 0.5 * (logPhi - logKChar);
logEntry = rawLogEntry - median(rawLogEntry, 'omitnan');

% Driver terms are centered separately; residual accounts for non-additive
% medians and is reported so the decomposition remains transparent.
kTerm = -0.5 * (logKChar - median(logKChar, 'omitnan'));
phiTerm = 0.5 * (logPhi - median(logPhi, 'omitnan'));
residualTerm = logEntry - kTerm - phiTerm;

D = struct();
D.logEntry = logEntry;
D.logKChar = logKChar;
D.logPhi = logPhi;
D.kTerm = kTerm;
D.phiTerm = phiTerm;
D.residualTerm = residualTerm;
D.weights = weights(valid) ./ sum(weights(valid));
D.vcl = replay.Grid.vcl(valid);
if isfield(replay.Grid, 'isSmear')
    D.isSmear = logical(replay.Grid.isSmear(valid));
else
    D.isSmear = false(size(D.logEntry));
end
end


function q = weightedQuantile(x, w, probs)
% Weighted quantiles for finite values with nonnegative weights.

x = x(:);
w = w(:);
mask = isfinite(x) & isfinite(w) & w > 0;
x = x(mask);
w = w(mask);
if isempty(x)
    q = nan(size(probs));
    return
end
[x, order] = sort(x);
w = w(order);
cw = cumsum(w) ./ sum(w);
q = interp1(cw, x, probs, 'linear', 'extrap');
end


function y = weightedMean(x, w)
% Weighted mean with finite positive weights.

x = x(:);
w = w(:);
mask = isfinite(x) & isfinite(w) & w > 0;
if ~any(mask)
    y = NaN;
else
    y = sum(x(mask) .* w(mask)) ./ sum(w(mask));
end
end


function windowStats = summarizeWindows(sliceStats, topTailStats, windows)
% Aggregate slice-level diagnostics by window.

rows = cell(numel(windows), 15);
for iw = 1:numel(windows)
    mask = sliceStats.Window == windows(iw);
    tailMask = topTailStats.Window == windows(iw);
    rows(iw, :) = { ...
        windows(iw), sum(mask), ...
        median(sliceStats.LogEntryP99(mask), 'omitnan'), ...
        max(sliceStats.LogEntryP99(mask), [], 'omitnan'), ...
        sum(sliceStats.TailMassAbove10x(mask) > 0), ...
        sum(sliceStats.TailMassAbove100x(mask) > 0), ...
        max(sliceStats.TailMassAbove10x(mask), [], 'omitnan'), ...
        max(sliceStats.TailMassAbove100x(mask), [], 'omitnan'), ...
        median(sliceStats.LogKCharP01(mask), 'omitnan'), ...
        min(sliceStats.LogKCharP01(mask), [], 'omitnan'), ...
        median(sliceStats.LogPhiP99(mask) - sliceStats.LogPhiP01(mask), 'omitnan'), ...
        median(sliceStats.LogKCharP99(mask) - sliceStats.LogKCharP01(mask), 'omitnan'), ...
        median(sliceStats.Top1MeanKTerm(mask), 'omitnan'), ...
        median(sliceStats.Top1MeanPhiTerm(mask), 'omitnan'), ...
        median(topTailStats.Tail100SmearFraction(tailMask), 'omitnan')};
end

windowStats = cell2table(rows, 'VariableNames', ...
    {'Window', 'NumSlices', 'MedianLogEntryP99', 'MaxLogEntryP99', ...
     'NumSlicesWithTailAbove10x', 'NumSlicesWithTailAbove100x', ...
     'MaxTailMassAbove10x', 'MaxTailMassAbove100x', ...
     'MedianLogKCharP01', 'MinLogKCharP01', ...
     'MedianLogPhiRange99_01', 'MedianLogKCharRange99_01', ...
     'MedianTop1KTerm', 'MedianTop1PhiTerm', 'MedianTail100SmearFraction'});
end


function makeDriverSummaryFigure(windowStats, windowLabels, figureDir)
% Plot the core quantitative diagnostics behind W1/W6 variability.

fig = figure('Color', 'w', 'Position', [80, 80, 1450, 620]);
tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
x = 1:height(windowStats);

nexttile
bar(x - 0.18, windowStats.NumSlicesWithTailAbove10x, 0.34, ...
    'FaceColor', [0.26 0.48 0.70], 'EdgeColor', 'none');
hold on
bar(x + 0.18, windowStats.NumSlicesWithTailAbove100x, 0.34, ...
    'FaceColor', [0.86 0.37 0.34], 'EdgeColor', 'none');
formatAxis(windowLabels)
ylabel('Number of slices');
title('How often high-entry tails appear');
legend({'>10x median', '>100x median'}, 'Location', 'northwest');

nexttile
bar(x - 0.18, windowStats.MedianLogEntryP99, 0.34, ...
    'FaceColor', [0.40 0.64 0.82], 'EdgeColor', 'none');
hold on
bar(x + 0.18, windowStats.MaxLogEntryP99, 0.34, ...
    'FaceColor', [0.93 0.69 0.13], 'EdgeColor', 'none');
formatAxis(windowLabels)
ylabel('P99 of log_{10}(entry / median)');
title('Typical vs extreme slice tail strength');
legend({'Median slice P99', 'Maximum slice P99'}, 'Location', 'northwest');

nexttile
bar(x - 0.18, windowStats.MedianTop1KTerm, 0.34, ...
    'FaceColor', [0.35 0.60 0.35], 'EdgeColor', 'none');
hold on
bar(x + 0.18, windowStats.MedianTop1PhiTerm, 0.34, ...
    'FaceColor', [0.55 0.42 0.70], 'EdgeColor', 'none');
formatAxis(windowLabels)
ylabel('Contribution to top-1% log entry');
title('Driver decomposition');
legend({'Permeability term', 'Porosity term'}, 'Location', 'northwest');

sgtitle('Case01 entry-pressure variability drivers', ...
    'FontSize', 24, 'FontWeight', 'bold');

saveFigureBoth(fig, figureDir, 'case01_entry_pressure_driver_summary');
close(fig);
end


function formatAxis(windowLabels)
% Consistent axis style.

set(gca, 'XTick', 1:numel(windowLabels), 'XTickLabel', cellstr(windowLabels), ...
    'FontSize', 15, 'LineWidth', 1.0);
grid on; box on
end


function saveFigureBoth(fig, outputDir, baseName)
% Save figure as PNG/PDF without MATLAB axes-toolbar overlays.

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
% Create a folder if it does not already exist.

if exist(folderPath, 'dir') ~= 7
    mkdir(folderPath);
end
end
