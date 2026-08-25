function summary = generate_si_predict_convergence_summary()
%GENERATE_SI_PREDICT_CONVERGENCE_SUMMARY Build the SI convergence figure.
%
% The source data are the completed cell_union_psmear reference-floor
% convergence tables. Each repeated small-ensemble score is normalized by
% the matching window-, component-, and metric-specific reference-floor
% median. The plotted line and band summarize all six windows, all three
% permeability components, and all 30 repeats at each tested sample size.
%
% Outputs:
%   paper/supplement/figures/predict_convergence_reference_floor_summary.pdf
%   paper/supplement/figures/predict_convergence_reference_floor_summary.png
%   paper/supplement/tables/predict_convergence_reference_floor_summary.csv
%   paper/supplement/tables/predict_convergence_median_distance_mae_by_window.csv
%   paper/supplement/tables/predict_convergence_median_distance_wasserstein_by_window.csv

toolsDir = fileparts(mfilename('fullpath'));
paperDir = fileparts(toolsDir);
repoRoot = fileparts(paperDir);
sourceDir = fullfile(repoRoot, 'examples', ...
    'gom_reference_floor_cell_union_psmear_full', 'tables');
figureDir = fullfile(paperDir, 'supplement', 'figures');
tableDir = fullfile(paperDir, 'supplement', 'tables');

repeatFile = fullfile(sourceDir, 'repeat_score_long.csv');
floorFile = fullfile(sourceDir, 'reference_floor_long.csv');
fitFile = fullfile(sourceDir, 'fit_summary_long.csv');
convergenceFile = fullfile(sourceDir, 'convergence_summary_long.csv');
assert(isfile(repeatFile), 'Missing convergence table: %s', repeatFile);
assert(isfile(floorFile), 'Missing convergence table: %s', floorFile);
assert(isfile(fitFile), 'Missing convergence table: %s', fitFile);
assert(isfile(convergenceFile), ...
    'Missing convergence table: %s', convergenceFile);

if ~isfolder(figureDir)
    mkdir(figureDir);
end
if ~isfolder(tableDir)
    mkdir(tableDir);
end

repeatScores = readtable(repeatFile, 'TextType', 'string');
referenceFloor = readtable(floorFile, 'TextType', 'string');
fitSummary = readtable(fitFile, 'TextType', 'string');
convergenceSummary = readtable(convergenceFile, 'TextType', 'string');

floorByKey = unique(referenceFloor(:, ...
    {'Window', 'Metric', 'Component', 'FloorMedian'}), 'rows');
normalized = innerjoin(repeatScores, floorByKey, ...
    'Keys', {'Window', 'Metric', 'Component'});
normalized.NormalizedScore = normalized.Score ./ normalized.FloorMedian;

metrics = ["MAE", "Wasserstein"];
panelTitles = ["(a) Histogram-probability MAE", ...
    "(b) Wasserstein-1 distance"];
lineColor = [0.10 0.10 0.10];
bandColor = [0.86 0.86 0.86];
floorColor = [0.38 0.38 0.38];
textSize = 11;
testNsims = unique(normalized.Nsim);

numMetrics = numel(metrics);
medianRatio = nan(numel(testNsims), numMetrics);
p10Ratio = nan(numel(testNsims), numMetrics);
p90Ratio = nan(numel(testNsims), numMetrics);
summaryRows = cell(numMetrics, 7);

fig = figure('Color', 'w', 'Units', 'inches', ...
    'Position', [1 1 7.2 3.35]);
panelPositions = [0.065 0.145 0.390 0.700; ...
                  0.545 0.145 0.390 0.700];

for im = 1:numMetrics
    metric = metrics(im);
    for in = 1:numel(testNsims)
        mask = normalized.Metric == metric & normalized.Nsim == testNsims(in);
        values = normalized.NormalizedScore(mask);
        medianRatio(in, im) = median(values, 'omitnan');
        p10Ratio(in, im) = empiricalQuantile(values, 0.10);
        p90Ratio(in, im) = empiricalQuantile(values, 0.90);
    end

    [pooledFitIntercept, pooledFitSlope, pooledFitR2] = ...
        fitLogLogSeries(testNsims, medianRatio(:, im));
    fitNsims = logspace(log10(min(testNsims)), ...
        log10(max(testNsims)), 200)';
    fitRatios = 10.^(pooledFitIntercept + ...
        pooledFitSlope .* log10(fitNsims));

    ax = axes(fig, 'Units', 'normalized', ...
        'Position', panelPositions(im, :));
    hold(ax, 'on');
    band = fill(ax, ...
        [testNsims; flipud(testNsims)], ...
        [p10Ratio(:, im); flipud(p90Ratio(:, im))], ...
        bandColor, 'FaceAlpha', 0.50, 'EdgeColor', 'none');
    pooledFitLine = plot(ax, fitNsims, fitRatios, '-', ...
        'Color', lineColor, 'LineWidth', 1.35);
    floorLine = yline(ax, 1, '--', 'Color', floorColor, ...
        'LineWidth', 1.0);

    set(ax, 'XScale', 'log', 'YScale', 'log', ...
        'FontSize', textSize, 'LineWidth', 0.72, ...
        'XColor', [0 0 0], 'YColor', [0 0 0], ...
        'TickLabelInterpreter', 'latex', 'TickDir', 'in', ...
        'TickLength', [0.012 0.012], 'Box', 'off', ...
        'Layer', 'top');
    grid(ax, 'on');
    ax.XGrid = 'off';
    ax.YGrid = 'on';
    ax.GridColor = [0.82 0.82 0.82];
    ax.GridAlpha = 0.18;
    ax.MinorGridAlpha = 0;
    ax.XMinorGrid = 'off';
    ax.YMinorGrid = 'off';
    ax.XMinorTick = 'off';
    ax.YMinorTick = 'off';
    xlim(ax, [20 2000]);
    ylim(ax, [0.5 50]);
    xticks(ax, [20 50 100 200 500 2000]);
    xticklabels(ax, {'$20$', '$50$', '$100$', '$200$', '$500$', '$2000$'});
    xtickangle(ax, 0);
    yticks(ax, [0.5 1 2 5 10 20 50]);
    yticklabels(ax, {'$0.5$', '$1$', '$2$', '$5$', '$10$', '$20$', '$50$'});
    axisLimitsX = xlim(ax);
    axisLimitsY = ylim(ax);
    plot(ax, axisLimitsX, [axisLimitsY(2) axisLimitsY(2)], '-', ...
        'Color', [0 0 0], 'LineWidth', 0.72, 'HandleVisibility', 'off');
    plot(ax, [axisLimitsX(2) axisLimitsX(2)], axisLimitsY, '-', ...
        'Color', [0 0 0], 'LineWidth', 0.72, 'HandleVisibility', 'off');
    medianPoints = plot(ax, testNsims, medianRatio(:, im), 'o', ...
        'Color', lineColor, 'MarkerFaceColor', lineColor, ...
        'MarkerEdgeColor', lineColor, 'MarkerSize', 4.2, ...
        'LineStyle', 'none', 'Clipping', 'off');
    titleHandle = title(ax, panelTitles(im), 'FontSize', textSize, ...
        'FontWeight', 'normal', 'Interpreter', 'latex');
    titleHandle.Units = 'normalized';
    titlePosition = titleHandle.Position;
    titlePosition(2) = 1 + 1.20 .* (titlePosition(2) - 1);
    titleHandle.Position = titlePosition;
    xlabel(ax, 'Ensemble size, $N$', 'FontSize', textSize, ...
        'Interpreter', 'latex');
    if im == 1
        ylabel(ax, 'Normalized distance, $D/D_{\mathrm{ref}}$', ...
            'FontSize', textSize, 'Interpreter', 'latex');
        legendHandle = legend(ax, ...
            [band, medianPoints, pooledFitLine, floorLine], ...
            {'10th--90th percentile', 'Pooled median', ...
            'Log--log fit', 'Reference floor'}, ...
            'Orientation', 'horizontal', 'NumColumns', 4, ...
            'FontSize', textSize, 'Box', 'off', 'Interpreter', 'latex');
        legendHandle.TextColor = [0 0 0];
        legendHandle.ItemTokenSize = [16 8];
        legendHandle.Units = 'normalized';
        legendHandle.Position = [0.165 0.925 0.670 0.055];
    end

    fitMask = fitSummary.Metric == metric;
    fitSlopes = fitSummary.FitSlope(fitMask);
    n2000Mask = normalized.Metric == metric & normalized.Nsim == 2000;
    n2000Values = normalized.NormalizedScore(n2000Mask);
    summaryRows(im, :) = {metric, median(n2000Values, 'omitnan'), ...
        empiricalQuantile(n2000Values, 0.10), ...
        empiricalQuantile(n2000Values, 0.90), ...
        median(fitSlopes, 'omitnan'), min(fitSlopes), max(fitSlopes)};

    fitLabel = {sprintf('$\\mathrm{Pooled\\ slope}=%.3f$', ...
        pooledFitSlope), ...
        sprintf('$\\mathrm{Pooled}\\ R^2=%.3f$', pooledFitR2)};
    annotationCenterY = (log10(sqrt(20 * 50)) - log10(axisLimitsY(1))) / ...
        (log10(axisLimitsY(2)) - log10(axisLimitsY(1))) - 0.018;
    text(ax, 0.97, annotationCenterY, fitLabel, 'Units', 'normalized', ...
        'FontSize', textSize, 'VerticalAlignment', 'middle', ...
        'HorizontalAlignment', 'right', ...
        'Color', [0 0 0], 'Interpreter', 'latex');
end

pdfFile = fullfile(figureDir, ...
    'predict_convergence_reference_floor_summary.pdf');
pngFile = fullfile(figureDir, ...
    'predict_convergence_reference_floor_summary.png');
exportgraphics(fig, pdfFile, 'ContentType', 'vector');
exportgraphics(fig, pngFile, 'Resolution', 600);
close(fig);

summary = cell2table(summaryRows, 'VariableNames', ...
    {'Metric', 'N2000MedianFloorRatio', 'N2000P10FloorRatio', ...
     'N2000P90FloorRatio', 'MedianLogLogSlope', ...
     'MinimumLogLogSlope', 'MaximumLogLogSlope'});
writetable(summary, fullfile(tableDir, ...
    'predict_convergence_reference_floor_summary.csv'));

writeWideMedianDistanceTable(convergenceSummary, "MAE", testNsims, ...
    fullfile(tableDir, ...
    'predict_convergence_median_distance_mae_by_window.csv'));
writeWideMedianDistanceTable(convergenceSummary, "Wasserstein", testNsims, ...
    fullfile(tableDir, ...
    'predict_convergence_median_distance_wasserstein_by_window.csv'));

fprintf('Saved %s\n', pdfFile);
fprintf('Saved %s\n', pngFile);
disp(summary);
end


function writeWideMedianDistanceTable(convergenceSummary, metric, testNsims, outputFile)
%WRITEWIDEMEDIANDISTANCETABLE Pivot median distances to component-N by window.

components = ["kxx", "kyy", "kzz"];
windowNames = "famp" + string(1:6);
outputWindowNames = "W" + string(1:6);
numRows = numel(components) * numel(testNsims);

componentColumn = strings(numRows, 1);
nSimColumn = zeros(numRows, 1);
windowValues = nan(numRows, numel(windowNames));

row = 0;
for ic = 1:numel(components)
    for in = 1:numel(testNsims)
        row = row + 1;
        componentColumn(row) = components(ic);
        nSimColumn(row) = testNsims(in);

        for iw = 1:numel(windowNames)
            mask = convergenceSummary.Metric == metric & ...
                convergenceSummary.Component == components(ic) & ...
                convergenceSummary.Nsim == testNsims(in) & ...
                convergenceSummary.Window == windowNames(iw);
            assert(nnz(mask) == 1, ...
                'Expected one summary row for %s, %s, N=%d, %s.', ...
                metric, components(ic), testNsims(in), windowNames(iw));
            windowValues(row, iw) = convergenceSummary.MedianScore(mask);
        end
    end
end

wideTable = table(componentColumn, nSimColumn, ...
    'VariableNames', {'Component', 'Nsim'});
wideTable = [wideTable, array2table(windowValues, ...
    'VariableNames', cellstr(outputWindowNames))];
writetable(wideTable, outputFile);
fprintf('Saved %s\n', outputFile);
end


function q = empiricalQuantile(values, probability)
%EMPIRICALQUANTILE Linearly interpolated quantile without toolboxes.

values = sort(values(isfinite(values)));
if isempty(values)
    q = NaN;
    return
end
if isscalar(values)
    q = values;
    return
end

position = 1 + (numel(values) - 1) * probability;
lowerIndex = floor(position);
upperIndex = ceil(position);
fraction = position - lowerIndex;
q = values(lowerIndex) + ...
    fraction * (values(upperIndex) - values(lowerIndex));
end


function [intercept, slope, r2] = fitLogLogSeries(x, y)
%FITLOGLOGSERIES Fit log10(y) = intercept + slope*log10(x).

x = x(:);
y = y(:);
mask = isfinite(x) & isfinite(y) & x > 0 & y > 0;
assert(nnz(mask) >= 2, 'At least two positive finite points are required.');

logX = log10(x(mask));
logY = log10(y(mask));
coefficients = polyfit(logX, logY, 1);
slope = coefficients(1);
intercept = coefficients(2);

logYFit = polyval(coefficients, logX);
residualSumSquares = sum((logY - logYFit).^2);
totalSumSquares = sum((logY - mean(logY)).^2);
if totalSumSquares > 0
    r2 = 1 - residualSumSquares / totalSumSquares;
else
    r2 = 1;
end
end
