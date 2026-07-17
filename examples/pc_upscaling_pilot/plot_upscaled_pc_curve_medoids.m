function outputFiles = plot_upscaled_pc_curve_medoids( ...
        pcNativeCsv, pcMedoidSummaryCsv, outputDir)
%PLOT_UPSCALED_PC_CURVE_MEDOIDS Plot native Pc curves and full-curve medoids.
%
%   OUTPUTFILES = PLOT_UPSCALED_PC_CURVE_MEDOIDS(PCNATIVECSV,
%   PCMEDOIDSUMMARYCSV, OUTPUTDIR) creates one publication-quality six-panel
%   figure for a single geology and Level 3 case. Every native slice curve is
%   shown in gray, and the full Pc-curve medoid is drawn last as a thick,
%   opaque red line so coincident gray curves remain visible beneath it.
%
%   The medoid is a visualization diagnostic selected using the full Pc-curve
%   distance. It does not replace the 87 slice-specific Pc curves used by the
%   reservoir model and is distinct from the Swi medoid used for Kr upscaling.

arguments
    pcNativeCsv (1, 1) string
    pcMedoidSummaryCsv (1, 1) string
    outputDir (1, 1) string
end

assert(isfile(pcNativeCsv), 'Native Pc CSV not found: %s', pcNativeCsv);
assert(isfile(pcMedoidSummaryCsv), ...
    'Pc medoid summary CSV not found: %s', pcMedoidSummaryCsv);
if ~isfolder(outputDir)
    mkdir(outputDir);
end

pc = readtable(pcNativeCsv, 'TextType', 'string');
medoids = readtable(pcMedoidSummaryCsv, 'TextType', 'string');
requireColumns(pc, {'CurveId', 'GeologyId', 'Level3CaseId', ...
    'Level3CaseName', 'Window', 'SliceIndex', 'GasSaturation', 'PcBar', ...
    'BulkSgMax'});
requireColumns(medoids, {'GeologyId', 'Level3CaseId', 'Window', ...
    'MedoidCurveId'});

pc.GeologyId = string(pc.GeologyId);
pc.Level3CaseId = double(pc.Level3CaseId);
pc.Level3CaseName = string(pc.Level3CaseName);
pc.Window = string(pc.Window);
medoids.GeologyId = string(medoids.GeologyId);
medoids.Level3CaseId = double(medoids.Level3CaseId);
medoids.Window = string(medoids.Window);

geologyIds = unique(pc.GeologyId);
caseIds = unique(pc.Level3CaseId);
assert(isscalar(geologyIds) && isscalar(caseIds), ...
    'Native Pc input must contain one geology and one Level 3 case.');
assert(all(medoids.GeologyId == geologyIds(1)) && ...
    all(medoids.Level3CaseId == caseIds(1)), ...
    'Native Pc and medoid-summary case identifiers do not match.');

caseId = caseIds(1);
caseName = replace(pc.Level3CaseName(1), '_', ' ');
windows = orderedWindows(pc.Window);
assert(numel(windows) == 6, ...
    'Expected six windows but found %d.', numel(windows));

gray = [0.67, 0.69, 0.72];
grayMarker = [0.49, 0.52, 0.56];
red = [0.88, 0.10, 0.06];
fig = figure('Color', 'w', 'Position', [60, 40, 1800, 1100]);
layout = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', ...
    'Padding', 'compact');
legendHandles = gobjects(3, 1);

for i = 1:numel(windows)
    windowName = windows(i);
    W = pc(pc.Window == windowName, :);
    M = medoids(medoids.Window == windowName, :);
    assert(height(M) == 1, ...
        'Expected one full-curve medoid for %s.', windowName);
    medoidCurveId = double(M.MedoidCurveId(1));
    sources = unique(double(W.CurveId), 'sorted');
    assert(numel(sources) == 87, ...
        'Expected 87 Pc curves for %s but found %d.', ...
        windowName, numel(sources));
    assert(ismember(medoidCurveId, sources), ...
        'Full-curve medoid %d is missing from %s.', ...
        medoidCurveId, windowName);

    ax = nexttile(layout);
    hold(ax, 'on');
    if i == 1
        legendHandles(1) = plot(ax, NaN, NaN, '-', ...
            'Color', gray, 'LineWidth', 1.0);
        legendHandles(2) = plot(ax, NaN, NaN, 'o', ...
            'MarkerFaceColor', grayMarker, 'MarkerEdgeColor', ...
            [0.43, 0.46, 0.50], 'MarkerSize', 5.5);
        legendHandles(3) = plot(ax, NaN, NaN, '-', ...
            'Color', red, 'LineWidth', 4.2);
    end

    endpoints = nan(numel(sources), 1);
    for j = 1:numel(sources)
        C = W(double(W.CurveId) == sources(j), :);
        C = sortrows(C, 'GasSaturation');
        valid = isfinite(C.GasSaturation) & isfinite(C.PcBar) & C.PcBar > 0;
        sg = double(C.GasSaturation(valid));
        pcBar = double(C.PcBar(valid));
        assert(~isempty(sg), 'Pc curve %d contains no valid points.', sources(j));
        plot(ax, sg, pcBar, '-', 'Color', gray, 'LineWidth', 0.65);
        plot(ax, sg(end), pcBar(end), 'o', ...
            'MarkerFaceColor', grayMarker, ...
            'MarkerEdgeColor', grayMarker, 'MarkerSize', 3.2);
        endpoints(j) = double(C.BulkSgMax(1));
    end

    medoidCurve = W(double(W.CurveId) == medoidCurveId, :);
    medoidCurve = sortrows(medoidCurve, 'GasSaturation');
    valid = isfinite(medoidCurve.GasSaturation) & ...
        isfinite(medoidCurve.PcBar) & medoidCurve.PcBar > 0;
    medoidSg = double(medoidCurve.GasSaturation(valid));
    medoidPcBar = double(medoidCurve.PcBar(valid));
    medoidHandle = plot(ax, medoidSg, medoidPcBar, '-', ...
        'Color', red, 'LineWidth', 4.2, 'HandleVisibility', 'off');
    plot(ax, medoidSg(end), medoidPcBar(end), 'o', ...
        'MarkerFaceColor', red, 'MarkerEdgeColor', 'white', ...
        'LineWidth', 0.9, 'MarkerSize', 7.5, 'HandleVisibility', 'off');
    uistack(medoidHandle, 'top');

    stylePcAxis(ax);
    endpointLabel = sprintf('S_{g,max}: %.2f-%.2f', ...
        min(endpoints), max(endpoints));
    title(ax, sprintf('%s | n = %d', upper(windowName), numel(sources)), ...
        endpointLabel, 'FontSize', 16, 'FontWeight', 'bold');
    if i > 3
        xlabel(ax, 'Gas saturation', 'FontSize', 16);
    end
    if i == 1 || i == 4
        ylabel(ax, 'Pc [bar]', 'FontSize', 16);
    end
end

title(layout, {sprintf('Case %02d: %s', caseId, caseName), ...
    'Native-endpoint invasion-percolation Pc curves', ...
    'Gray = 87 slices; red = full Pc-curve medoid (visualization only)'}, ...
    'FontName', 'Arial', 'FontSize', 22, 'FontWeight', 'bold');
lgd = legend(legendHandles, {'87 native slice curves', ...
    'native endpoints', 'full Pc-curve medoid'}, ...
    'Orientation', 'horizontal');
lgd.Layout.Tile = 'south';
lgd.FontName = 'Arial';
lgd.FontSize = 14;

baseName = sprintf('case%02d_upscaled_pc_curves_full_curve_medoid', caseId);
pngFile = fullfile(outputDir, baseName + ".png");
pdfFile = fullfile(outputDir, baseName + ".pdf");
exportgraphics(fig, pngFile, 'Resolution', 300, ...
    'BackgroundColor', 'white');
exportgraphics(fig, pdfFile, 'ContentType', 'vector', ...
    'BackgroundColor', 'white');
close(fig);

outputFiles = struct('png', string(pngFile), 'pdf', string(pdfFile));
fprintf('Saved full Pc-curve medoid figure: %s\n', pngFile);
end


function stylePcAxis(ax)
% Apply fixed publication formatting to one Pc panel.

set(ax, 'YScale', 'log');
xlim(ax, [0, 1]);
ylim(ax, [1e-2, 1e3]);
xticks(ax, 0:0.2:1);
yticks(ax, 10 .^ (-2:3));
grid(ax, 'on');
ax.GridColor = [0.82, 0.82, 0.82];
ax.GridAlpha = 0.55;
ax.MinorGridAlpha = 0.20;
ax.FontName = 'Arial';
ax.FontSize = 14;
ax.LineWidth = 1.0;
box(ax, 'on');
pbaspect(ax, [1.30, 1.0, 1.0]);
end


function windows = orderedWindows(values)
% Return unique window labels ordered by numeric suffix.

windows = unique(string(values), 'stable');
numbers = arrayfun(@windowNumber, windows);
[~, order] = sort(numbers);
windows = windows(order);
end


function n = windowNumber(windowName)
% Extract the numeric suffix from a window label.

token = regexp(char(windowName), '(\d+)$', 'tokens', 'once');
assert(~isempty(token), 'Window label lacks a numeric suffix: %s', windowName);
n = str2double(token{1});
end


function requireColumns(T, required)
% Require every named column in a table.

missing = setdiff(required, T.Properties.VariableNames);
assert(isempty(missing), 'Input table lacks columns: %s', ...
    strjoin(missing, ', '));
end
