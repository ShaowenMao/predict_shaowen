function outputFiles = plot_pc_guided_kr_validation_case( ...
        pcNativeCsv, fullKrSummaryCsv, representativeKrSummaryCsv, outputDir)
%PLOT_PC_GUIDED_KR_VALIDATION_CASE Plot Pc and Kr validation families.
%
%   OUTPUTFILES = PLOT_PC_GUIDED_KR_VALIDATION_CASE(PCNATIVECSV,
%   FULLKRSUMMARYCSV, REPRESENTATIVEKRSUMMARYCSV, OUTPUTDIR) creates three
%   six-panel
%   figures for one geology and one Level 3 case:
%     1. native-endpoint invasion-percolation Pc curves;
%     2. native-endpoint full dynamic-Kr curves;
%     3. one representative normalized dynamic-Kr shape per window mapped
%        back to every slice's physical Pc-derived saturation endpoint.
%
%   In every panel, the curve selected by the Pc median-effective-Swi rule
%   is highlighted. The function writes PNG and PDF versions and does not
%   modify the underlying upscaling data.

arguments
    pcNativeCsv (1, 1) string
    fullKrSummaryCsv (1, 1) string
    representativeKrSummaryCsv (1, 1) string
    outputDir (1, 1) string
end

assert(isfile(pcNativeCsv), 'Pc curve CSV not found: %s', pcNativeCsv);
assert(isfile(fullKrSummaryCsv), ...
    'Full Kr summary CSV not found: %s', fullKrSummaryCsv);
assert(isfile(representativeKrSummaryCsv), ...
    'Representative Kr summary CSV not found: %s', ...
    representativeKrSummaryCsv);
if ~isfolder(outputDir)
    mkdir(outputDir);
end

pc = readtable(pcNativeCsv, 'TextType', 'string');
fullKr = readtable(fullKrSummaryCsv, 'TextType', 'string');
representativeKr = readtable(representativeKrSummaryCsv, ...
    'TextType', 'string');
requireColumns(pc, {'ReplaySourceRow', 'GeologyId', 'Level3CaseId', ...
    'Level3CaseName', 'Window', 'SliceIndex', 'GasSaturation', 'PcBar', ...
    'BulkSgMax'});
requireColumns(fullKr, {'SourceRow', 'GeologyId', 'Level3CaseId', ...
    'Level3CaseName', 'Window', 'SliceIndex', 'PcMaxSg', ...
    'IrreducibleWaterSaturation', 'BrineCoreyExponent', ...
    'GasCoreyExponent'});
requireColumns(representativeKr, {'SourceRow', 'GeologyId', ...
    'Level3CaseId', 'Level3CaseName', 'Window', 'SliceIndex', 'PcMaxSg', ...
    'IrreducibleWaterSaturation', 'BrineCoreyExponent', ...
    'GasCoreyExponent'});

geologyIds = unique(fullKr.GeologyId);
caseIds = unique(fullKr.Level3CaseId);
assert(isscalar(geologyIds) && isscalar(caseIds), ...
    'Inputs must contain one geology and one Level 3 case.');
assert(all(pc.GeologyId == geologyIds(1)) && ...
    all(representativeKr.GeologyId == geologyIds(1)), ...
    'Pc, full Kr, and representative Kr geology IDs do not match.');
assert(all(pc.Level3CaseId == caseIds(1)) && ...
    all(representativeKr.Level3CaseId == caseIds(1)), ...
    'Pc, full Kr, and representative Kr case IDs do not match.');

caseId = double(caseIds(1));
caseName = replace(fullKr.Level3CaseName(1), '_', ' ');
windowNames = orderedWindows(fullKr.Window);
assert(numel(windowNames) == 6, ...
    'Expected six windows but found %d.', numel(windowNames));

colors.fullPc = [0.72, 0.75, 0.79];
colors.selectedPc = [0.90, 0.20, 0.12];
colors.gasLight = [0.96, 0.75, 0.50];
colors.gasDark = [0.86, 0.33, 0.07];
colors.waterLight = [0.65, 0.78, 0.91];
colors.waterDark = [0.06, 0.33, 0.62];

outputFiles = struct();
[outputFiles.pcPng, outputFiles.pcPdf] = plotPcFamily( ...
    pc, representativeKr, windowNames, caseId, caseName, outputDir, colors);
[outputFiles.krPng, outputFiles.krPdf] = plotFullKrFamily( ...
    fullKr, representativeKr, windowNames, caseId, caseName, ...
    outputDir, colors);
[outputFiles.scaledKrPng, outputFiles.scaledKrPdf] = ...
    plotScaledRepresentativeKrFamily( ...
    fullKr, representativeKr, windowNames, caseId, caseName, ...
    outputDir, colors);
end


function [pngFile, pdfFile] = plotPcFamily( ...
        pc, representativeKr, windows, caseId, caseName, outputDir, colors)
% Plot all native-endpoint Pc curves and the median-Swi selection.

fig = newCaseFigure();
layout = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', ...
    'Padding', 'compact');
legendHandles = gobjects(2, 1);

for i = 1:numel(windows)
    windowName = windows(i);
    W = pc(pc.Window == windowName, :);
    Rkr = representativeKr(representativeKr.Window == windowName, :);
    selectedSource = unique(Rkr.SourceRow);
    assert(isscalar(selectedSource), ...
        'Expected one representative source row for %s.', windowName);
    sources = unique(W.ReplaySourceRow, 'sorted');
    ax = nexttile(layout);
    hold(ax, 'on');
    if i == 1
        legendHandles(1) = plot(ax, NaN, NaN, '-', ...
            'Color', colors.fullPc, 'LineWidth', 1.0);
        legendHandles(2) = plot(ax, NaN, NaN, '-', ...
            'Color', colors.selectedPc, 'LineWidth', 3.2);
    end

    endpoints = nan(numel(sources), 1);
    selectedFound = false;
    for j = 1:numel(sources)
        C = W(W.ReplaySourceRow == sources(j), :);
        C = sortrows(C, 'GasSaturation');
        valid = isfinite(C.PcBar) & C.PcBar > 0;
        plot(ax, C.GasSaturation(valid), C.PcBar(valid), '-', ...
            'Color', colors.fullPc, 'LineWidth', 0.75);
        endpoints(j) = C.BulkSgMax(1);
        if sources(j) == selectedSource
            plot(ax, C.GasSaturation(valid), C.PcBar(valid), '-', ...
                'Color', colors.selectedPc, 'LineWidth', 3.2);
            selectedFound = true;
        end
    end
    assert(selectedFound, ...
        'Representative source row was not found in Pc curves for %s.', ...
        windowName);

    styleAxis(ax, true);
    endpointLabel = sprintf('S_{g,max}: %.2f-%.2f', ...
        min(endpoints), max(endpoints));
    title(ax, sprintf('%s | %d slices', upper(windowName), numel(sources)), ...
        endpointLabel, 'FontSize', 19, 'FontWeight', 'bold');
    if i > 3
        xlabel(ax, 'Gas saturation', 'FontSize', 19);
    end
    if i == 1 || i == 4
        ylabel(ax, 'Pc [bar]', 'FontSize', 19);
    end
end

title(layout, {sprintf('Case %02d: %s', caseId, caseName), ...
    'Full-87 native-endpoint invasion-percolation Pc curves', ...
    'Red = Pc curve for the median-Swi realization selected for dynamic Kr'}, ...
    'FontName', 'Arial', 'FontSize', 25, 'FontWeight', 'bold');
lgd = legend(legendHandles, {'87 upscaled Pc curves', ...
    'median-Swi selected Pc curve'}, 'Orientation', 'horizontal');
lgd.Layout.Tile = 'south';
lgd.FontName = 'Arial';
lgd.FontSize = 17;

baseName = sprintf('case%02d_upscaled_pc_curves_median_swi_selection', caseId);
[pngFile, pdfFile] = exportCaseFigure(fig, outputDir, baseName);
end


function [pngFile, pdfFile] = plotFullKrFamily( ...
        fullKr, representativeKr, windows, caseId, caseName, ...
        outputDir, colors)
% Plot native-endpoint full-87 Kr curves with the selected curve.

fig = newCaseFigure();
layout = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', ...
    'Padding', 'compact');
legendHandles = gobjects(4, 1);
uGrid = linspace(0, 1, 201)';

for i = 1:numel(windows)
    windowName = windows(i);
    W = fullKr(fullKr.Window == windowName, :);
    R = representativeKr(representativeKr.Window == windowName, :);
    W = sortrows(W, 'SourceRow');
    sources = W.SourceRow;
    ax = nexttile(layout);
    hold(ax, 'on');
    if i == 1
        legendHandles(1) = plot(ax, NaN, NaN, '-', ...
            'Color', colors.gasLight, 'LineWidth', 1.0);
        legendHandles(2) = plot(ax, NaN, NaN, '-', ...
            'Color', colors.waterLight, 'LineWidth', 1.0);
        legendHandles(3) = plot(ax, NaN, NaN, '-', ...
            'Color', colors.gasDark, 'LineWidth', 3.0);
        legendHandles(4) = plot(ax, NaN, NaN, '-', ...
            'Color', colors.waterDark, 'LineWidth', 3.0);
    end

    endpoints = double(W.PcMaxSg);
    for j = 1:numel(sources)
        C = W(j, :);
        krg = uGrid .^ double(C.GasCoreyExponent);
        krw = (1.0 - uGrid) .^ double(C.BrineCoreyExponent);
        x = uGrid .* double(C.PcMaxSg);
        plot(ax, x, krg, '-', 'Color', colors.gasLight, ...
            'LineWidth', 0.65);
        plot(ax, x, krw, '-', 'Color', colors.waterLight, ...
            'LineWidth', 0.65);
    end

    assert(height(R) == 1, ...
        'Expected one representative Kr summary row for %s.', windowName);
    xRep = uGrid .* double(R.PcMaxSg);
    krgRep = uGrid .^ double(R.GasCoreyExponent);
    krwRep = (1.0 - uGrid) .^ double(R.BrineCoreyExponent);
    plot(ax, xRep, krgRep, '-', 'Color', colors.gasDark, ...
        'LineWidth', 3.2);
    plot(ax, xRep, krwRep, '-', 'Color', colors.waterDark, ...
        'LineWidth', 3.2);

    styleAxis(ax, false);
    endpointLabel = sprintf('S_{g,max}: %.2f-%.2f', ...
        min(endpoints), max(endpoints));
    title(ax, sprintf('%s | %d slices', upper(windowName), numel(sources)), ...
        endpointLabel, 'FontSize', 19, 'FontWeight', 'bold');
    if i > 3
        xlabel(ax, 'Gas saturation', 'FontSize', 19);
    end
    if i == 1 || i == 4
        ylabel(ax, 'Relative permeability', 'FontSize', 19);
    end
end

titleLines = {sprintf('Case %02d: %s', caseId, caseName), ...
    'Full-87 native-endpoint dynamic Kr curves', ...
    'Dark curves = median-Swi selected representative'};
baseName = sprintf('case%02d_upscaled_dynamic_kr_curves', caseId);
title(layout, titleLines, 'FontName', 'Arial', 'FontSize', 25, ...
    'FontWeight', 'bold');
lgd = legend(legendHandles, {'87 full Krg curves', '87 full Krw curves', ...
    'selected representative Krg', 'selected representative Krw'}, ...
    'Orientation', 'horizontal');
lgd.Layout.Tile = 'south';
lgd.FontName = 'Arial';
lgd.FontSize = 17;
[pngFile, pdfFile] = exportCaseFigure(fig, outputDir, baseName);
end


function [pngFile, pdfFile] = plotScaledRepresentativeKrFamily( ...
        fullKr, representativeKr, windows, caseId, caseName, ...
        outputDir, colors)
% Map one representative normalized Kr shape to every physical endpoint.

fig = newCaseFigure();
layout = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', ...
    'Padding', 'compact');
legendHandles = gobjects(4, 1);
uGrid = linspace(0, 1, 201)';

for i = 1:numel(windows)
    windowName = windows(i);
    W = sortrows(fullKr(fullKr.Window == windowName, :), 'SourceRow');
    R = representativeKr(representativeKr.Window == windowName, :);
    assert(height(R) == 1, ...
        'Expected one representative Kr summary row for %s.', windowName);

    endpoints = double(W.PcMaxSg);
    physicalSwi = 1.0 - endpoints;
    krgShape = uGrid .^ double(R.GasCoreyExponent);
    krwShape = (1.0 - uGrid) .^ double(R.BrineCoreyExponent);

    ax = nexttile(layout);
    hold(ax, 'on');
    if i == 1
        legendHandles(1) = plot(ax, NaN, NaN, '-', ...
            'Color', colors.gasLight, 'LineWidth', 1.0);
        legendHandles(2) = plot(ax, NaN, NaN, '-', ...
            'Color', colors.waterLight, 'LineWidth', 1.0);
        legendHandles(3) = plot(ax, NaN, NaN, '-', ...
            'Color', colors.gasDark, 'LineWidth', 3.0);
        legendHandles(4) = plot(ax, NaN, NaN, '-', ...
            'Color', colors.waterDark, 'LineWidth', 3.0);
    end

    for j = 1:numel(endpoints)
        physicalSg = uGrid .* endpoints(j);
        plot(ax, physicalSg, krgShape, '-', 'Color', colors.gasLight, ...
            'LineWidth', 0.65);
        plot(ax, physicalSg, krwShape, '-', 'Color', colors.waterLight, ...
            'LineWidth', 0.65);
    end

    representativeSg = uGrid .* double(R.PcMaxSg);
    plot(ax, representativeSg, krgShape, '-', ...
        'Color', colors.gasDark, 'LineWidth', 3.2);
    plot(ax, representativeSg, krwShape, '-', ...
        'Color', colors.waterDark, 'LineWidth', 3.2);

    styleAxis(ax, false);
    title(ax, sprintf('%s | %d slices', upper(windowName), height(W)), ...
        sprintf('physical S_{wi}: %.2f-%.2f', ...
        min(physicalSwi), max(physicalSwi)), ...
        'FontSize', 19, 'FontWeight', 'bold');
    if i > 3
        xlabel(ax, 'Physical gas saturation', 'FontSize', 19);
    end
    if i == 1 || i == 4
        ylabel(ax, 'Relative permeability', 'FontSize', 19);
    end
end

title(layout, {sprintf('Case %02d: %s', caseId, caseName), ...
    'Pc-guided slice-scaled dynamic Kr curves on the physical saturation axis', ...
    'One representative normalized shape per window; each curve ends at S_{g,max}=1-S_{wi}'}, ...
    'FontName', 'Arial', 'FontSize', 25, 'FontWeight', 'bold');
lgd = legend(legendHandles, ...
    {'87 slice-scaled Krg curves', '87 slice-scaled Krw curves', ...
    'selected representative Krg', 'selected representative Krw'}, ...
    'Orientation', 'horizontal');
lgd.Layout.Tile = 'south';
lgd.FontName = 'Arial';
lgd.FontSize = 17;
baseName = sprintf('case%02d_pc_guided_slice_scaled_dynamic_kr_curves', ...
    caseId);
[pngFile, pdfFile] = exportCaseFigure(fig, outputDir, baseName);
end


function fig = newCaseFigure()
% Create the common publication-size figure canvas.

fig = figure('Color', 'w', 'Position', [60, 35, 2100, 1420]);
end


function styleAxis(ax, logarithmicY)
% Apply common panel formatting.

xlim(ax, [0, 1]);
xticks(ax, 0:0.2:1);
if logarithmicY
    set(ax, 'YScale', 'log');
    ylim(ax, [1e-2, 1e3]);
    yticks(ax, 10 .^ (-2:3));
else
    ylim(ax, [0, 1]);
    yticks(ax, 0:0.2:1);
end
grid(ax, 'on');
ax.GridColor = [0.82, 0.82, 0.82];
ax.GridAlpha = 0.55;
ax.MinorGridAlpha = 0.20;
ax.FontName = 'Arial';
ax.FontSize = 16;
ax.LineWidth = 1.0;
box(ax, 'on');
axis(ax, 'square');
end


function windows = orderedWindows(windowColumn)
% Return window labels sorted by their numeric suffix.

windows = unique(windowColumn, 'stable');
numbers = arrayfun(@windowNumber, windows);
[~, order] = sort(numbers);
windows = windows(order);
end


function n = windowNumber(windowName)
% Extract the numeric suffix from a window label.

token = regexp(char(windowName), '(\d+)$', 'tokens', 'once');
if isempty(token)
    n = Inf;
else
    n = str2double(token{1});
end
end


function requireColumns(T, required)
% Assert that a table contains every required column.

missing = setdiff(required, T.Properties.VariableNames);
assert(isempty(missing), 'Input table lacks columns: %s', ...
    strjoin(missing, ', '));
end


function [pngFile, pdfFile] = exportCaseFigure(fig, outputDir, baseName)
% Export one figure in raster and vector formats.

pngFile = fullfile(outputDir, baseName + ".png");
pdfFile = fullfile(outputDir, baseName + ".pdf");
exportgraphics(fig, pngFile, 'Resolution', 300, ...
    'BackgroundColor', 'white');
exportgraphics(fig, pdfFile, 'ContentType', 'vector', ...
    'BackgroundColor', 'white');
close(fig);
fprintf('Saved figure: %s\n', pngFile);
end
