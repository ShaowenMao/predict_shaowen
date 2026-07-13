function outputFiles = plot_pc_guided_scaled_kr_curves(sliceCurveCsv, outputDir)
%PLOT_PC_GUIDED_SCALED_KR_CURVES Plot Pc-endpoint-scaled dynamic Kr curves.
%
%   OUTPUTFILES = PLOT_PC_GUIDED_SCALED_KR_CURVES(SLICECURVECSV, OUTPUTDIR)
%   reads the slice-level curves exported by
%   run_kr_upscaling_dyn_median_examples_full87 in median_swi mode. For
%   each throw window, it plots all slice curves obtained by mapping one
%   representative normalized dynamic-Kr shape to each slice's Pc-derived
%   BulkSgMax = 1 - EffectiveSwi. The median-Swi slice used for the dynamic
%   Kr calculation is highlighted.
%
%   The function writes publication-ready PNG and PDF figures and returns
%   their paths in OUTPUTFILES. It does not alter any upscaling results.

arguments
    sliceCurveCsv (1, 1) string
    outputDir (1, 1) string
end

if ~isfile(sliceCurveCsv)
    error('ScaledKrPlot:MissingInput', ...
        'Slice-curve CSV does not exist: %s', sliceCurveCsv);
end
if ~isfolder(outputDir)
    mkdir(outputDir);
end

T = readtable(sliceCurveCsv, 'TextType', 'string');
required = {'GeologyId', 'Level3CaseId', 'Level3CaseName', 'Window', ...
    'SliceIndex', 'PcReplaySourceRow', 'RepresentativeKrReplaySourceRow', ...
    'BulkSgMax', 'EffectiveSwi', 'NormalizedSg', 'GasSaturation', ...
    'Krg', 'Krw'};
missing = setdiff(required, T.Properties.VariableNames);
if ~isempty(missing)
    error('ScaledKrPlot:MissingColumns', ...
        'Slice-curve CSV is missing columns: %s', strjoin(missing, ', '));
end

geologyIds = unique(T.GeologyId);
caseIds = unique(T.Level3CaseId);
if numel(geologyIds) ~= 1 || numel(caseIds) ~= 1
    error('ScaledKrPlot:MixedCases', ...
        'Input must contain exactly one geology and one Level 3 case.');
end

windowNames = unique(T.Window, 'stable');
windowNumbers = arrayfun(@windowNumber, windowNames);
[~, order] = sort(windowNumbers);
windowNames = windowNames(order);
if numel(windowNames) ~= 6
    warning('ScaledKrPlot:WindowCount', ...
        'Expected six windows but found %d.', numel(windowNames));
end

caseId = double(caseIds(1));
caseName = replace(T.Level3CaseName(1), '_', ' ');
geologyId = T.GeologyId(1);

gasLight = [0.95, 0.73, 0.48];
gasDark = [0.86, 0.34, 0.08];
waterLight = [0.62, 0.76, 0.90];
waterDark = [0.08, 0.34, 0.62];
endpointColor = [0.34, 0.34, 0.34];

fig = figure('Color', 'w', 'Position', [60, 40, 2100, 1420]);
layout = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', ...
    'Padding', 'compact');

legendHandles = gobjects(4, 1);
for i = 1:numel(windowNames)
    windowName = windowNames(i);
    W = T(T.Window == windowName, :);
    sliceIds = unique(W.SliceIndex, 'sorted');
    ax = nexttile(layout);
    hold(ax, 'on');

    if i == 1
        legendHandles(1) = plot(ax, NaN, NaN, '-', 'Color', gasLight, ...
            'LineWidth', 1.0);
        legendHandles(2) = plot(ax, NaN, NaN, '-', 'Color', waterLight, ...
            'LineWidth', 1.0);
        legendHandles(3) = plot(ax, NaN, NaN, '-', 'Color', gasDark, ...
            'LineWidth', 3.0);
        legendHandles(4) = plot(ax, NaN, NaN, '-', 'Color', waterDark, ...
            'LineWidth', 3.0);
    end

    endpointSg = zeros(numel(sliceIds), 1);
    for j = 1:numel(sliceIds)
        C = W(W.SliceIndex == sliceIds(j), :);
        C = sortrows(C, 'NormalizedSg');
        plot(ax, C.GasSaturation, C.Krg, '-', 'Color', gasLight, ...
            'LineWidth', 0.65);
        plot(ax, C.GasSaturation, C.Krw, '-', 'Color', waterLight, ...
            'LineWidth', 0.65);
        endpointSg(j) = C.BulkSgMax(1);
    end

    representativeSource = unique(W.RepresentativeKrReplaySourceRow);
    representativeMask = W.PcReplaySourceRow == representativeSource(1);
    if ~any(representativeMask)
        medianEndpoint = median(endpointSg);
        [~, nearestId] = min(abs(endpointSg - medianEndpoint));
        representativeMask = W.SliceIndex == sliceIds(nearestId);
        warning('ScaledKrPlot:RepresentativeFallback', ...
            '%s used the slice nearest median BulkSgMax.', windowName);
    end
    R = sortrows(W(representativeMask, :), 'NormalizedSg');
    plot(ax, R.GasSaturation, R.Krg, '-', 'Color', gasDark, ...
        'LineWidth', 3.2);
    plot(ax, R.GasSaturation, R.Krw, '-', 'Color', waterDark, ...
        'LineWidth', 3.2);
    scatter(ax, endpointSg, ones(size(endpointSg)), 12, endpointColor, ...
        'filled', 'MarkerFaceAlpha', 0.32, 'MarkerEdgeAlpha', 0.32);

    xlim(ax, [0, 1]);
    ylim(ax, [0, 1]);
    xticks(ax, 0:0.2:1);
    yticks(ax, 0:0.2:1);
    grid(ax, 'on');
    ax.GridColor = [0.82, 0.82, 0.82];
    ax.GridAlpha = 0.55;
    ax.MinorGridAlpha = 0.18;
    ax.FontName = 'Arial';
    ax.FontSize = 17;
    ax.LineWidth = 1.0;
    box(ax, 'on');
    axis(ax, 'square');

    if i > 3
        xlabel(ax, 'Gas saturation', 'FontSize', 20);
    end
    if i == 1 || i == 4
        ylabel(ax, 'Relative permeability', 'FontSize', 20);
    end
    title(ax, sprintf('%s | %d slices', upper(windowName), numel(sliceIds)), ...
        sprintf('S_{g,max}: %.2f-%.2f (median %.2f)', ...
        min(endpointSg), max(endpointSg), median(endpointSg)), ...
        'FontSize', 20, 'FontWeight', 'bold');
end

title(layout, {sprintf('Case %02d: %s', caseId, caseName), ...
    'Pc-guided slice-scaled dynamic Kr curves by window', ...
    'One normalized shape per window; each slice endpoint uses BulkSgMax = 1 - EffectiveSwi from Pc upscaling'}, ...
    'FontName', 'Arial', 'FontSize', 27, 'FontWeight', 'bold');

lgd = legend(legendHandles, ...
    {'87 slice-scaled Krg', '87 slice-scaled Krw', ...
    'selected representative Krg', 'selected representative Krw'}, ...
    'Orientation', 'horizontal');
lgd.Layout.Tile = 'south';
lgd.FontName = 'Arial';
lgd.FontSize = 18;

baseName = sprintf('case%02d_pc_guided_slice_scaled_dynamic_kr_by_window', ...
    caseId);
pngFile = fullfile(outputDir, baseName + ".png");
pdfFile = fullfile(outputDir, baseName + ".pdf");
exportgraphics(fig, pngFile, 'Resolution', 300);
exportgraphics(fig, pdfFile, 'ContentType', 'vector');
close(fig);

outputFiles = struct('png', pngFile, 'pdf', pdfFile, ...
    'geologyId', geologyId, 'caseId', caseId);
fprintf('Saved slice-scaled Kr PNG: %s\n', pngFile);
fprintf('Saved slice-scaled Kr PDF: %s\n', pdfFile);
end


function n = windowNumber(windowName)
% Return the numeric suffix of a window label such as famp6.

token = regexp(char(windowName), '(\d+)$', 'tokens', 'once');
if isempty(token)
    n = Inf;
else
    n = str2double(token{1});
end
end
