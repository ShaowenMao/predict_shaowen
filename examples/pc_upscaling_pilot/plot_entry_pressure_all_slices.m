function outputFiles = plot_entry_pressure_all_slices( ...
        fullSliceInputs, reducedInputs, outputDir)
%PLOT_ENTRY_PRESSURE_ALL_SLICES Visualize Pe for every window and slice.
%
%   OUTPUTFILES = PLOT_ENTRY_PRESSURE_ALL_SLICES(FULLSLICEINPUTS,
%   REDUCEDINPUTS, OUTPUTDIR) accepts matching arrays of rigorous full-slice
%   and Pe-branch-medoid reservoir-ready MAT files or structures. It writes:
%     1. one shared-scale heatmap comparing all requested cases; and
%     2. one six-panel all-slice Pe profile figure per case.
%
%   Entry pressure is read directly from each native Pc curve as its first
%   positive Pc value. Profile colors show the automatically detected Pe
%   branches, and thick horizontal segments show each branch medoid's Pe.
%   The function is diagnostic only and does not modify reservoir inputs.

fullSliceInputs = normalizeInputs(fullSliceInputs);
reducedInputs = normalizeInputs(reducedInputs);
assert(numel(fullSliceInputs) == numel(reducedInputs) && ...
    ~isempty(fullSliceInputs), 'PePlot:InputCount', ...
    'Full-slice and reduced input arrays must be nonempty and the same size.');
ensureFolder(outputDir);

nCases = numel(fullSliceInputs);
cases = repmat(struct(), nCases, 1);
allLogPe = zeros(0, 1);

for c = 1:nCases
    fullSlice = loadReservoirReady(fullSliceInputs{c});
    reduced = loadReservoirReady(reducedInputs{c});
    assert(string(fullSlice.geologyId) == string(reduced.geologyId) && ...
        double(fullSlice.level3CaseId) == double(reduced.level3CaseId), ...
        'PePlot:CaseMismatch', ...
        'Full and reduced inputs at position %d describe different cases.', c);
    assert(isfield(reduced, 'peBranchModel'), 'PePlot:MissingBranchModel', ...
        'Reduced input at position %d lacks peBranchModel.', c);

    windows = string(fullSlice.windowLabels(:));
    slices = double(fullSlice.sliceIndices(:));
    peBar = extractEntryPressure(fullSlice.pcCurves);
    labels = double(reduced.peBranchModel.branchLabels);
    assert(isequal(size(peBar), size(labels)), 'PePlot:CoverageMismatch', ...
        'Pe and branch-label coverage differ for case %02d.', ...
        double(fullSlice.level3CaseId));

    cases(c).fullSlice = fullSlice;
    cases(c).reduced = reduced;
    cases(c).windows = windows;
    cases(c).slices = slices;
    cases(c).peBar = peBar;
    cases(c).labels = labels;
    allLogPe = [allLogPe; log10(peBar(:))]; %#ok<AGROW>
end

colorLimits = [floor(min(allLogPe) * 2) / 2, ...
    ceil(max(allLogPe) * 2) / 2];
if colorLimits(1) == colorLimits(2)
    colorLimits = colorLimits + [-0.5, 0.5];
end

outputFiles = struct();
[outputFiles.heatmapPng, outputFiles.heatmapPdf] = ...
    makeSharedHeatmap(cases, colorLimits, outputDir);
outputFiles.profilePng = strings(nCases, 1);
outputFiles.profilePdf = strings(nCases, 1);
for c = 1:nCases
    [outputFiles.profilePng(c), outputFiles.profilePdf(c)] = ...
        makeCaseProfiles(cases(c), outputDir);
end
end


function [pngFile, pdfFile] = makeSharedHeatmap(cases, colorLimits, outputDir)
% Plot all requested cases with one physical entry-pressure color scale.

nCases = numel(cases);
nColumns = min(2, nCases);
nRows = ceil(nCases / nColumns);
fig = figure('Color', 'w', 'Position', [40, 60, 1780, 420 * nRows]);
t = tiledlayout(fig, nRows, nColumns, 'TileSpacing', 'compact', ...
    'Padding', 'compact');
panelLabels = compose('(%c)', 'a' + (0:nCases - 1));

for c = 1:nCases
    ax = nexttile(t, c);
    imagesc(ax, cases(c).slices, 1:numel(cases(c).windows), ...
        log10(cases(c).peBar));
    set(ax, 'YDir', 'normal', 'YTick', 1:numel(cases(c).windows), ...
        'YTickLabel', upper(cases(c).windows), 'FontSize', 15, ...
        'LineWidth', 1.0);
    clim(ax, colorLimits);
    xlabel(ax, 'Along-strike slice', 'FontSize', 16);
    ylabel(ax, 'Throw window', 'FontSize', 16);
    title(ax, sprintf('Case %02d: %s', ...
        double(cases(c).fullSlice.level3CaseId), ...
        strrep(char(cases(c).fullSlice.level3CaseName), '_', ' ')), ...
        'FontSize', 19, 'FontWeight', 'bold');
    text(ax, 0.99, 0.96, panelLabels(c), 'Units', 'normalized', ...
        'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
        'Color', 'w', 'FontSize', 16, 'FontWeight', 'bold');
end

colormap(fig, parula(256));
cb = colorbar;
cb.Layout.Tile = 'east';
cb.Ticks = linspace(colorLimits(1), colorLimits(2), 4);
cb.TickLabels = compose('%.3g', 10 .^ cb.Ticks);
cb.FontSize = 15;
cb.Label.String = 'Entry pressure, Pe [bar]';
cb.Label.FontSize = 17;
title(t, 'Upscaled entry pressure for every throw window and along-strike slice', ...
    'FontSize', 26, 'FontWeight', 'bold');

ids = arrayfun(@(x) double(x.fullSlice.level3CaseId), cases);
token = sprintf('%02d_', ids);
token = token(1:end-1);
baseName = sprintf('entry_pressure_all_slices_heatmap_cases_%s', token);
pngFile = string(fullfile(outputDir, [baseName, '.png']));
pdfFile = string(fullfile(outputDir, [baseName, '.pdf']));
exportgraphics(fig, pngFile, 'Resolution', 240);
exportgraphics(fig, pdfFile, 'ContentType', 'vector');
close(fig);
fprintf('Saved shared all-slice Pe heatmap: %s\n', pngFile);
end


function [pngFile, pdfFile] = makeCaseProfiles(caseData, outputDir)
% Plot all slice entry pressures and branch-medoid levels for one case.

palette = [0.10, 0.42, 0.72; 0.90, 0.38, 0.08; 0.16, 0.62, 0.34];
windows = caseData.windows;
slices = caseData.slices;
labels = caseData.labels;
branchSummary = caseData.reduced.peBranchModel.branchSummary;
maxBranch = max(labels, [], 'all');
panelLabels = compose('(%c)', 'a' + (0:numel(windows) - 1));

fig = figure('Color', 'w', 'Position', [40, 50, 1760, 930]);
t = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', ...
    'Padding', 'compact');

for w = 1:numel(windows)
    ax = nexttile(t, w);
    hold(ax, 'on');
    nBranches = max(labels(w, :));
    branchSizes = zeros(1, nBranches);
    for b = 1:nBranches
        mask = labels(w, :) == b;
        branchSizes(b) = sum(mask);
        color = palette(min(b, size(palette, 1)), :);
        scatter(ax, slices(mask), caseData.peBar(w, mask), 23, ...
            'MarkerFaceColor', color, 'MarkerEdgeColor', 'w', ...
            'LineWidth', 0.35, 'MarkerFaceAlpha', 0.82);
        summaryRow = branchSummary( ...
            branchSummary.Window == windows(w) & ...
            branchSummary.PeBranchId == b, :);
        assert(height(summaryRow) == 1, 'PePlot:MissingBranchSummary', ...
            'Missing branch summary for %s branch %d.', windows(w), b);
        medoidPe = double(summaryRow.MedoidEntryPcBar);
        xRange = [min(slices(mask)), max(slices(mask))];
        if xRange(1) == xRange(2)
            xRange = xRange + [-0.6, 0.6];
        end
        plot(ax, xRange, [medoidPe, medoidPe], '-', ...
            'Color', color, 'LineWidth', 3.0);
    end
    set(ax, 'YScale', 'log', 'YLim', [1.0e-2, 1.0e2], ...
        'XLim', [min(slices) - 1, max(slices) + 1], ...
        'FontSize', 14, 'LineWidth', 1.0);
    grid(ax, 'on');
    ax.XMinorGrid = 'on';
    ax.YMinorGrid = 'on';
    xlabel(ax, 'Along-strike slice', 'FontSize', 15);
    ylabel(ax, 'Entry pressure, Pe [bar]', 'FontSize', 15);
    title(ax, sprintf('%s | %d branch%s | n = %s', ...
        upper(char(windows(w))), nBranches, pluralSuffix(nBranches), ...
        strjoin(string(branchSizes), ' + ')), ...
        'FontSize', 17, 'FontWeight', 'bold');
    text(ax, 0.98, 0.96, panelLabels(w), 'Units', 'normalized', ...
        'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
        'FontSize', 15, 'FontWeight', 'bold');
end

title(t, {sprintf('Case %02d: %s', ...
    double(caseData.fullSlice.level3CaseId), ...
    strrep(char(caseData.fullSlice.level3CaseName), '_', ' ')), ...
    'Entry pressure for all 87 slices by throw window'}, ...
    'FontSize', 25, 'FontWeight', 'bold');

legendHandles = gobjects(maxBranch, 1);
legendLabels = strings(maxBranch, 1);
lastAxis = gca;
for b = 1:maxBranch
    legendHandles(b) = plot(lastAxis, nan, nan, 'o-', ...
        'Color', palette(b, :), 'MarkerFaceColor', palette(b, :), ...
        'LineWidth', 2.5, 'MarkerSize', 6);
    legendLabels(b) = sprintf('Branch %d medoid', b);
end
legend(legendHandles, cellstr(legendLabels), ...
    'Orientation', 'horizontal', 'Location', 'southoutside', ...
    'FontSize', 14, 'Box', 'off');

caseId = double(caseData.fullSlice.level3CaseId);
baseName = sprintf('case%02d_entry_pressure_all_slices_by_window', caseId);
pngFile = string(fullfile(outputDir, [baseName, '.png']));
pdfFile = string(fullfile(outputDir, [baseName, '.pdf']));
exportgraphics(fig, pngFile, 'Resolution', 240);
exportgraphics(fig, pdfFile, 'ContentType', 'vector');
close(fig);
fprintf('Saved Case %02d all-slice Pe profiles: %s\n', caseId, pngFile);
end


function peBar = extractEntryPressure(pcCurves)
% Read the first positive pressure from every native Pc curve.

peBar = nan(size(pcCurves));
for i = 1:numel(pcCurves)
    P = pcCurves{i};
    sg = double(P.gasSaturation(:));
    pc = double(P.pcBar(:));
    [~, order] = sort(sg);
    pc = pc(order);
    firstPositive = find(pc > 0 & isfinite(pc), 1, 'first');
    assert(~isempty(firstPositive), 'PePlot:MissingEntryPressure', ...
        'A native Pc curve lacks a positive entry pressure.');
    peBar(i) = pc(firstPositive);
end
end


function inputs = normalizeInputs(value)
% Convert supported scalar or array inputs into a cell vector.

if iscell(value)
    inputs = value(:);
elseif isstring(value) || ischar(value)
    inputs = cellstr(string(value(:)));
elseif isstruct(value)
    inputs = num2cell(value(:));
else
    error('PePlot:UnsupportedInput', ...
        'Inputs must be MAT paths, structures, or cell arrays.');
end
end


function S = loadReservoirReady(value)
% Load one reservoirReady structure from memory or MAT storage.

if isstruct(value)
    if isfield(value, 'reservoirReady')
        S = value.reservoirReady;
    else
        S = value;
    end
else
    file = string(value);
    assert(isscalar(file) && isfile(file), ...
        'PePlot:MissingInput', 'MAT input not found: %s', file);
    loaded = load(file, 'reservoirReady');
    assert(isfield(loaded, 'reservoirReady'), ...
        'PePlot:MissingVariable', ...
        'MAT input lacks reservoirReady: %s', file);
    S = loaded.reservoirReady;
end
end


function suffix = pluralSuffix(count)
% Return the suffix needed by the word branch.

if count == 1
    suffix = '';
else
    suffix = 'es';
end
end


function ensureFolder(folderPath)
% Create a folder when needed.

if ~isfolder(folderPath)
    mkdir(folderPath);
end
end
