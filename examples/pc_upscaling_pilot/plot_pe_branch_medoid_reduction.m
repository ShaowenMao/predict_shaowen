function outputFiles = plot_pe_branch_medoid_reduction( ...
        fullSliceInput, reducedInput, outputDir)
%PLOT_PE_BRANCH_MEDOID_REDUCTION Review the Pe-branch Pc simplification.
%
%   OUTPUTFILES = PLOT_PE_BRANCH_MEDOID_REDUCTION(FULLSLICEINPUT,
%   REDUCEDINPUT, OUTPUTDIR) creates:
%     1. a six-panel comparison of all native Pc curves and the selected
%        within-branch full-curve medoids; and
%     2. a window-by-slice map of the resulting Pe-branch assignments.
%
%   FULLSLICEINPUT and REDUCEDINPUT can be reservoirReady structures or MAT
%   paths. The routine is diagnostic only and does not alter either input.

fullSlice = loadReservoirReady(fullSliceInput);
reduced = loadReservoirReady(reducedInput);
assert(isfield(reduced, 'peBranchModel') && ...
    string(reduced.pcRepresentation) == "pe_branch_medoid", ...
    'PeBranchPlot:NotReduced', ...
    'The reduced input is not a Pe-branch-medoid artifact.');
assert(string(fullSlice.geologyId) == string(reduced.geologyId) && ...
    double(fullSlice.level3CaseId) == double(reduced.level3CaseId), ...
    'PeBranchPlot:CaseMismatch', ...
    'Full and reduced inputs describe different geology/case combinations.');
ensureFolder(outputDir);

windows = string(fullSlice.windowLabels(:));
slices = double(fullSlice.sliceIndices(:));
labels = double(reduced.peBranchModel.branchLabels);
branchSummary = reduced.peBranchModel.branchSummary;
palette = [0.10, 0.42, 0.72; 0.90, 0.38, 0.08; 0.16, 0.62, 0.34];
lightPalette = 0.72 + 0.28 .* palette;
panelLabels = compose('(%c)', 'a' + (0:numel(windows) - 1));
panelAxes = gobjects(numel(windows), 1);

fig = figure('Color', 'w', 'Position', [50, 50, 1750, 940]);
t = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', ...
    'Padding', 'compact');

for w = 1:numel(windows)
    ax = nexttile(t, w);
    panelAxes(w) = ax;
    hold(ax, 'on');
    nBranches = max(labels(w, :));
    branchSizes = zeros(1, nBranches);
    for b = 1:nBranches
        branchColumns = find(labels(w, :) == b);
        branchSizes(b) = numel(branchColumns);
        color = palette(min(b, size(palette, 1)), :);
        lightColor = lightPalette(min(b, size(lightPalette, 1)), :);
        for s = branchColumns
            P = fullSlice.pcCurves{w, s};
            semilogy(ax, double(P.gasSaturation), ...
                max(double(P.pcBar), 1.0e-12), '-', ...
                'Color', lightColor, 'LineWidth', 0.75);
        end

        summaryRow = branchSummary( ...
            branchSummary.Window == windows(w) & ...
            branchSummary.PeBranchId == b, :);
        assert(height(summaryRow) == 1, ...
            'PeBranchPlot:MissingBranchSummary', ...
            'Missing branch summary for %s branch %d.', windows(w), b);
        medoidSlice = double(summaryRow.MedoidSliceIndex);
        [found, medoidColumn] = ismember(medoidSlice, slices);
        assert(found, 'PeBranchPlot:MissingMedoidSlice', ...
            'Medoid slice %d is missing.', medoidSlice);
        P = fullSlice.pcCurves{w, medoidColumn};
        semilogy(ax, double(P.gasSaturation), ...
            max(double(P.pcBar), 1.0e-12), '-', ...
            'Color', color, 'LineWidth', 3.2, ...
            'DisplayName', sprintf('Pe branch %d medoid', b));
        semilogy(ax, double(P.gasSaturation(end)), ...
            max(double(P.pcBar(end)), 1.0e-12), 'o', ...
            'MarkerSize', 6.5, 'MarkerFaceColor', color, ...
            'MarkerEdgeColor', 'w', 'HandleVisibility', 'off');
    end

    title(ax, sprintf('%s | %d branch%s | n = %s', ...
        upper(char(windows(w))), nBranches, pluralSuffix(nBranches), ...
        strjoin(string(branchSizes), ' + ')), ...
        'FontSize', 17, 'FontWeight', 'bold');
    text(ax, 0.98, 0.96, panelLabels(w), 'Units', 'normalized', ...
        'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
        'FontSize', 15, 'FontWeight', 'bold');
    xlim(ax, [0, 1]);
    ylim(ax, [1.0e-2, 1.0e3]);
    grid(ax, 'on');
    ax.XMinorGrid = 'on';
    ax.YMinorGrid = 'on';
    ax.FontSize = 14;
    ax.LineWidth = 1.0;
    xlabel(ax, 'Gas saturation', 'FontSize', 15);
    ylabel(ax, 'Pc [bar]', 'FontSize', 15);
end

title(t, {sprintf('Case %02d: %s', double(fullSlice.level3CaseId), ...
    strrep(char(fullSlice.level3CaseName), '_', ' ')), ...
    'Entry-pressure branches and within-branch full-Pc-curve medoids'}, ...
    'FontSize', 25, 'FontWeight', 'bold');
xlabel(t, 'Pe branch membership uses log10 entry pressure; medoids use the complete Pc curve', ...
    'FontSize', 16);

medoidHandles = gobjects(max(labels, [], 'all'), 1);
medoidNames = strings(numel(medoidHandles), 1);
for b = 1:numel(medoidHandles)
    medoidHandles(b) = plot(panelAxes(end), nan, nan, '-', ...
        'Color', palette(min(b, size(palette, 1)), :), 'LineWidth', 3.2);
    medoidNames(b) = sprintf('Pe branch %d medoid', b);
end
set(panelAxes, 'YScale', 'log', 'YLim', [1.0e-2, 1.0e3]);
legend(medoidHandles, cellstr(medoidNames), 'Orientation', 'horizontal', ...
    'Location', 'southoutside', 'FontSize', 14, 'Box', 'off');

token = sprintf('case%02d_pe_branch_pc_medoids_by_window', ...
    double(fullSlice.level3CaseId));
pcPng = fullfile(outputDir, [token, '.png']);
pcPdf = fullfile(outputDir, [token, '.pdf']);
exportgraphics(fig, pcPng, 'Resolution', 240);
exportgraphics(fig, pcPdf, 'ContentType', 'vector');
close(fig);

fig = figure('Color', 'w', 'Position', [100, 100, 1800, 500]);
ax = axes(fig);
imagesc(ax, slices, 1:numel(windows), labels);
set(ax, 'YDir', 'normal', 'YTick', 1:numel(windows), ...
    'YTickLabel', upper(windows), 'FontSize', 16, 'LineWidth', 1.0);
maxBranch = max(labels, [], 'all');
colormap(ax, palette(1:maxBranch, :));
clim(ax, [0.5, maxBranch + 0.5]);
cb = colorbar(ax);
cb.Ticks = 1:maxBranch;
cb.TickLabels = compose('%d', cb.Ticks);
cb.Label.String = 'Pe branch (low to high entry pressure)';
cb.Label.FontSize = 15;
cb.FontSize = 14;
xlabel(ax, 'Along-strike slice', 'FontSize', 18);
ylabel(ax, 'Throw window', 'FontSize', 18);
title(ax, sprintf(['Case %02d: Pe-branch assignments along strike ', ...
    '(branch 1 = lowest Pe)'], double(fullSlice.level3CaseId)), ...
    'FontSize', 23, 'FontWeight', 'bold');
grid(ax, 'off');

token = sprintf('case%02d_pe_branch_assignments_along_strike', ...
    double(fullSlice.level3CaseId));
mapPng = fullfile(outputDir, [token, '.png']);
mapPdf = fullfile(outputDir, [token, '.pdf']);
exportgraphics(fig, mapPng, 'Resolution', 240);
exportgraphics(fig, mapPdf, 'ContentType', 'vector');
close(fig);

outputFiles = struct('pcPng', string(pcPng), 'pcPdf', string(pcPdf), ...
    'assignmentPng', string(mapPng), 'assignmentPdf', string(mapPdf));
fprintf('Saved Pe-branch Pc review figure: %s\n', pcPng);
fprintf('Saved Pe-branch assignment map: %s\n', mapPng);
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
        'PeBranchPlot:MissingInput', 'MAT input not found: %s', file);
    loaded = load(file, 'reservoirReady');
    assert(isfield(loaded, 'reservoirReady'), ...
        'PeBranchPlot:MissingVariable', ...
        'MAT input lacks reservoirReady: %s', file);
    S = loaded.reservoirReady;
end
end


function suffix = pluralSuffix(count)
% Return a compact plural suffix for panel titles.

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
