function outputFiles = plot_pe_branch_medoid_entry_pressure_spatial_maps( ...
        reducedInputs, outputDir)
%PLOT_PE_BRANCH_MEDOID_ENTRY_PRESSURE_SPATIAL_MAPS Plot reduced Pe fields.
%
%   OUTPUTFILES = PLOT_PE_BRANCH_MEDOID_ENTRY_PRESSURE_SPATIAL_MAPS(
%   REDUCEDINPUTS, OUTPUTDIR) reads Pe-branch-medoid reservoir-ready MAT
%   files and writes a stacked spatial map in the same format used for the
%   rigorous full-slice entry-pressure comparison. Each map cell contains
%   the entry pressure of the complete Pc-curve medoid assigned to that
%   slice's Pe branch, rather than the slice-specific entry pressure.
%
%   REDUCEDINPUTS may be a MAT-file path, a string array of paths, a
%   reservoirReady structure, or a cell array containing those values.

reducedInputs = normalizeInputs(reducedInputs);
assert(~isempty(reducedInputs), 'PeBranchMap:EmptyInput', ...
    'At least one Pe-branch-medoid reservoir-ready input is required.');
ensureFolder(outputDir);

nCases = numel(reducedInputs);
caseIds = nan(nCases, 1);
caseTitles = strings(nCases, 1);
maps = cell(nCases, 1);
highEntryFraction = nan(nCases, 1);
meanWithinWindowSd = nan(nCases, 1);
representativeLevelCount = nan(nCases, 1);
windowLabels = compose("W%d", 1:6);

for c = 1:nCases
    reservoirReady = loadReservoirReady(reducedInputs{c});
    assert(isfield(reservoirReady, 'peBranchModel') && ...
        isfield(reservoirReady.peBranchModel, 'sliceAssignments'), ...
        'PeBranchMap:MissingBranchModel', ...
        'Input %d does not contain Pe-branch slice assignments.', c);

    caseIds(c) = double(reservoirReady.level3CaseId);
    caseTitles(c) = displayCaseTitle(reservoirReady);
    assert(isfield(reservoirReady, 'pcCurves') && ...
        isequal(size(reservoirReady.pcCurves), [6, 87]), ...
        'PeBranchMap:UnexpectedCoverage', ...
        'Input %d must contain a 6-by-87 pcCurves cell array.', c);
    peMapBar = extractConnectedEntryPressure(reservoirReady.pcCurves);
    logPeMap = log10(peMapBar);

    maps{c} = logPeMap;
    highEntryFraction(c) = mean(peMapBar(:) >= 1.0);
    meanWithinWindowSd(c) = mean(std(logPeMap, 0, 2));
    representativeLevelCount(c) = numel(unique(peMapBar(:)));
end

% Match the rigorous full-slice figure so the two reductions can be
% compared directly without a color-scale change.
colorLimits = log10([0.03, 15.0]);
pressureTicks = [0.03, 0.1, 0.3, 1, 3, 10];
cmap = entryPressureMap(256);

figHeight = max(720, 290 * nCases + 160);
fig = figure('Color', 'w', 'Position', [50, 40, 2300, figHeight]);
layout = tiledlayout(fig, nCases, 1, 'TileSpacing', 'compact', ...
    'Padding', 'compact');

for c = 1:nCases
    ax = nexttile(layout);
    imagesc(ax, 1:87, 1:6, maps{c});
    set(ax, 'YDir', 'normal');
    axis(ax, 'image');
    clim(ax, colorLimits);
    colormap(ax, cmap);
    formatAxes(ax, windowLabels);

    title(ax, sprintf([ ...
        '%s | P_e >= 1 bar: %.1f%% | mean within-window SD = ', ...
        '%.2f log units'], ...
        caseTitles(c), 100 * highEntryFraction(c), ...
        meanWithinWindowSd(c)), ...
        'FontSize', 19, 'FontWeight', 'bold', 'Interpreter', 'tex');
    ylabel(ax, 'Window', 'FontSize', 17);
    if c < nCases
        ax.XTickLabel = [];
    else
        xlabel(ax, 'Along-strike slice', 'FontSize', 18);
    end
end

cb = colorbar(ax, 'eastoutside');
cb.Layout.Tile = 'east';
cb.Ticks = log10(pressureTicks);
cb.TickLabels = compose('%g', pressureTicks);
cb.Label.String = 'Upscaled entry pressure, P_e [bar]';
cb.Label.FontSize = 19;
cb.FontSize = 16;

title(layout, 'Spatial distribution of upscaled entry capillary pressure', ...
    'FontSize', 29, 'FontWeight', 'bold');
subtitle(layout, ['Pe-branch-medoid option: six throw windows by 87 ', ...
    'along-strike slices; W1 is at ', ...
    'the bottom. Each slice uses the complete Pc-curve medoid from its ', ...
    'assigned Pe branch; all cases share one logarithmic pressure scale.'], ...
    'FontSize', 17);

idToken = strjoin(compose('%02d', caseIds), '_');
base = fullfile(outputDir, sprintf( ...
    'pe_branch_medoid_entry_pressure_spatial_maps_cases_%s', idToken));
outputFiles = struct();
outputFiles.png = string(base + ".png");
outputFiles.pdf = string(base + ".pdf");
outputFiles.summaryCsv = string(fullfile(outputDir, sprintf( ...
    'pe_branch_medoid_entry_pressure_summary_cases_%s.csv', idToken)));
exportgraphics(fig, outputFiles.png, 'Resolution', 300);
exportgraphics(fig, outputFiles.pdf, 'ContentType', 'vector');
close(fig);

summary = table(caseIds, caseTitles, highEntryFraction, ...
    meanWithinWindowSd, representativeLevelCount, ...
    'VariableNames', {'CaseId', 'CaseTitle', 'FractionPeAtLeast1Bar', ...
    'MeanWithinWindowSdLog10Pe', 'DistinctBranchMedoidPeLevels'});
writetable(summary, outputFiles.summaryCsv);

fprintf('Saved Pe-branch-medoid spatial map: %s\n', outputFiles.png);
end


function peMapBar = extractConnectedEntryPressure(pcCurves)
% Read the connected-path invasion pressure after the pre-entry anchor.

peMapBar = nan(size(pcCurves));
for i = 1:numel(pcCurves)
    curve = pcCurves{i};
    sg = double(curve.gasSaturation(:));
    pcBar = double(curve.pcBar(:));
    assert(numel(sg) == numel(pcBar) && numel(sg) >= 3, ...
        'PeBranchMap:InvalidCurve', ...
        'A branch-medoid Pc curve is incomplete.');
    entryId = find(sg > 1.0e-5 * (1 + 1.0e-8), 1, 'first');
    assert(~isempty(entryId) && isfinite(pcBar(entryId)) && ...
        pcBar(entryId) > 0, 'PeBranchMap:MissingEntryPressure', ...
        'A branch-medoid Pc curve lacks a connected-path entry point.');
    peMapBar(i) = pcBar(entryId);
end
end


function titleText = displayCaseTitle(reservoirReady)
% Keep the established publication labels for the four reference cases.

caseId = double(reservoirReady.level3CaseId);
switch caseId
    case 1
        description = 'Independent draw 1';
    case 3
        description = 'Fault-wide low, local';
    case 4
        description = 'Fault-wide high, local';
    case 7
        description = 'Grouped low/high, local';
    otherwise
        description = strrep(char(reservoirReady.level3CaseName), '_', ' ');
        if ~isempty(description)
            description(1) = upper(description(1));
        end
end
titleText = sprintf('Case %02d | %s', caseId, description);
end


function formatAxes(ax, windowLabels)
% Match the axes used by the full-slice spatial entry-pressure figure.

xticks(ax, [1, 15, 29, 43, 58, 72, 87]);
yticks(ax, 1:6);
yticklabels(ax, windowLabels);
ax.FontSize = 17;
ax.LineWidth = 1.1;
ax.TickDir = 'out';
ax.Layer = 'top';
ax.Toolbar.Visible = 'off';
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
    error('PeBranchMap:UnsupportedInput', ...
        'Inputs must be MAT paths, structures, or cell arrays.');
end
end


function reservoirReady = loadReservoirReady(value)
% Load one reservoirReady structure from memory or MAT storage.

if isstruct(value)
    if isfield(value, 'reservoirReady')
        reservoirReady = value.reservoirReady;
    else
        reservoirReady = value;
    end
else
    file = string(value);
    assert(isscalar(file) && isfile(file), ...
        'PeBranchMap:MissingInput', 'MAT input not found: %s', file);
    loaded = load(file, 'reservoirReady');
    assert(isfield(loaded, 'reservoirReady'), ...
        'PeBranchMap:MissingVariable', ...
        'MAT input does not contain reservoirReady: %s', file);
    reservoirReady = loaded.reservoirReady;
end
end


function ensureFolder(folderPath)
% Create the output folder when needed.

if ~isfolder(folderPath)
    mkdir(folderPath);
end
end


function cmap = entryPressureMap(n)
% Sequential colorblind-friendly map: low pressure blue, high pressure gold.

anchors = [ ...
    0.08, 0.18, 0.42; ...
    0.10, 0.43, 0.67; ...
    0.25, 0.68, 0.70; ...
    0.88, 0.82, 0.47; ...
    0.88, 0.50, 0.12; ...
    0.60, 0.24, 0.08];
x = linspace(0, 1, size(anchors, 1));
xq = linspace(0, 1, n);
cmap = interp1(x, anchors, xq, 'pchip');
cmap = min(max(cmap, 0), 1);
end
