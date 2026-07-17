function outputFiles = plot_effective_swi_spatial_maps( ...
        reservoirInputs, outputDir, options)
%PLOT_EFFECTIVE_SWI_SPATIAL_MAPS Plot 6-by-87 effective-Swi fields.
%
%   OUTPUTFILES = PLOT_EFFECTIVE_SWI_SPATIAL_MAPS(INPUTS, OUTPUTDIR)
%   reads full-slice or Pe-branch-medoid reservoir-ready MAT files and
%   plots the upscaled effective irreducible water saturation assigned to
%   every throw-window/slice pair. The format matches the corresponding
%   entry-pressure spatial-map figures so the two quantities can be
%   reviewed together.
%
%   INPUTS may be a MAT-file path, a string array of paths, a
%   reservoirReady structure, or a cell array containing those values.
%   All inputs in one call must use the same Pc representation.
%
%   Name-value options:
%     ColorLimits - shared linear color limits, default [0.12, 0.28]
%
%   The function verifies complete 6-by-87 coverage and checks that every
%   curve satisfies EffectiveSwi = 1 - BulkSgMax before plotting.

arguments
    reservoirInputs
    outputDir {mustBeTextScalar}
    options.ColorLimits (1, 2) double = [0.12, 0.28]
end

reservoirInputs = normalizeInputs(reservoirInputs);
assert(~isempty(reservoirInputs), 'EffectiveSwiMap:EmptyInput', ...
    'At least one reservoir-ready input is required.');
assert(all(isfinite(options.ColorLimits)) && ...
    options.ColorLimits(1) < options.ColorLimits(2), ...
    'EffectiveSwiMap:InvalidColorLimits', ...
    'ColorLimits must contain two finite increasing values.');
ensureFolder(outputDir);

nCases = numel(reservoirInputs);
caseIds = nan(nCases, 1);
caseTitles = strings(nCases, 1);
maps = cell(nCases, 1);
caseMean = nan(nCases, 1);
caseMinimum = nan(nCases, 1);
caseMaximum = nan(nCases, 1);
meanWithinWindowSd = nan(nCases, 1);
distinctValueCount = nan(nCases, 1);
representation = strings(nCases, 1);
windowLabels = compose("W%d", 1:6);

for c = 1:nCases
    reservoirReady = loadReservoirReady(reservoirInputs{c});
    assert(isfield(reservoirReady, 'pcCurves') && ...
        isequal(size(reservoirReady.pcCurves), [6, 87]), ...
        'EffectiveSwiMap:UnexpectedCoverage', ...
        'Input %d must contain a 6-by-87 pcCurves cell array.', c);

    caseIds(c) = double(reservoirReady.level3CaseId);
    caseTitles(c) = displayCaseTitle(reservoirReady);
    representation(c) = identifyRepresentation(reservoirReady);
    swiMap = extractEffectiveSwi(reservoirReady.pcCurves);

    assert(all(swiMap(:) >= options.ColorLimits(1) & ...
        swiMap(:) <= options.ColorLimits(2)), ...
        'EffectiveSwiMap:ColorLimitsExcludeData', ...
        ['Color limits [%.4g, %.4g] exclude values in case %02d ', ...
        '[%.4g, %.4g].'], options.ColorLimits(1), ...
        options.ColorLimits(2), caseIds(c), min(swiMap(:)), ...
        max(swiMap(:)));

    maps{c} = swiMap;
    caseMean(c) = mean(swiMap(:));
    caseMinimum(c) = min(swiMap(:));
    caseMaximum(c) = max(swiMap(:));
    meanWithinWindowSd(c) = mean(std(swiMap, 0, 2));
    distinctValueCount(c) = numel(unique(swiMap(:)));
end

assert(isscalar(unique(representation)), ...
    'EffectiveSwiMap:MixedRepresentations', ...
    'All inputs in one figure must use the same Pc representation.');
representation = representation(1);
[representationLabel, outputPrefix, subtitleText] = ...
    representationMetadata(representation);

cmap = effectiveSwiMap(256);
figHeight = max(720, 290 * nCases + 160);
fig = figure('Color', 'w', 'Position', [50, 40, 2300, figHeight]);
layout = tiledlayout(fig, nCases, 1, 'TileSpacing', 'compact', ...
    'Padding', 'compact');

for c = 1:nCases
    ax = nexttile(layout);
    imagesc(ax, 1:87, 1:6, maps{c});
    set(ax, 'YDir', 'normal');
    axis(ax, 'image');
    clim(ax, options.ColorLimits);
    colormap(ax, cmap);
    formatAxes(ax, windowLabels);

    title(ax, sprintf( ...
        '%s | mean S_{wi,eff} = %.3f | mean within-window SD = %.3f', ...
        caseTitles(c), caseMean(c), meanWithinWindowSd(c)), ...
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
cb.Ticks = linspace(options.ColorLimits(1), options.ColorLimits(2), 5);
cb.TickLabels = compose('%.2f', cb.Ticks);
cb.Label.String = 'Upscaled effective S_{wi} [-]';
cb.Label.Interpreter = 'tex';
cb.Label.FontSize = 19;
cb.FontSize = 16;

title(layout, ...
    'Spatial distribution of upscaled effective irreducible water saturation', ...
    'FontSize', 29, 'FontWeight', 'bold');
subtitle(layout, subtitleText, 'FontSize', 17);

idToken = strjoin(compose('%02d', caseIds), '_');
base = fullfile(outputDir, sprintf( ...
    '%s_effective_swi_spatial_maps_cases_%s', outputPrefix, idToken));
outputFiles = struct();
outputFiles.png = string(base + ".png");
outputFiles.pdf = string(base + ".pdf");
outputFiles.summaryCsv = string(fullfile(outputDir, sprintf( ...
    '%s_effective_swi_summary_cases_%s.csv', outputPrefix, idToken)));
exportgraphics(fig, outputFiles.png, 'Resolution', 300);
exportgraphics(fig, outputFiles.pdf, 'ContentType', 'vector');
close(fig);

representationColumn = repmat(representationLabel, nCases, 1);
summary = table(caseIds, caseTitles, representationColumn, caseMean, ...
    caseMinimum, caseMaximum, meanWithinWindowSd, distinctValueCount, ...
    'VariableNames', {'CaseId', 'CaseTitle', 'PcRepresentation', ...
    'MeanEffectiveSwi', 'MinimumEffectiveSwi', 'MaximumEffectiveSwi', ...
    'MeanWithinWindowSdEffectiveSwi', 'DistinctEffectiveSwiValues'});
writetable(summary, outputFiles.summaryCsv);

fprintf('Saved %s effective-Swi spatial map: %s\n', ...
    representationLabel, outputFiles.png);
end


function swiMap = extractEffectiveSwi(pcCurves)
% Read effective Swi and verify its native Pc endpoint definition.

swiMap = nan(size(pcCurves));
for i = 1:numel(pcCurves)
    curve = pcCurves{i};
    assert(isfield(curve, 'effectiveSwi') && ...
        isfield(curve, 'bulkSgMax'), ...
        'EffectiveSwiMap:MissingEndpointFields', ...
        'Every Pc curve must contain effectiveSwi and bulkSgMax.');
    swi = double(curve.effectiveSwi);
    bulkSgMax = double(curve.bulkSgMax);
    assert(isscalar(swi) && isfinite(swi) && swi >= 0 && swi <= 1, ...
        'EffectiveSwiMap:InvalidEffectiveSwi', ...
        'Every effectiveSwi value must be a finite scalar in [0, 1].');
    assert(isscalar(bulkSgMax) && isfinite(bulkSgMax) && ...
        bulkSgMax >= 0 && bulkSgMax <= 1, ...
        'EffectiveSwiMap:InvalidBulkSgMax', ...
        'Every bulkSgMax value must be a finite scalar in [0, 1].');
    assert(abs(swi - (1 - bulkSgMax)) <= 1.0e-10, ...
        'EffectiveSwiMap:EndpointMismatch', ...
        'effectiveSwi must equal 1 - bulkSgMax for every Pc curve.');
    swiMap(i) = swi;
end
end


function representation = identifyRepresentation(reservoirReady)
% Infer legacy full-slice files and read explicit reduced-file metadata.

if isfield(reservoirReady, 'pcRepresentation')
    representation = string(reservoirReady.pcRepresentation);
elseif isfield(reservoirReady, 'peBranchModel')
    representation = "pe_branch_medoid";
else
    representation = "full_slice";
end
assert(any(representation == ["full_slice", "pe_branch_medoid"]), ...
    'EffectiveSwiMap:UnknownRepresentation', ...
    'Unsupported Pc representation: %s', representation);
end


function [label, outputPrefix, subtitleText] = ...
        representationMetadata(representation)
% Supply matched figure text without duplicating the plotting workflow.

switch representation
    case "full_slice"
        label = "full_slice";
        outputPrefix = "full_slice";
        subtitleText = [ ...
            'Full-slice option: six throw windows by 87 along-strike ', ...
            'slices; W1 is at the bottom. Each slice retains its native ', ...
            'upscaled effective S_{wi}; both options use the same ', ...
            'linear color scale.'];
    case "pe_branch_medoid"
        label = "pe_branch_medoid";
        outputPrefix = "pe_branch_medoid";
        subtitleText = [ ...
            'Pe-branch-medoid option: six throw windows by 87 ', ...
            'along-strike slices; W1 is at the bottom. Each slice uses ', ...
            'the effective S_{wi} of its assigned Pc-curve medoid; both ', ...
            'options use the same linear color scale.'];
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
% Match the axes used by the entry-pressure spatial figures.

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
    error('EffectiveSwiMap:UnsupportedInput', ...
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
        'EffectiveSwiMap:MissingInput', 'MAT input not found: %s', file);
    loaded = load(file, 'reservoirReady');
    assert(isfield(loaded, 'reservoirReady'), ...
        'EffectiveSwiMap:MissingVariable', ...
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


function cmap = effectiveSwiMap(n)
% Sequential colorblind-friendly map: low Swi blue, high Swi orange-red.

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
