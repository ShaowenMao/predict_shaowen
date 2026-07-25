function outputs = plot_qualification_batch_visuals(batchRoot, outputDir, options)
%PLOT_QUALIFICATION_BATCH_VISUALS Create visual QA for a 24-case batch.
%
%   OUTPUTS = PLOT_QUALIFICATION_BATCH_VISUALS(BATCHROOT, OUTPUTDIR)
%   reads the reservoir-ready MAT files produced by the six-scenario
%   qualification batch. It creates:
%
%     * one distribution summary covering all 24 production cases;
%     * fixed-scale spatial atlases for kxx, kyy, kzz, porosity, entry
%       pressure, and effective irreducible water saturation; and
%     * window-resolved marginal permeability distributions for Case 07
%       in every thickness scenario; and
%     * one focused three-component Case 07 spatial map for the
%       representative geology, using component-specific color limits; and
%     * detailed Pc and Kr curve atlases for one representative geology.
%
%   Spatial atlases use four case rows and six thickness-scenario columns.
%   Every panel is a complete 6-window by 87-slice fault-property field.
%   Common color limits are retained across all 24 panels in each atlas.
%
%   Name-value option:
%     RepresentativeGeology - geology used for the detailed curve atlases.
%                             Default: "s05_c012".

arguments
    batchRoot (1, 1) string
    outputDir (1, 1) string
    options.RepresentativeGeology (1, 1) string = "s05_c012"
end

geologyIds = compose("s%02d_c012", 1:6);
caseIds = [1, 3, 4, 7];
assert(isfolder(batchRoot), 'QualificationPlots:MissingBatch', ...
    'Qualification batch folder does not exist: %s', batchRoot);
assert(any(options.RepresentativeGeology == geologyIds), ...
    'QualificationPlots:UnknownRepresentative', ...
    'RepresentativeGeology must be one of: %s', ...
    strjoin(geologyIds, ', '));
ensureFolder(outputDir);

records = cell(numel(geologyIds), numel(caseIds));
for g = 1:numel(geologyIds)
    for c = 1:numel(caseIds)
        records{g, c} = loadQualificationRecord( ...
            batchRoot, geologyIds(g), caseIds(c));
    end
end

summaryCsv = writePropertySummary(records, geologyIds, caseIds, outputDir);
summaryFigure = plotPropertySummary( ...
    records, geologyIds, caseIds, outputDir);

allPe = collectProperty(records, "entry_pressure");
allSwi = collectProperty(records, "effective_swi");
allPhi = collectProperty(records, "porosity");

spatialSpecs = [ ...
    makeSpatialSpec("log_kxx", "Upscaled log_{10}(k_{xx})", ...
        "log_{10}(k_{xx}) [mD]", [-7, 2.5], ...
        [-7, -4, -1, 2], compose('%g', [-7, -4, -1, 2])), ...
    makeSpatialSpec("log_kyy", "Upscaled log_{10}(k_{yy})", ...
        "log_{10}(k_{yy}) [mD]", [-7, 2.5], ...
        [-7, -4, -1, 2], compose('%g', [-7, -4, -1, 2])), ...
    makeSpatialSpec("log_kzz", "Upscaled log_{10}(k_{zz})", ...
        "log_{10}(k_{zz}) [mD]", [-7, 2.5], ...
        [-7, -4, -1, 2], compose('%g', [-7, -4, -1, 2])), ...
    linearSpatialSpec("porosity", "Upscaled porosity", ...
        "Porosity [-]", allPhi, 5, '%.2f'), ...
    logPressureSpatialSpec(allPe), ...
    linearSpatialSpec("effective_swi", ...
        "Upscaled effective irreducible water saturation", ...
        "Effective S_{wi} [-]", allSwi, 5, '%.2f')];

spatialFigures = strings(numel(spatialSpecs), 1);
for i = 1:numel(spatialSpecs)
    spatialFigures(i) = plotSpatialAtlas( ...
        records, geologyIds, caseIds, spatialSpecs(i), outputDir);
end

case07Index = find(caseIds == 7, 1);
case07DistributionFigures = strings(numel(geologyIds), 1);
for g = 1:numel(geologyIds)
    case07DistributionFigures(g) = ...
        plotCase07PermeabilityDistributions( ...
        records{g, case07Index}, outputDir);
end

representativeId = options.RepresentativeGeology;
representativeIndex = find(geologyIds == representativeId, 1);
case07SpatialFigure = plotCase07PermeabilitySpatialMaps( ...
    records{representativeIndex, case07Index}, outputDir);
pcFigure = plotPcCurveAtlas( ...
    records(representativeIndex, :), representativeId, caseIds, outputDir);
krFigure = plotKrCurveAtlas( ...
    records(representativeIndex, :), representativeId, caseIds, outputDir);

outputs = struct();
outputs.summaryCsv = summaryCsv;
outputs.summaryFigure = summaryFigure;
outputs.spatialFigures = spatialFigures;
outputs.case07DistributionFigures = case07DistributionFigures;
outputs.case07SpatialFigure = case07SpatialFigure;
outputs.pcCurveFigure = pcFigure;
outputs.krCurveFigure = krFigure;

fprintf('Saved qualification visual QA package: %s\n', outputDir);
end


function file = plotCase07PermeabilityDistributions(record, outputDir)
% Plot Case 07 marginal permeability distributions by window and component.

componentKeys = ["log_kxx", "log_kyy", "log_kzz"];
componentLabels = ["log_{10}(k_{xx}) [mD]", ...
    "log_{10}(k_{yy}) [mD]", "log_{10}(k_{zz}) [mD]"];
binEdges = -7:0.25:2.5;
binCenters = 0.5 .* (binEdges(1:end-1) + binEdges(2:end));
barColor = [0.33, 0.38, 0.44];
medianColor = [0.86, 0.18, 0.12];
quartileColor = [0.25, 0.25, 0.25];

fig = figure('Color', 'w', 'Position', [30, 30, 2450, 1400]);
layout = tiledlayout(fig, 3, 6, ...
    'TileSpacing', 'compact', 'Padding', 'loose');
legendHandles = gobjects(3, 1);

for component = 1:numel(componentKeys)
    values = extractProperty(record, componentKeys(component));
    for window = 1:6
        ax = nexttile(layout);
        hold(ax, 'on');
        sample = double(values(window, :));
        probability = histcounts(sample, binEdges, ...
            'Normalization', 'probability');
        bar(ax, binCenters, probability, 1.0, ...
            'FaceColor', barColor, 'EdgeColor', 'none');
        q25 = percentile(sample, 0.25);
        q50 = percentile(sample, 0.50);
        q75 = percentile(sample, 0.75);
        xline(ax, q25, '--', 'Color', quartileColor, ...
            'LineWidth', 1.2);
        xline(ax, q75, '--', 'Color', quartileColor, ...
            'LineWidth', 1.2);
        xline(ax, q50, '-', 'Color', medianColor, ...
            'LineWidth', 2.0);

        if component == 1 && window == 1
            legendHandles(1) = bar(ax, NaN, NaN, 1.0, ...
                'FaceColor', barColor, 'EdgeColor', 'none');
            legendHandles(2) = plot(ax, NaN, NaN, '-', ...
                'Color', medianColor, 'LineWidth', 2.0);
            legendHandles(3) = plot(ax, NaN, NaN, '--', ...
                'Color', quartileColor, 'LineWidth', 1.2);
        end

        xlim(ax, [-7, 2.5]);
        ylim(ax, [0, 1]);
        xticks(ax, [-7, -4, -1, 2]);
        yticks(ax, 0:0.25:1);
        grid(ax, 'on');
        ax.GridColor = [0.84, 0.84, 0.84];
        ax.GridAlpha = 0.50;
        ax.FontName = 'Arial';
        ax.FontSize = 12;
        ax.LineWidth = 0.9;
        ax.TickDir = 'out';
        box(ax, 'on');

        if component == 1
            title(ax, sprintf('W%d', window), ...
                'FontSize', 16, 'FontWeight', 'bold');
        end
        if component == 3
            xlabel(ax, componentLabels(component), ...
                'FontSize', 13, 'Interpreter', 'tex');
        else
            ax.XTickLabel = [];
        end
        if window == 1
            ylabel(ax, {componentLabels(component); 'Probability'}, ...
                'FontSize', 13, 'Interpreter', 'tex');
        else
            ax.YTickLabel = [];
        end
    end
end

title(layout, sprintf('%s | %s permeability distributions', ...
    scenarioLabel(record.geologyId), recordCaseLabel(record)), ...
    'FontSize', 25, 'FontWeight', 'bold');
subtitle(layout, ['Each panel contains 87 along-strike slices. ', ...
    'All windows share fixed bins and axes.'], 'FontSize', 14);
lgd = legend(legendHandles, ...
    {'Empirical slice probability', 'Median', '25th/75th percentiles'}, ...
    'Location', 'northwest');
lgd.FontSize = 10;
lgd.Box = 'on';

file = fullfile(outputDir, record.geologyId + ...
    "_case07_permeability_distributions.png");
exportgraphics(fig, file, 'Resolution', 260);
close(fig);
end


function file = plotCase07PermeabilitySpatialMaps(record, outputDir)
% Plot all three permeability components for one Case 07 field.

componentKeys = ["log_kxx", "log_kyy", "log_kzz"];
componentTitles = ["Across-fault permeability, k_{xx}", ...
    "Along-strike permeability, k_{yy}", ...
    "Down-dip permeability, k_{zz}"];
componentLabels = ["log_{10}(k_{xx}) [mD]", ...
    "log_{10}(k_{yy}) [mD]", "log_{10}(k_{zz}) [mD]"];

fig = figure('Color', 'w', 'Position', [30, 30, 2200, 1120]);
layout = tiledlayout(fig, 3, 1, ...
    'TileSpacing', 'compact', 'Padding', 'loose');

for component = 1:numel(componentKeys)
    ax = nexttile(layout);
    values = extractProperty(record, componentKeys(component));
    imagesc(ax, 1:87, 1:6, values);
    set(ax, 'YDir', 'normal');
    colormap(ax, qualificationMap(256));

    finiteValues = values(isfinite(values));
    limits = roundedColorLimits(finiteValues, 0.5);
    clim(ax, limits);
    ticks = linspace(limits(1), limits(2), 4);

    hold(ax, 'on');
    yline(ax, 1.5:1:5.5, '-', 'Color', [0.88, 0.88, 0.88], ...
        'LineWidth', 0.7);
    hold(ax, 'off');

    pbaspect(ax, [8.5, 1, 1]);
    xlim(ax, [0.5, 87.5]);
    ylim(ax, [0.5, 6.5]);
    xticks(ax, [1, 15, 29, 43, 58, 72, 87]);
    yticks(ax, 1:6);
    yticklabels(ax, compose("W%d", 1:6));
    xlabel(ax, 'Along-strike slice', 'FontSize', 15);
    ylabel(ax, 'Throw window', 'FontSize', 15);
    title(ax, componentTitles(component), ...
        'FontSize', 18, 'FontWeight', 'bold', 'Interpreter', 'tex');
    ax.FontName = 'Arial';
    ax.FontSize = 13;
    ax.LineWidth = 0.9;
    ax.TickDir = 'out';
    box(ax, 'on');

    cb = colorbar(ax, 'eastoutside');
    cb.Ticks = ticks;
    cb.TickLabels = compose('%.1f', ticks);
    cb.Label.String = componentLabels(component);
    cb.Label.FontSize = 14;
    cb.Label.Interpreter = 'tex';
    cb.FontSize = 12;
end

title(layout, sprintf('%s | %s permeability field', ...
    scenarioLabel(record.geologyId), recordCaseLabel(record)), ...
    'FontSize', 25, 'FontWeight', 'bold');
subtitle(layout, ['Six throw windows by 87 along-strike slices. ', ...
    'Each component uses its own full observed color range.'], ...
    'FontSize', 14);

file = fullfile(outputDir, record.geologyId + ...
    "_case07_permeability_spatial_maps.png");
exportgraphics(fig, file, 'Resolution', 260);
close(fig);
end


function record = loadQualificationRecord(batchRoot, geologyId, caseId)
% Load one production MAT and retain compact fields used by the figures.

matFile = fullfile(batchRoot, "cases", geologyId, ...
    sprintf("case%02d", caseId), "kr_dyn_swi_medoid", ...
    "reservoir_ready", sprintf( ...
    "reservoir_ready_%s_case%02d.mat", geologyId, caseId));
assert(isfile(matFile), 'QualificationPlots:MissingMat', ...
    'Production reservoir-ready MAT not found: %s', matFile);

loaded = load(matFile, 'reservoirReady');
assert(isfield(loaded, 'reservoirReady'), ...
    'QualificationPlots:MissingVariable', ...
    'MAT does not contain reservoirReady: %s', matFile);
rr = loaded.reservoirReady;
assert(isequal(size(rr.effectivePermeability.mD), [6, 87, 3]), ...
    'QualificationPlots:PermeabilityCoverage', ...
    'Permeability must have size 6-by-87-by-3: %s', matFile);
assert(isequal(size(rr.upscaledPorosity), [6, 87]) && ...
    isequal(size(rr.pcCurves), [6, 87]) && ...
    isequal(size(rr.krCurves), [6, 87]), ...
    'QualificationPlots:PropertyCoverage', ...
    'Porosity, Pc, and Kr must each cover 6 windows by 87 slices.');

record = struct();
record.matFile = string(matFile);
record.geologyId = string(rr.geologyId);
record.caseId = double(rr.level3CaseId);
record.caseName = string(rr.level3CaseName);
record.logK = log10(double(rr.effectivePermeability.mD));
record.porosity = double(rr.upscaledPorosity);
record.entryPressure = extractEntryPressure(rr.pcCurves);
record.effectiveSwi = extractEffectiveSwi(rr.pcCurves);
end


function pe = extractEntryPressure(pcCurves)
% Read the first connected state after the explicit pre-entry anchor.

pe = nan(size(pcCurves));
for i = 1:numel(pcCurves)
    curve = pcCurves{i};
    sg = double(curve.gasSaturation(:));
    pc = double(curve.pcBar(:));
    assert(numel(sg) == numel(pc) && numel(sg) >= 3 && ...
        all(diff(sg) >= 0), 'QualificationPlots:InvalidPcCurve', ...
        'Pc curves must contain an ordered anchor and connected states.');
    assert(sg(1) < sg(2), 'QualificationPlots:InvalidPcAnchor', ...
        'The pre-entry anchor must precede the first connected state.');
    pe(i) = pc(2);
end
end


function swi = extractEffectiveSwi(pcCurves)
% Extract Pc-derived endpoints and verify their saturation identity.

swi = nan(size(pcCurves));
for i = 1:numel(pcCurves)
    curve = pcCurves{i};
    value = double(curve.effectiveSwi);
    bulkSgMax = double(curve.bulkSgMax);
    assert(isfinite(value) && value >= 0 && value <= 1 && ...
        abs(value - (1 - bulkSgMax)) <= 1.0e-10, ...
        'QualificationPlots:InvalidSwi', ...
        'Every curve must satisfy effectiveSwi = 1 - bulkSgMax.');
    swi(i) = value;
end
end


function file = writePropertySummary(records, geologyIds, caseIds, outputDir)
% Save median and 10th-90th percentile summaries used in the overview.

metricKeys = ["log_kxx", "log_kyy", "log_kzz", ...
    "porosity", "log_entry_pressure", "effective_swi"];
metricLabels = ["log10(kxx [mD])", "log10(kyy [mD])", ...
    "log10(kzz [mD])", "porosity", "log10(Pe [bar])", ...
    "effective Swi"];
nRows = numel(geologyIds) * numel(caseIds) * numel(metricKeys);
rows = cell(nRows, 9);
row = 0;
for g = 1:numel(geologyIds)
    for c = 1:numel(caseIds)
        for m = 1:numel(metricKeys)
            row = row + 1;
            values = extractProperty(records{g, c}, metricKeys(m));
            values = values(isfinite(values));
            rows(row, :) = {geologyIds(g), scenarioLabel(geologyIds(g)), ...
                caseIds(c), recordCaseLabel(records{g, c}), metricKeys(m), ...
                metricLabels(m), percentile(values, 0.10), ...
                percentile(values, 0.50), percentile(values, 0.90)};
        end
    end
end
summary = cell2table(rows, 'VariableNames', { ...
    'GeologyId', 'ThicknessScenario', 'CaseId', 'CaseLabel', ...
    'Metric', 'MetricLabel', 'P10', 'Median', 'P90'});
file = fullfile(outputDir, "qualification_property_summary.csv");
writetable(summary, file);
end


function file = plotPropertySummary(records, geologyIds, caseIds, outputDir)
% Plot 10th-90th percentile intervals and medians for all 24 cases.

metricKeys = ["log_kxx", "log_kyy", "log_kzz", ...
    "porosity", "log_entry_pressure", "effective_swi"];
metricTitles = ["Across-fault permeability, kxx", ...
    "Along-strike permeability, kyy", ...
    "Down-dip permeability, kzz", "Upscaled porosity", ...
    "Entry capillary pressure", "Effective irreducible water saturation"];
yLabels = ["log_{10}(k_{xx}) [mD]", "log_{10}(k_{yy}) [mD]", ...
    "log_{10}(k_{zz}) [mD]", "Porosity [-]", ...
    "log_{10}(P_e) [bar]", "Effective S_{wi} [-]"];
colors = caseColors();
offsets = linspace(-0.27, 0.27, numel(caseIds));

fig = figure('Color', 'w', 'Position', [40, 30, 2300, 1450]);
layout = tiledlayout(fig, 2, 3, ...
    'TileSpacing', 'compact', 'Padding', 'compact');

for m = 1:numel(metricKeys)
    ax = nexttile(layout);
    hold(ax, 'on');
    for c = 1:numel(caseIds)
        medians = nan(numel(geologyIds), 1);
        lower = nan(numel(geologyIds), 1);
        upper = nan(numel(geologyIds), 1);
        for g = 1:numel(geologyIds)
            values = extractProperty(records{g, c}, metricKeys(m));
            values = values(isfinite(values));
            medians(g) = percentile(values, 0.50);
            lower(g) = medians(g) - percentile(values, 0.10);
            upper(g) = percentile(values, 0.90) - medians(g);
        end
        x = (1:numel(geologyIds)) + offsets(c);
        errorbar(ax, x, medians, lower, upper, 'o', ...
            'Color', colors(c, :), 'MarkerFaceColor', colors(c, :), ...
            'MarkerEdgeColor', 'w', 'MarkerSize', 8, ...
            'LineWidth', 2.0, 'CapSize', 7);
    end
    xlim(ax, [0.5, numel(geologyIds) + 0.5]);
    xticks(ax, 1:numel(geologyIds));
    xticklabels(ax, scenarioTickLabels());
    ylabel(ax, yLabels(m), 'FontSize', 17, 'Interpreter', 'tex');
    title(ax, metricTitles(m), 'FontSize', 19, 'FontWeight', 'bold');
    formatSummaryAxes(ax);
end

layout.OuterPosition = [0.03, 0.03, 0.94, 0.86];
annotation(fig, 'textbox', [0.08, 0.945, 0.84, 0.04], ...
    'String', 'Qualification batch: property distributions across 24 cases', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'FontName', 'Arial', 'FontSize', 25, 'FontWeight', 'bold', ...
    'EdgeColor', 'none');
annotation(fig, 'textbox', [0.08, 0.905, 0.84, 0.035], ...
    'String', ['Points are medians; bars span the 10th-90th percentiles. ', ...
    'Within each scenario, left to right: C01 independent, C03 low, ', ...
    'C04 high, C07 geology-specific.'], ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'FontName', 'Arial', 'FontSize', 14, 'EdgeColor', 'none');

file = fullfile(outputDir, "qualification_property_distribution_summary.png");
exportgraphics(fig, file, 'Resolution', 260);
close(fig);
end


function spec = makeSpatialSpec(key, titleText, colorbarText, ...
        colorLimits, ticks, tickLabels)
% Construct one explicit spatial-atlas rendering specification.

spec = struct('key', string(key), 'title', string(titleText), ...
    'colorbar', string(colorbarText), ...
    'limits', double(colorLimits), 'ticks', double(ticks), ...
    'tickLabels', string(tickLabels), 'mapType', "sequential");
end


function spec = linearSpatialSpec( ...
        key, titleText, colorbarText, values, nTicks, formatText)
% Build shared linear limits without excluding qualification values.

limits = paddedLimits(values, 0.04);
ticks = linspace(limits(1), limits(2), nTicks);
tickLabels = compose(formatText, ticks);
spec = makeSpatialSpec( ...
    key, titleText, colorbarText, limits, ticks, tickLabels);
end


function spec = logPressureSpatialSpec(entryPressure)
% Build one physical-pressure color scale shared by all 24 cases.

logValues = log10(entryPressure(:));
limits = paddedLimits(logValues, 0.04);
physicalCandidates = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300];
mask = log10(physicalCandidates) >= limits(1) & ...
    log10(physicalCandidates) <= limits(2);
physicalTicks = physicalCandidates(mask);
if numel(physicalTicks) > 6
    pick = round(linspace(1, numel(physicalTicks), 6));
    physicalTicks = physicalTicks(pick);
end
spec = makeSpatialSpec("log_entry_pressure", ...
    "Upscaled entry capillary pressure", ...
    "Entry pressure, P_e [bar]", limits, log10(physicalTicks), ...
    compose('%g', physicalTicks));
end


function file = plotSpatialAtlas( ...
        records, geologyIds, caseIds, spec, outputDir)
% Plot one property for all six geologies and four case designs.

fig = figure('Color', 'w', 'Position', [20, 20, 2900, 760]);
layout = tiledlayout(fig, numel(caseIds), numel(geologyIds), ...
    'TileSpacing', 'compact', 'Padding', 'compact');

for c = 1:numel(caseIds)
    for g = 1:numel(geologyIds)
        ax = nexttile(layout);
        values = extractProperty(records{g, c}, spec.key);
        imagesc(ax, 1:87, 1:6, values);
        set(ax, 'YDir', 'normal');
        clim(ax, spec.limits);
        colormap(ax, qualificationMap(256));
        pbaspect(ax, [4.0, 1, 1]);
        formatSpatialAxes(ax, g, c, numel(caseIds));
        if c == 1
            title(ax, scenarioLabel(geologyIds(g)), ...
                'FontSize', 17, 'FontWeight', 'bold');
        end
        if g == 1
            text(ax, 0.02, 0.82, compose("C%02d", caseIds(c)), ...
                'Units', 'normalized', 'FontSize', 12, ...
                'FontWeight', 'bold', 'Color', [0.05, 0.05, 0.05], ...
                'BackgroundColor', 'w', 'Margin', 2);
        end
    end
end

cb = colorbar(ax, 'eastoutside');
cb.Layout.Tile = 'east';
cb.Ticks = spec.ticks;
cb.TickLabels = spec.tickLabels;
cb.Label.String = spec.colorbar;
cb.Label.FontSize = 18;
cb.Label.Interpreter = 'tex';
cb.FontSize = 15;

title(layout, spec.title + " across the qualification batch", ...
    'FontSize', 25, 'FontWeight', 'bold', 'Interpreter', 'tex');
subtitle(layout, ['Rows are Level 3 case designs; columns are thickness ', ...
    'scenarios. C01 = independent; C03 = low; C04 = high; ', ...
    'C07 = grouped when supported, otherwise an independent fallback. ', ...
    'W1 is at the bottom; all 87 slices are shown.'], ...
    'FontSize', 14);

file = fullfile(outputDir, "spatial_atlas_" + spec.key + ".png");
exportgraphics(fig, file, 'Resolution', 260);
close(fig);
end


function file = plotPcCurveAtlas(records, geologyId, caseIds, outputDir)
% Plot native-endpoint Pc curves for four cases of one geology.

fig = figure('Color', 'w', 'Position', [20, 20, 2850, 1700]);
layout = tiledlayout(fig, numel(caseIds), 6, ...
    'TileSpacing', 'compact', 'Padding', 'compact');
gray = [0.68, 0.72, 0.77];
red = [0.88, 0.20, 0.12];
legendHandles = gobjects(2, 1);
legendAxis = gobjects(1);

for c = 1:numel(caseIds)
    rr = loadReservoirReady(records{c}.matFile);
    for w = 1:6
        ax = nexttile(layout);
        hold(ax, 'on');
        set(ax, 'YScale', 'log');
        selectedRow = selectedSwiMedoidReplayRow(rr, w);
        selectedCurve = [];
        for s = 1:87
            curve = rr.pcCurves{w, s};
            semilogy(ax, curve.gasSaturation, curve.pcBar, ...
                '-', 'Color', gray, 'LineWidth', 0.55);
            if double(curve.replaySourceRow) == selectedRow
                selectedCurve = curve;
            end
        end
        assert(~isempty(selectedCurve), ...
            'QualificationPlots:MissingSelectedPc', ...
            'Swi-medoid Pc slice was not found for %s case %02d W%d.', ...
            geologyId, caseIds(c), w);
        semilogy(ax, selectedCurve.gasSaturation, selectedCurve.pcBar, ...
            '-', 'Color', red, 'LineWidth', 2.7);
        if c == 1 && w == 1
            legendAxis = ax;
            legendHandles(1) = semilogy(ax, NaN, NaN, '-', ...
                'Color', gray, 'LineWidth', 1.2);
            legendHandles(2) = semilogy(ax, NaN, NaN, '-', ...
                'Color', red, 'LineWidth', 2.7);
        end
        xlim(ax, [0, 1]);
        ylim(ax, [1.0e-2, 1.0e3]);
        formatCurveAxes(ax, c, w, caseIds, "P_c [bar]");
    end
end

title(layout, sprintf( ...
    '%s: native-endpoint invasion-percolation Pc curves', ...
    scenarioLabel(geologyId)), ...
    'FontSize', 25, 'FontWeight', 'bold');
subtitle(layout, ...
    'Grey = 87 slices; red = Pc curve at the Swi-medoid slice used for Kr', ...
    'FontSize', 15);
lgd = legend(legendAxis, legendHandles, ...
    {'87 native slice curves', 'Swi-medoid selection slice'}, ...
    'Location', 'northwest');
lgd.FontSize = 11;
lgd.Box = 'on';

file = fullfile(outputDir, geologyId + "_pc_curve_atlas.png");
exportgraphics(fig, file, 'Resolution', 240);
close(fig);
end


function file = plotKrCurveAtlas(records, geologyId, caseIds, outputDir)
% Plot Pc-endpoint-scaled dynamic Kr curves for one representative geology.

fig = figure('Color', 'w', 'Position', [20, 20, 2850, 1700]);
layout = tiledlayout(fig, numel(caseIds), 6, ...
    'TileSpacing', 'compact', 'Padding', 'compact');
gasLight = [0.93, 0.68, 0.43];
waterLight = [0.57, 0.72, 0.87];
gasDark = [0.84, 0.30, 0.06];
waterDark = [0.05, 0.32, 0.60];
legendHandles = gobjects(4, 1);
legendAxis = gobjects(1);

for c = 1:numel(caseIds)
    rr = loadReservoirReady(records{c}.matFile);
    for w = 1:6
        ax = nexttile(layout);
        hold(ax, 'on');
        selectedRow = selectedSwiMedoidReplayRow(rr, w);
        selectedSlice = NaN;
        for s = 1:87
            curve = rr.krCurves{w, s};
            plot(ax, curve.gasSaturation, curve.krg, ...
                '-', 'Color', gasLight, 'LineWidth', 0.5);
            plot(ax, curve.gasSaturation, curve.krw, ...
                '-', 'Color', waterLight, 'LineWidth', 0.5);
            if double(rr.pcCurves{w, s}.replaySourceRow) == selectedRow
                selectedSlice = s;
            end
        end
        assert(isfinite(selectedSlice), ...
            'QualificationPlots:MissingSelectedKr', ...
            'Swi-medoid Kr slice was not found for %s case %02d W%d.', ...
            geologyId, caseIds(c), w);
        selected = rr.krCurves{w, selectedSlice};
        plot(ax, selected.gasSaturation, selected.krg, ...
            '-', 'Color', gasDark, 'LineWidth', 2.7);
        plot(ax, selected.gasSaturation, selected.krw, ...
            '-', 'Color', waterDark, 'LineWidth', 2.7);
        if c == 1 && w == 1
            legendAxis = ax;
            legendHandles(1) = plot(ax, NaN, NaN, '-', ...
                'Color', gasLight, 'LineWidth', 1.2);
            legendHandles(2) = plot(ax, NaN, NaN, '-', ...
                'Color', waterLight, 'LineWidth', 1.2);
            legendHandles(3) = plot(ax, NaN, NaN, '-', ...
                'Color', gasDark, 'LineWidth', 2.7);
            legendHandles(4) = plot(ax, NaN, NaN, '-', ...
                'Color', waterDark, 'LineWidth', 2.7);
        end
        xlim(ax, [0, 1]);
        ylim(ax, [0, 1]);
        formatCurveAxes(ax, c, w, caseIds, "Relative permeability");
    end
end

title(layout, sprintf( ...
    '%s: Pc-guided dynamic relative-permeability curves', ...
    scenarioLabel(geologyId)), ...
    'FontSize', 25, 'FontWeight', 'bold');
subtitle(layout, ['Light = 87 endpoint-scaled slices; dark = selected ', ...
    'Swi-medoid representative'], 'FontSize', 15);
lgd = legend(legendAxis, legendHandles, ...
    {'87 slice Krg', '87 slice Krw', ...
    'selected Krg', 'selected Krw'}, ...
    'Location', 'northwest', 'NumColumns', 2);
lgd.FontSize = 10;
lgd.Box = 'on';

file = fullfile(outputDir, geologyId + "_kr_curve_atlas.png");
exportgraphics(fig, file, 'Resolution', 240);
close(fig);
end


function rr = loadReservoirReady(matFile)
% Load a reservoirReady structure from one production MAT.

loaded = load(matFile, 'reservoirReady');
rr = loaded.reservoirReady;
end


function selectedRow = selectedSwiMedoidReplayRow(rr, windowIndex)
% Read the replay-source row selected by scalar Swi medoid matching.

windowLabel = string(rr.windowLabels(windowIndex));
selection = rr.swiMedoidSelection;
mask = string(selection.Window) == windowLabel;
assert(nnz(mask) == 1, 'QualificationPlots:SelectionCount', ...
    'Expected one Swi-medoid selection for %s.', windowLabel);
selectedRow = double(selection.SelectedReplaySourceRow(mask));
end


function values = collectProperty(records, key)
% Concatenate one property across every qualification record.

values = [];
for i = 1:numel(records)
    current = extractProperty(records{i}, key);
    values = [values; current(:)]; %#ok<AGROW>
end
values = values(isfinite(values));
end


function values = extractProperty(record, key)
% Return one window-by-slice property array.

switch string(key)
    case "log_kxx"
        values = record.logK(:, :, 1);
    case "log_kyy"
        values = record.logK(:, :, 2);
    case "log_kzz"
        values = record.logK(:, :, 3);
    case "porosity"
        values = record.porosity;
    case "entry_pressure"
        values = record.entryPressure;
    case "log_entry_pressure"
        values = log10(record.entryPressure);
    case "effective_swi"
        values = record.effectiveSwi;
    otherwise
        error('QualificationPlots:UnknownProperty', ...
            'Unknown property key: %s', key);
end
end


function formatSummaryAxes(ax)
% Apply one publication-style format to distribution-summary panels.

grid(ax, 'on');
ax.GridColor = [0.82, 0.82, 0.82];
ax.GridAlpha = 0.55;
ax.FontName = 'Arial';
ax.FontSize = 14;
ax.LineWidth = 1.0;
ax.TickDir = 'out';
box(ax, 'on');
end


function formatSpatialAxes(ax, geologyIndex, caseIndex, nCases)
% Format the compact 6-by-87 maps without redundant labels.

xticks(ax, [1, 29, 58, 87]);
yticks(ax, 1:6);
if caseIndex == nCases
    xticklabels(ax, {'1', '29', '58', '87'});
    xlabel(ax, 'Along-strike slice', 'FontSize', 13);
else
    ax.XTickLabel = [];
end
if geologyIndex == 1
    yticklabels(ax, compose('W%d', 1:6));
else
    ax.YTickLabel = [];
end
ax.FontName = 'Arial';
ax.FontSize = 12;
ax.LineWidth = 0.9;
ax.TickDir = 'out';
ax.Layer = 'top';
box(ax, 'on');
end


function formatCurveAxes(ax, caseIndex, windowIndex, caseIds, yLabelText)
% Format one panel in a four-case by six-window curve atlas.

grid(ax, 'on');
ax.GridColor = [0.82, 0.82, 0.82];
ax.GridAlpha = 0.50;
ax.MinorGridAlpha = 0.15;
ax.FontName = 'Arial';
ax.FontSize = 12;
ax.LineWidth = 0.9;
box(ax, 'on');
if caseIndex == 1
    title(ax, sprintf('W%d', windowIndex), ...
        'FontSize', 16, 'FontWeight', 'bold');
end
if caseIndex == numel(caseIds)
    xlabel(ax, 'Gas saturation', 'FontSize', 14);
else
    ax.XTickLabel = [];
end
if windowIndex == 1
    ylabel(ax, {compose("C%02d", caseIds(caseIndex)); yLabelText}, ...
        'FontSize', 13, 'FontWeight', 'bold');
else
    ax.YTickLabel = [];
end
end


function colors = caseColors()
% Colorblind-friendly case colors: independent, low, high, grouped.

colors = [ ...
    0.30, 0.30, 0.30; ...
    0.13, 0.45, 0.70; ...
    0.84, 0.37, 0.10; ...
    0.00, 0.62, 0.45];
end


function label = recordCaseLabel(record)
% Return the actual geology-specific Level 3 case name.

caseName = replace(string(record.caseName), "_", " ");
label = compose("Case %02d: %s", record.caseId, caseName);
end


function label = scenarioLabel(geologyId)
% Map qualification IDs to reader-facing thickness-scenario labels.

switch string(geologyId)
    case "s01_c012"
        label = "Low sand, uniform";
    case "s02_c012"
        label = "Medium sand, uniform";
    case "s03_c012"
        label = "High sand, uniform";
    case "s04_c012"
        label = "Low sand, nonuniform";
    case "s05_c012"
        label = "Medium sand, nonuniform";
    case "s06_c012"
        label = "High sand, nonuniform";
    otherwise
        label = string(geologyId);
end
end


function labels = scenarioTickLabels()
% Compact labels for the six qualification thickness scenarios.

labels = ["Low U", "Medium U", "High U", ...
    "Low NU", "Medium NU", "High NU"];
end


function q = percentile(values, probability)
% Compute one linearly interpolated percentile without toolbox dependency.

values = sort(double(values(:)));
assert(~isempty(values) && probability >= 0 && probability <= 1, ...
    'QualificationPlots:InvalidPercentile', ...
    'Percentiles require finite values and probability in [0,1].');
position = 1 + (numel(values) - 1) * probability;
lowerIndex = floor(position);
upperIndex = ceil(position);
weight = position - lowerIndex;
q = (1 - weight) * values(lowerIndex) + weight * values(upperIndex);
end


function limits = roundedColorLimits(values, step)
% Round full observed limits outward to clean colorbar endpoints.

values = double(values(:));
values = values(isfinite(values));
assert(~isempty(values) && isfinite(step) && step > 0, ...
    'QualificationPlots:InvalidColorLimits', ...
    'Color limits require finite values and a positive rounding step.');
limits = [floor(min(values) / step), ceil(max(values) / step)] * step;
if limits(1) == limits(2)
    limits = limits + [-step, step];
end
end


function limits = paddedLimits(values, fraction)
% Create finite increasing limits with a small visual margin.

values = double(values(:));
values = values(isfinite(values));
assert(~isempty(values), 'QualificationPlots:EmptyLimits', ...
    'Cannot define color limits from an empty property.');
limits = [min(values), max(values)];
span = diff(limits);
if span <= eps(max(abs(limits)))
    span = max(1.0e-3, 0.1 * max(abs(limits)));
end
limits = limits + [-1, 1] * fraction * span;
end


function cmap = qualificationMap(n)
% Sequential blue-to-gold map with strong low/high contrast.

anchors = [ ...
    0.05, 0.16, 0.36; ...
    0.08, 0.36, 0.61; ...
    0.20, 0.60, 0.72; ...
    0.70, 0.80, 0.62; ...
    0.93, 0.68, 0.24; ...
    0.72, 0.25, 0.08];
x = linspace(0, 1, size(anchors, 1));
xq = linspace(0, 1, n);
cmap = interp1(x, anchors, xq, 'pchip');
cmap = min(max(cmap, 0), 1);
end


function ensureFolder(folderPath)
% Create the output directory when it does not already exist.

if ~isfolder(folderPath)
    mkdir(folderPath);
end
end
