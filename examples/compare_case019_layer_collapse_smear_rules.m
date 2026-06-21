function results = compare_case019_layer_collapse_smear_rules(outputDir, varargin)
%COMPARE_CASE019_LAYER_COLLAPSE_SMEAR_RULES Compare layer-collapse and smear rules.
%
% This focused diagnostic tests case019 in the medium-sand nonuniform
% thickness scenario:
%   faulting depth = 1000 m, sand Vcl = 0.1, clay Vcl = 0.4.
%
% Four setups are compared with matched random seeds:
%   1. legacy_thin:          legacy random overlap, original thin S/C layers
%   2. legacy_collapsed:     legacy random overlap, adjacent S/C layers merged
%   3. cell_union_thin:      cell_union_psmear, original thin S/C layers
%   4. cell_union_collapsed: cell_union_psmear, adjacent S/C layers merged
%
% Outputs include summary tables, standard marginal-distribution figures,
% and representative fine-scale maps for selected windows.

if nargin < 1 || isempty(outputDir)
    outputDir = fullfile('D:', 'codex_gom', ...
        'case019_layer_collapse_smear_rule_comparison');
end

parser = inputParser;
parser.addParameter('Nsim', 20, @(x) isnumeric(x) && isscalar(x) && x > 0);
parser.addParameter('Windows', {'famp1','famp2','famp3','famp4','famp5','famp6'}, ...
    @(x) iscell(x) || isstring(x));
parser.addParameter('MapWindows', {'famp2','famp5','famp6'}, ...
    @(x) iscell(x) || isstring(x));
parser.addParameter('BaseSeed', 271828, @(x) isnumeric(x) && isscalar(x));
parser.addParameter('CorrCoef', 0, @(x) isnumeric(x) && isscalar(x));
parser.addParameter('FigureDpi', 220, @(x) isnumeric(x) && isscalar(x));
parser.addParameter('ShowProgress', true, @(x) islogical(x) || isnumeric(x));
parser.parse(varargin{:});
opt = parser.Results;
opt.Windows = cellstr(string(opt.Windows));
opt.MapWindows = cellstr(string(opt.MapWindows));
opt.Nsim = round(opt.Nsim);
opt.ShowProgress = logical(opt.ShowProgress);

setupPredictPaths();
assert(exist('mrstModule', 'file') == 2, ...
    ['MRST is not on the MATLAB path. Run startup.m from MRST before ' ...
     'calling compare_case019_layer_collapse_smear_rules.'])
mrstModule add mrst-gui coarsegrid upscaling incomp mpfa mimetic
mrstVerbose off

dirs = makeOutputDirs(outputDir);
caseInfo = makeCase019Info();
scenarioRows = buildScenario05MediumNonuniformRows();
setups = buildComparisonSetups();
U = makeUpscalingOptions();

summaryRows = {};
metadataRows = {};
results = struct();
results.OutputDir = outputDir;
results.Options = opt;
results.CaseInfo = caseInfo;
results.ScenarioRows = scenarioRows;
results.Setups = setups;
results.WindowResults = struct();

for isetup = 1:numel(setups)
    setup = setups(isetup);
    if opt.ShowProgress
        fprintf('\n=== Setup %d/%d: %s ===\n', isetup, numel(setups), setup.Label);
    end

    for iw = 1:numel(opt.Windows)
        window = opt.Windows{iw};
        if opt.ShowProgress
            fprintf('  %s: %s\n', window, setup.Description);
        end

        baseWindowOpt = getWindowOptions(window);
        scenarioWindow = getScenarioWindowRow(scenarioRows, window);
        windowOpt = applyCaseToWindow(baseWindowOpt, scenarioWindow, caseInfo);
        if setup.CollapseAdjacent
            windowOpt = collapseAdjacentWindowLayers(windowOpt, caseInfo);
        end

        mySect = buildFaultedSection(windowOpt);
        seedBase = opt.BaseSeed + 100000*parseWindowId(window);
        [perms, meta, replay] = runWindowSamplesWithReplay( ...
            mySect, windowOpt, opt.Nsim, opt.CorrCoef, U, seedBase, ...
            setup.SmearOverlapRule, opt.ShowProgress);

        setupDir = fullfile(dirs.Data, setup.Label, window);
        ensureFolder(setupDir);
        save(fullfile(setupDir, 'predict_runs.mat'), ...
            'perms', 'meta', 'replay', 'windowOpt', 'setup', 'caseInfo', '-v7.3');

        resultKey = matlab.lang.makeValidName([setup.Label '_' window]);
        results.WindowResults.(resultKey) = struct( ...
            'Setup', setup, 'Window', window, 'Perms', perms, ...
            'Meta', meta, 'Replay', replay, 'WindowOpt', windowOpt);

        summaryRows = [summaryRows; summarizePerms(setup, window, perms)]; %#ok<AGROW>
        metadataRows = [metadataRows; makeMetadataRow(setup, window, windowOpt, meta)]; %#ok<AGROW>
    end
end

summaryTable = cell2table(summaryRows, 'VariableNames', summaryVariableNames());
metadataTable = cell2table(metadataRows, 'VariableNames', metadataVariableNames());
writetable(summaryTable, fullfile(dirs.Tables, 'case019_four_setup_perm_summary.csv'));
writetable(metadataTable, fullfile(dirs.Tables, 'case019_four_setup_metadata.csv'));

plotMarginalOverlay(results, dirs.Figures, opt);
for isetup = 1:numel(setups)
    plotSetupMarginalHistograms(results, setups(isetup), dirs.Figures, opt);
end
plotMedianDifferenceHeatmap(summaryTable, dirs.Figures, opt);
for iw = 1:numel(opt.MapWindows)
    plotFineMapComparison(results, opt.MapWindows{iw}, dirs.FineMaps, opt);
end

save(fullfile(outputDir, 'case019_layer_collapse_smear_rule_comparison.mat'), ...
    'results', 'summaryTable', 'metadataTable', '-v7.3');

if opt.ShowProgress
    fprintf('\nSaved comparison outputs to:\n  %s\n', outputDir);
end
end


function caseInfo = makeCase019Info()
caseInfo = struct();
caseInfo.CaseIndex = 19;
caseInfo.CaseLabel = 'case_019_zf1000_svcl010_cvcl040';
caseInfo.FaultingDepth = 1000;
caseInfo.SandVcl = 0.1;
caseInfo.ClayVcl = 0.4;
caseInfo.IsClayVcl = 0.35;
caseInfo.ScenarioIndex = 5;
caseInfo.ScenarioLabel = 'scenario_05_medium_sand_nonuniform';
caseInfo.ScenarioName = 'medium sand, nonuniform';
end


function setups = buildComparisonSetups()
setups = repmat(struct( ...
    'Label', '', 'ShortLabel', '', 'Description', '', ...
    'SmearOverlapRule', '', 'CollapseAdjacent', false), 1, 4);

setups(1).Label = 'legacy_thin';
setups(1).ShortLabel = 'Legacy thin';
setups(1).Description = 'Legacy random overlap rule with original thin S/C layers.';
setups(1).SmearOverlapRule = 'random';
setups(1).CollapseAdjacent = false;

setups(2).Label = 'legacy_collapsed';
setups(2).ShortLabel = 'Legacy collapsed';
setups(2).Description = 'Legacy random overlap rule with adjacent equal lithologies collapsed.';
setups(2).SmearOverlapRule = 'random';
setups(2).CollapseAdjacent = true;

setups(3).Label = 'cell_union_thin';
setups(3).ShortLabel = 'Cell-union thin';
setups(3).Description = 'Cell-union Psmear rule with original thin S/C layers.';
setups(3).SmearOverlapRule = 'cell_union_psmear';
setups(3).CollapseAdjacent = false;

setups(4).Label = 'cell_union_collapsed';
setups(4).ShortLabel = 'Cell-union collapsed';
setups(4).Description = 'Cell-union Psmear rule with adjacent equal lithologies collapsed.';
setups(4).SmearOverlapRule = 'cell_union_psmear';
setups(4).CollapseAdjacent = true;
end


function scenarioRows = buildScenario05MediumNonuniformRows()
scenarioIndex = repmat(5, 6, 1);
scenarioLabel = repmat("scenario_05_medium_sand_nonuniform", 6, 1);
scenarioName = repmat("medium sand, nonuniform", 6, 1);
window = ["famp1"; "famp2"; "famp3"; "famp4"; "famp5"; "famp6"];
fwPattern = ["SC"; "SSSC"; "SCCC"; "SSCC"; "SCCC"; "SSSC"];
hwPattern = ["SSSC"; "SCCC"; "SSCC"; "SCCC"; "SSSC"; "S"];
scenarioRows = table(scenarioIndex, scenarioLabel, scenarioName, window, ...
    fwPattern, hwPattern, 'VariableNames', ...
    {'ScenarioIndex', 'ScenarioLabel', 'ScenarioName', 'Window', ...
     'FWPattern', 'HWPattern'});
end


function dirs = makeOutputDirs(outputDir)
dirs = struct();
dirs.Root = outputDir;
dirs.Data = fullfile(outputDir, 'data');
dirs.Tables = fullfile(outputDir, 'tables');
dirs.Figures = fullfile(outputDir, 'figures');
dirs.FineMaps = fullfile(dirs.Figures, 'fine_maps');
ensureFolder(outputDir);
ensureFolder(dirs.Data);
ensureFolder(dirs.Tables);
ensureFolder(dirs.Figures);
ensureFolder(dirs.FineMaps);
end


function scenarioWindow = getScenarioWindowRow(scenarioRows, window)
match = strcmpi(string(scenarioRows.Window), string(window));
assert(nnz(match) == 1, 'Expected exactly one scenario row for %s.', window)
scenarioWindow = scenarioRows(match, :);
end


function windowOpt = applyCaseToWindow(baseWindowOpt, scenarioWindow, caseInfo)
windowOpt = baseWindowOpt;
windowOpt.zf = [caseInfo.FaultingDepth, caseInfo.FaultingDepth];
windowOpt.pattern = {char(scenarioWindow.FWPattern), char(scenarioWindow.HWPattern)};
windowOpt.vcl = { ...
    patternToVcl(scenarioWindow.FWPattern, caseInfo.SandVcl, caseInfo.ClayVcl), ...
    patternToVcl(scenarioWindow.HWPattern, caseInfo.SandVcl, caseInfo.ClayVcl)};
end


function vcl = patternToVcl(pattern, sandVcl, clayVcl)
pattern = upper(char(pattern));
vcl = nan(1, numel(pattern));
vcl(pattern == 'S') = sandVcl;
vcl(pattern == 'C') = clayVcl;
assert(all(isfinite(vcl)), 'Pattern must contain only S and C.')
end


function windowOpt = collapseAdjacentWindowLayers(windowOpt, caseInfo)
for side = 1:2
    [windowOpt.thick{side}, windowOpt.vcl{side}, windowOpt.zmax{side}, ...
     windowOpt.pattern{side}] = collapseAdjacentLayers( ...
        windowOpt.thick{side}, windowOpt.vcl{side}, windowOpt.zmax{side}, ...
        caseInfo.SandVcl, caseInfo.ClayVcl);
end
end


function [thickOut, vclOut, zmaxOut, patternOut] = collapseAdjacentLayers(thick, vcl, zmax, sandVcl, clayVcl)
thick = asRow(thick);
vcl = asRow(vcl);
zmax = asRow(zmax);
assert(numel(thick) == numel(vcl) && numel(thick) == numel(zmax), ...
    'Layer thickness, Vcl, and zmax must have matching lengths.')

isClay = vcl >= 0.35;
if isempty(thick)
    thickOut = thick;
    vclOut = vcl;
    zmaxOut = zmax;
    patternOut = '';
    return
end

groups = [1, find(diff(isClay) ~= 0) + 1, numel(thick) + 1];
thickOut = zeros(1, numel(groups)-1);
vclOut = zeros(1, numel(groups)-1);
zmaxOut = zeros(1, numel(groups)-1);
patternChars = repmat('S', 1, numel(groups)-1);

for g = 1:(numel(groups)-1)
    ids = groups(g):(groups(g+1)-1);
    thickOut(g) = sum(thick(ids));
    zmaxOut(g) = weightedMean(zmax(ids), thick(ids));
    if isClay(ids(1))
        vclOut(g) = clayVcl;
        patternChars(g) = 'C';
    else
        vclOut(g) = sandVcl;
        patternChars(g) = 'S';
    end
end
patternOut = patternChars;
end


function value = weightedMean(values, weights)
value = sum(values(:) .* weights(:)) ./ sum(weights(:));
end


function x = asRow(x)
x = x(:).';
end


function mySect = buildFaultedSection(windowOpt)
footwall = Stratigraphy(windowOpt.thick{1}, windowOpt.vcl{1}, ...
    'Dip', windowOpt.dip(1), ...
    'DepthFaulting', windowOpt.zf(1), ...
    'DepthBurial', windowOpt.zmax{1});
hangingwall = Stratigraphy(windowOpt.thick{2}, windowOpt.vcl{2}, ...
    'Dip', windowOpt.dip(2), ...
    'IsHW', 1, ...
    'NumLayersFW', footwall.NumLayers, ...
    'DepthFaulting', windowOpt.zf(2), ...
    'DepthBurial', windowOpt.zmax{2});

if isfield(windowOpt, 'totThick')
    mySect = FaultedSection(footwall, hangingwall, windowOpt.fDip, ...
        'maxPerm', windowOpt.maxPerm, 'totThick', windowOpt.totThick);
else
    mySect = FaultedSection(footwall, hangingwall, windowOpt.fDip, ...
        'maxPerm', windowOpt.maxPerm);
end
mySect = mySect.getMatPropDistr();
end


function [perms, meta, firstReplay] = runWindowSamplesWithReplay(mySect, windowOpt, targetN, rho, U, seedBase, smearOverlapRule, showProgress)
perms = nan(targetN, 3);
numValid = 0;
numAttempts = 0;
numRejected = 0;
firstReplay = struct();
firstReplayCaptured = false;

while numValid < targetN
    numAttempts = numAttempts + 1;
    rng(seedBase + numAttempts - 1, 'twister');

    if ~firstReplayCaptured
        [replay, ok, errMsg] = runDetailedRealization(mySect, windowOpt, rho, U, smearOverlapRule);
        if ok
            numValid = numValid + 1;
            perms(numValid, :) = replay.PermMD;
            replay.AttemptIndex = numAttempts;
            replay.Seed = seedBase + numAttempts - 1;
            replay.ErrorMessage = errMsg;
            firstReplay = replay;
            firstReplayCaptured = true;
        else
            numRejected = numRejected + 1;
        end
    else
        perm = runSingleWindowPermRealization(mySect, windowOpt, rho, U, smearOverlapRule);
        if all(isfinite(perm)) && all(perm > 0)
            numValid = numValid + 1;
            perms(numValid, :) = perm;
        else
            numRejected = numRejected + 1;
        end
    end

    if numAttempts > 20*targetN
        error('Too many invalid realizations after %d attempts.', numAttempts)
    end
end

if showProgress
    fprintf('    valid %d / %d, attempts %d, rejected %d\n', ...
        numValid, targetN, numAttempts, numRejected);
end

meta = struct();
meta.TargetN = targetN;
meta.NumReturned = size(perms, 1);
meta.NumAttempts = numAttempts;
meta.NumRejected = numRejected;
meta.AcceptanceRatio = targetN / numAttempts;
meta.SeedBase = seedBase;
meta.SmearOverlapRule = char(smearOverlapRule);
meta.FirstReplaySeed = firstReplay.Seed;
meta.CompletedOn = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
end


function [replay, ok, errMsg] = runDetailedRealization(mySect, windowOpt, rho, U, smearOverlapRule)
replay = struct();
ok = false;
errMsg = "";

try
    nSeg = getNSeg(mySect.Vcl, mySect.IsClayVcl, mySect.DepthFaulting);
    myFaultSection = Fault2D(mySect, windowOpt.fDip);
    myFault = Fault3D(myFaultSection, mySect);

    if U.flexible
        [myFault, Urun] = myFault.getSegmentationLength(U, nSeg.fcn);
    else
        myFault = myFault.getSegmentationLength(U, nSeg.fcn);
        Urun = U;
    end

    G = [];
    sectionDetails = repmat(struct( ...
        'MatProps', [], 'MatMap', [], 'Grid', [], 'Poro', [], ...
        'PermMD', [], 'Vcl', [], 'Smear', []), 1, numel(myFault.SegLen));

    for k = 1:numel(myFault.SegLen)
        myFaultSection = myFaultSection.getMaterialProperties(mySect, ...
            'corrCoef', rho);
        myFaultSection.MatProps.thick = myFault.Thick;
        if isempty(G)
            G = makeFaultGrid(myFault.Thick, myFault.Disp, ...
                myFault.Length, myFault.SegLen, Urun);
        end

        smear = Smear(mySect, myFaultSection, G, 1);
        myFaultSection = myFaultSection.placeMaterials( ...
            mySect, smear, G, 'SmearOverlapRule', smearOverlapRule);

        sectionDetails(k).MatProps = myFaultSection.MatProps;
        sectionDetails(k).MatMap = myFaultSection.MatMap;
        sectionDetails(k).Grid = myFaultSection.Grid;
        sectionDetails(k).Poro = mean(myFaultSection.Grid.poro);
        sectionDetails(k).PermMD = myFaultSection.Perm ./ (milli*darcy);
        sectionDetails(k).Vcl = mean(myFaultSection.Grid.vcl);
        sectionDetails(k).Smear = smear;

        myFault = myFault.assignExtrudedVals(G, myFaultSection, k);
    end

    [myFault, CG, Urun] = myFault.upscaleProps(G, Urun);
    replay.PermMD = myFault.Perm ./ (milli*darcy);
    replay.Log10PermMD = log10(replay.PermMD);
    replay.Poro = myFault.Poro;
    replay.Vcl = myFault.Vcl;
    replay.Thick = myFault.Thick;
    replay.SegLen = myFault.SegLen;
    replay.Dip = myFault.Dip;
    replay.Disp = myFault.Disp;
    replay.Length = myFault.Length;
    replay.Grid = myFault.Grid;
    replay.G = G;
    replay.CG = CG;
    replay.U = Urun;
    replay.SectionDetails = sectionDetails;
    ok = all(isfinite(replay.PermMD)) && all(replay.PermMD > 0);
catch ME
    errMsg = string(ME.message);
    replay.ErrorIdentifier = string(ME.identifier);
    replay.ErrorMessage = errMsg;
end
end


function perm = runSingleWindowPermRealization(mySect, windowOpt, rho, U, smearOverlapRule)
perm = nan(1, 3);
try
    nSeg = getNSeg(mySect.Vcl, mySect.IsClayVcl, mySect.DepthFaulting);
    myFaultSection = Fault2D(mySect, windowOpt.fDip);
    myFault = Fault3D(myFaultSection, mySect);

    if U.flexible
        [myFault, Urun] = myFault.getSegmentationLength(U, nSeg.fcn);
    else
        myFault = myFault.getSegmentationLength(U, nSeg.fcn);
        Urun = U;
    end

    G = [];
    for k = 1:numel(myFault.SegLen)
        myFaultSection = myFaultSection.getMaterialProperties(mySect, ...
            'corrCoef', rho);
        myFaultSection.MatProps.thick = myFault.Thick;
        if isempty(G)
            G = makeFaultGrid(myFault.Thick, myFault.Disp, ...
                myFault.Length, myFault.SegLen, Urun);
        end

        smear = Smear(mySect, myFaultSection, G, 1);
        myFaultSection = myFaultSection.placeMaterials( ...
            mySect, smear, G, 'SmearOverlapRule', smearOverlapRule);
        myFault = myFault.assignExtrudedVals(G, myFaultSection, k);
    end

    [myFault, ~] = myFault.upscaleProps(G, Urun);
    perm = myFault.Perm ./ (milli*darcy);
catch
    perm(:) = NaN;
end
end


function rows = summarizePerms(setup, window, perms)
components = {'kxx', 'kyy', 'kzz'};
logPerms = log10(perms);
rows = cell(numel(components), numel(summaryVariableNames()));
for j = 1:numel(components)
    values = logPerms(:, j);
    qs = quantile(values, [0.05 0.25 0.5 0.75 0.95]);
    rows(j, :) = {setup.Label, setup.ShortLabel, char(window), components{j}, ...
        numel(values), mean(values), std(values), qs(1), qs(2), qs(3), ...
        qs(4), qs(5), qs(5)-qs(1), mean(values < -3), mean(values < -1), ...
        mean(values > 0)};
end
end


function names = summaryVariableNames()
names = {'Setup', 'SetupShortLabel', 'Window', 'Component', 'N', ...
    'MeanLog10K', 'StdLog10K', 'Q05Log10K', 'Q25Log10K', ...
    'MedianLog10K', 'Q75Log10K', 'Q95Log10K', 'Spread90Log10K', ...
    'FracBelowMinus3', 'FracBelowMinus1', 'FracAbove0'};
end


function row = makeMetadataRow(setup, window, windowOpt, meta)
row = {setup.Label, setup.ShortLabel, char(window), setup.SmearOverlapRule, ...
    logical(setup.CollapseAdjacent), char(windowOpt.pattern{1}), ...
    char(windowOpt.pattern{2}), layerVectorString(windowOpt.thick{1}), ...
    layerVectorString(windowOpt.thick{2}), layerVectorString(windowOpt.zmax{1}), ...
    layerVectorString(windowOpt.zmax{2}), meta.TargetN, meta.NumAttempts, ...
    meta.NumRejected, meta.SeedBase, meta.FirstReplaySeed};
end


function names = metadataVariableNames()
names = {'Setup', 'SetupShortLabel', 'Window', 'SmearOverlapRule', ...
    'CollapseAdjacent', 'FWPattern', 'HWPattern', 'FWThickness', ...
    'HWThickness', 'FWZmax', 'HWZmax', 'TargetN', 'NumAttempts', ...
    'NumRejected', 'SeedBase', 'FirstReplaySeed'};
end


function s = layerVectorString(x)
s = strjoin(compose('%.6g', asRow(x)), ' ');
end


function plotMarginalOverlay(results, figDir, opt)
setups = results.Setups;
windows = opt.Windows;
components = {'kxx', 'kyy', 'kzz'};
colors = [0.15 0.35 0.62; 0.85 0.45 0.10; 0.42 0.18 0.55; 0.10 0.55 0.30];

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [60 60 2100 1180]);
tiledlayout(3, numel(windows), 'Padding', 'compact', 'TileSpacing', 'compact');
binEdges = -7:0.25:2;

for ic = 1:3
    for iw = 1:numel(windows)
        ax = nexttile((ic-1)*numel(windows) + iw);
        hold(ax, 'on')
        for isetup = 1:numel(setups)
            perms = getPerms(results, setups(isetup).Label, windows{iw});
            vals = log10(perms(:, ic));
            h = histogram(ax, vals, binEdges, 'Normalization', 'probability', ...
                'DisplayStyle', 'stairs', 'LineWidth', 1.8, ...
                'EdgeColor', colors(isetup, :), ...
                'DisplayName', setups(isetup).ShortLabel);
            if isempty(h.Values)
                continue
            end
            med = median(vals);
            plot(ax, [med med], [0 max([h.Values, 0.01])], '-', ...
                'Color', colors(isetup, :), 'LineWidth', 1.0, ...
                'HandleVisibility', 'off');
        end
        hold(ax, 'off')
        grid(ax, 'on')
        xlim(ax, [-7 2])
        xticks(ax, [-7 -4 -1 2])
        set(ax, 'FontSize', 15)
        if iw == 1
            ylabel(ax, sprintf('log10(%s) probability', components{ic}));
        end
        if ic == 1
            title(ax, upper(strrep(windows{iw}, 'famp', 'W')), ...
                'FontWeight', 'bold');
        end
        if ic == 3
            xlabel(ax, 'log10(k) [mD]');
        end
    end
end

lg = legend('Location', 'southoutside', 'Orientation', 'horizontal', ...
    'FontSize', 15);
lg.Layout.Tile = 'south';
sgtitle(['case019 medium sand nonuniform: four setup marginal comparison ' ...
    '(matched seeds)'], 'FontSize', 24, 'FontWeight', 'bold');
exportgraphics(fig, fullfile(figDir, 'case019_four_setup_marginal_overlay.png'), ...
    'Resolution', opt.FigureDpi);
exportgraphics(fig, fullfile(figDir, 'case019_four_setup_marginal_overlay.pdf'), ...
    'ContentType', 'vector');
close(fig);
end


function plotSetupMarginalHistograms(results, setup, figDir, opt)
windows = opt.Windows;
components = {'kxx', 'kyy', 'kzz'};
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [60 60 2050 1160]);
tiledlayout(3, numel(windows), 'Padding', 'compact', 'TileSpacing', 'compact');
binEdges = -7:0.25:2;
letters = char('a' + (0:(3*numel(windows)-1)));

for ic = 1:3
    for iw = 1:numel(windows)
        ax = nexttile((ic-1)*numel(windows) + iw);
        perms = getPerms(results, setup.Label, windows{iw});
        vals = log10(perms(:, ic));
        histogram(ax, vals, binEdges, 'Normalization', 'probability', ...
            'FaceColor', [0.45 0.49 0.52], 'EdgeColor', 'none');
        hold(ax, 'on')
        med = median(vals);
        q25 = quantile(vals, 0.25);
        q75 = quantile(vals, 0.75);
        yLimits = ylim(ax);
        yTop = max(yLimits(2), 0.01);
        plot(ax, [med med], [0 yTop], 'r-', 'LineWidth', 1.2);
        plot(ax, [q25 q25], [0 yTop], '--', 'Color', [0.55 0.55 0.55], 'LineWidth', 1.0);
        plot(ax, [q75 q75], [0 yTop], '--', 'Color', [0.55 0.55 0.55], 'LineWidth', 1.0);
        hold(ax, 'off')
        grid(ax, 'on')
        xlim(ax, [-7 2])
        xticks(ax, [-7 -4 -1 2])
        set(ax, 'FontSize', 15)
        text(ax, 0.94, 0.90, sprintf('(%s)', letters((ic-1)*numel(windows)+iw)), ...
            'Units', 'normalized', 'HorizontalAlignment', 'right', ...
            'FontWeight', 'bold', 'FontSize', 14)
        if iw == 1
            ylabel(ax, 'Probability');
        end
        if ic == 1
            title(ax, upper(strrep(windows{iw}, 'famp', 'W')), ...
                'FontWeight', 'bold');
        end
        if ic == 3
            xlabel(ax, sprintf('log10(%s) [mD]', components{ic}));
        elseif iw == ceil(numel(windows)/2)
            xlabel(ax, sprintf('log10(%s) [mD]', components{ic}));
        end
    end
end

sgtitle(sprintf('case019 | %s', setup.ShortLabel), ...
    'FontSize', 24, 'FontWeight', 'bold');
annotation(fig, 'textbox', [0.34 0.012 0.38 0.03], ...
    'String', 'Gray bars = empirical bin probability; red = median; dashed gray = 25th and 75th percentiles.', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 14);
fileBase = sprintf('case019_%s_marginal_histograms', setup.Label);
exportgraphics(fig, fullfile(figDir, [fileBase '.png']), 'Resolution', opt.FigureDpi);
exportgraphics(fig, fullfile(figDir, [fileBase '.pdf']), 'ContentType', 'vector');
close(fig);
end


function plotMedianDifferenceHeatmap(summaryTable, figDir, opt)
windows = opt.Windows;
components = {'kxx', 'kyy', 'kzz'};
setups = {'legacy_thin', 'legacy_collapsed', 'cell_union_thin', 'cell_union_collapsed'};
setupLabels = {'Legacy thin', 'Legacy collapsed', 'Cell-union thin', 'Cell-union collapsed'};
baseline = 'legacy_thin';

vals = nan(numel(setups)-1, numel(windows)*numel(components));
labels = strings(1, numel(windows)*numel(components));
col = 0;
for iw = 1:numel(windows)
    for ic = 1:numel(components)
        col = col + 1;
        labels(col) = sprintf('%s %s', upper(strrep(windows{iw}, 'famp', 'W')), components{ic});
        baseMask = strcmp(summaryTable.Setup, baseline) & ...
            strcmp(summaryTable.Window, windows{iw}) & ...
            strcmp(summaryTable.Component, components{ic});
        baseValue = summaryTable.MedianLog10K(baseMask);
        for isetup = 2:numel(setups)
            mask = strcmp(summaryTable.Setup, setups{isetup}) & ...
                strcmp(summaryTable.Window, windows{iw}) & ...
                strcmp(summaryTable.Component, components{ic});
            vals(isetup-1, col) = summaryTable.MedianLog10K(mask) - baseValue;
        end
    end
end

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [80 80 2100 540]);
imagesc(vals);
colormap(redblueCmap(256));
mx = max(abs(vals(:)), [], 'omitnan');
if mx == 0 || ~isfinite(mx)
    mx = 1;
end
clim([-mx mx]);
cb = colorbar;
cb.Label.String = '\Delta median log10(k) relative to legacy thin';
set(gca, 'XTick', 1:numel(labels), 'XTickLabel', labels, ...
    'YTick', 1:(numel(setups)-1), 'YTickLabel', setupLabels(2:end), ...
    'FontSize', 13, 'TickLabelInterpreter', 'none')
xtickangle(45)
for r = 1:size(vals, 1)
    for c = 1:size(vals, 2)
        text(c, r, sprintf('%.2f', vals(r, c)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10, ...
            'Color', textColorForValue(vals(r, c), mx));
    end
end
title('Median permeability change from legacy-thin baseline', ...
    'FontSize', 21, 'FontWeight', 'bold')
exportgraphics(fig, fullfile(figDir, 'case019_median_difference_heatmap.png'), ...
    'Resolution', opt.FigureDpi);
exportgraphics(fig, fullfile(figDir, 'case019_median_difference_heatmap.pdf'), ...
    'ContentType', 'vector');
close(fig);
end


function plotFineMapComparison(results, window, mapDir, opt)
setups = results.Setups;
replays = cell(1, numel(setups));
for isetup = 1:numel(setups)
    key = matlab.lang.makeValidName([setups(isetup).Label '_' window]);
    replays{isetup} = results.WindowResults.(key).Replay;
end

segId = chooseCommonMapSegment(replays);
clims = componentColorLimits(replays, segId);
parentIds = parentUnitIdsForComparison(replays, segId);

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [40 40 2300 420*numel(setups)]);
tiledlayout(numel(setups), 5, 'Padding', 'compact', 'TileSpacing', 'compact');

for isetup = 1:numel(setups)
    replay = replays{isetup};
    section = replay.SectionDetails(segId);
    M = section.MatMap;
    units = gridVectorToMap(getGridField(section.Grid, 'units'), M);
    smear = double(M.vals);
    kxx = log10(gridVectorToMap(getPermColumn(section.Grid, 1) ./ (milli*darcy), M));
    kyy = log10(gridVectorToMap(getPermColumn(section.Grid, 2) ./ (milli*darcy), M));
    kzz = log10(gridVectorToMap(getPermColumn(section.Grid, 3) ./ (milli*darcy), M));
    panels = {units, smear, kxx, kyy, kzz};
    names = {'Parent unit', 'Smear map', 'log10(kxx) [mD]', ...
        'log10(kyy) [mD]', 'log10(kzz) [mD]'};

    for j = 1:5
        ax = nexttile((isetup-1)*5 + j);
        imagesc(ax, panels{j});
        axis(ax, 'image')
        set(ax, 'YDir', 'normal', 'FontSize', 12);
        xlabel(ax, 'fault-normal cell');
        if j == 1
            ylabel(ax, sprintf('%s\nfault-dip cell', setups(isetup).ShortLabel), ...
                'FontWeight', 'bold');
        end
        title(ax, names{j}, 'FontSize', 13, 'FontWeight', 'bold');
        if j == 1
            colormap(ax, parentUnitColorMap(parentIds));
            clim(ax, [min(parentIds)-0.5, max(parentIds)+0.5]);
            cb = colorbar(ax);
            cb.Ticks = parentIds;
            cb.TickLabels = compose('%d', parentIds);
            cb.Label.String = 'unit id';
        elseif j == 2
            colormap(ax, [0.86 0.78 0.55; 0.12 0.32 0.55]);
            clim(ax, [0 1]);
            cb = colorbar(ax);
            cb.Ticks = [0 1];
            cb.TickLabels = {'sand', 'smear'};
        else
            colormap(ax, parula(256));
            clim(ax, clims{j-2});
            cb = colorbar(ax);
            cb.Label.String = 'log10(mD)';
        end
    end
end

sgtitle(sprintf('%s case019 sample-1 replay, segment %d: four setup fine-scale maps', ...
    upper(strrep(window, 'famp', 'W')), segId), ...
    'FontSize', 19, 'FontWeight', 'bold');
fileBase = sprintf('case019_%s_four_setup_fine_maps', window);
exportgraphics(fig, fullfile(mapDir, [fileBase '.png']), 'Resolution', opt.FigureDpi);
exportgraphics(fig, fullfile(mapDir, [fileBase '.pdf']), 'ContentType', 'vector');
close(fig);
end


function perms = getPerms(results, setupLabel, window)
key = matlab.lang.makeValidName([setupLabel '_' window]);
perms = results.WindowResults.(key).Perms;
end


function segId = chooseCommonMapSegment(replays)
referenceReplay = replays{end};
seg = arrayfun(@summarizeSection, referenceReplay.SectionDetails);
sandFrac = [seg.SandFrac2D];
target = median(sandFrac, 'omitnan');
[~, segId] = min(abs(sandFrac - target));
maxSeg = min(cellfun(@(x) numel(x.SectionDetails), replays));
segId = min(segId, maxSeg);
end


function s = summarizeSection(section)
M = section.MatMap;
s = struct();
s.SandFrac2D = mean(~M.vals(:));
s.SmearFrac2D = mean(M.vals(:));
end


function ids = parentUnitIdsForComparison(replays, segId)
ids = [];
for i = 1:numel(replays)
    section = replays{i}.SectionDetails(segId);
    M = section.MatMap;
    if isfield(M, 'unitIn')
        ids = [ids, M.unitIn(:)']; %#ok<AGROW>
    else
        ids = [ids, unique(M.units(:))']; %#ok<AGROW>
    end
end
ids = unique(ids(isfinite(ids) & ids > 0));
if isempty(ids)
    ids = 1;
else
    ids = min(ids):max(ids);
end
end


function cmap = parentUnitColorMap(unitIds)
base = [ ...
    0.00 0.45 0.70;
    0.90 0.62 0.00;
    0.00 0.62 0.45;
    0.80 0.47 0.65;
    0.34 0.71 0.91;
    0.84 0.37 0.00;
    0.94 0.89 0.26;
    0.00 0.00 0.00;
    0.55 0.34 0.29;
    0.49 0.18 0.56];
n = max(unitIds) - min(unitIds) + 1;
if n <= size(base, 1)
    cmap = base(1:n, :);
else
    cmap = lines(n);
end
end


function clims = componentColorLimits(replays, segId)
vals = cell(1, 3);
for c = 1:3
    vals{c} = [];
end
for i = 1:numel(replays)
    section = replays{i}.SectionDetails(segId);
    M = section.MatMap;
    vals{1} = [vals{1}; log10(gridVectorToMap(getPermColumn(section.Grid, 1) ./ (milli*darcy), M))]; %#ok<AGROW>
    vals{2} = [vals{2}; log10(gridVectorToMap(getPermColumn(section.Grid, 2) ./ (milli*darcy), M))]; %#ok<AGROW>
    vals{3} = [vals{3}; log10(gridVectorToMap(getPermColumn(section.Grid, 3) ./ (milli*darcy), M))]; %#ok<AGROW>
end

clims = cell(1, 3);
for c = 1:3
    x = vals{c};
    x = x(isfinite(x));
    lo = quantile(x, 0.01);
    hi = quantile(x, 0.99);
    if lo == hi
        lo = lo - 0.5;
        hi = hi + 0.5;
    end
    clims{c} = [lo hi];
end
end


function v = getGridField(Grid, fieldName)
assert(isfield(Grid, fieldName), 'Grid field "%s" is missing.', fieldName)
v = Grid.(fieldName);
end


function v = getPermColumn(Grid, componentIndex)
switch componentIndex
    case 1
        if isfield(Grid, 'permx')
            v = Grid.permx;
        else
            v = Grid.perm(:, 1);
        end
    case 2
        if isfield(Grid, 'permy')
            v = Grid.permy;
        else
            v = Grid.perm(:, 2);
        end
    case 3
        if isfield(Grid, 'permz')
            v = Grid.permz;
        else
            v = Grid.perm(:, 3);
        end
end
end


function A = gridVectorToMap(v, M)
n = size(M.vals, 1);
A = flipud(transpose(reshape(v, n, n)));
end


function color = textColorForValue(v, mx)
if abs(v) > 0.55*mx
    color = [1 1 1];
else
    color = [0.05 0.09 0.18];
end
end


function cmap = redblueCmap(n)
if nargin < 1
    n = 256;
end
x = linspace(-1, 1, n)';
cmap = zeros(n, 3);
cmap(:, 1) = min(1, 1 + x);
cmap(:, 2) = 1 - abs(x)*0.75;
cmap(:, 3) = min(1, 1 - x);
cmap = max(0, min(1, cmap));
end


function U = makeUpscalingOptions()
U.useAcceleration = 1;
U.method = 'tpfa';
U.coarseDims = [1 1 1];
U.flexible = true;
U.exportJutulInputs = false;
end


function setupPredictPaths()
thisFile = mfilename('fullpath');
examplesDir = fileparts(thisFile);
repoRoot = fileparts(examplesDir);
pathsToAdd = {repoRoot, fullfile(repoRoot, 'classes'), ...
    fullfile(repoRoot, 'functions'), fullfile(repoRoot, 'utils'), ...
    fullfile(repoRoot, 'utils', 'mrst-based')};
for i = 1:numel(pathsToAdd)
    if exist(pathsToAdd{i}, 'dir')
        addpath(pathsToAdd{i});
    end
end
end


function ensureFolder(folderPath)
if ~exist(folderPath, 'dir')
    mkdir(folderPath);
end
end


function windowId = parseWindowId(window)
token = regexp(char(window), '\d+', 'match', 'once');
assert(~isempty(token), 'Could not parse window ID from %s.', char(window))
windowId = str2double(token);
end


function opt = getWindowOptions(window)
opt.window = window;
opt.maxPerm = 175;

switch lower(window)
    case 'famp1'
        opt.thick = {[115.6143 28.8949], [37.6113 37.6861 37.6113 31.6005]};
        opt.vcl = {[0.3 0.65], [0.2878 0.65 0.2878 0.65]};
        opt.dip = [0, -12.0136];
        opt.fDip = 41.6345;
        opt.zf = [200, 200];
        opt.zmax = {[1912 1861], [1934 1909 1884 1860]};

    case 'famp2'
        opt.thick = {[36.9255 35.8537 36.8537 36.3111], ...
                     [36.5042 36.5042 36.4314 36.5042]};
        opt.vcl = {[0.2878 0.65 0.2878 0.65], ...
                   [0.2878 0.65 0.2878 0.65]};
        opt.dip = [0, -13.8951];
        opt.fDip = 43.2508;
        opt.zf = [200, 200];
        opt.zmax = {[1837.5 1812.5 1787.5 1762.5], ...
                    [1837.5 1812.5 1787.5 1762.5]};

    case 'famp3'
        opt.thick = {[35.8537 35.8537 35.8537 35.8537], ...
                     [35.8537 35.8537 35.8537 35.8537]};
        opt.vcl = {[0.2878 0.65 0.2878 0.65], ...
                   [0.2878 0.65 0.2878 0.65]};
        opt.dip = [0, -9.7683];
        opt.fDip = 43.8;
        opt.zf = [200, 200];
        opt.zmax = {[1738.8 1713.8 1688.8 1663.8], ...
                    [1738.8 1713.8 1688.8 1663.8]};

    case 'famp4'
        opt.thick = {[35.8537 35.8537 35.8537 35.9255], ...
                     [35.8537 35.8537 35.8537 35.9255]};
        opt.vcl = {[0.2878 0.65 0.2878 0.65], ...
                   [0.2878 0.65 0.2878 0.65]};
        opt.dip = [0, -4.9456];
        opt.fDip = 44.1811;
        opt.zf = [200, 200];
        opt.zmax = {[1638.8 1613.8 1588.8 1563.8], ...
                    [1638.8 1613.8 1588.8 1563.8]};

    case 'famp5'
        opt.thick = {[35.8537 35.8537 35.8537 35.8537], ...
                     [37.4901 35.2847 35.3553 35.2847]};
        opt.vcl = {[0.2878 0.65 0.2878 0.65], ...
                   [0.2878 0.65 0.2878 0.65]};
        opt.dip = [0, -5.2221];
        opt.fDip = 45.0685;
        opt.zf = [200, 200];
        opt.zmax = {[1538.8 1513.82 1488.75 1463.99], ...
                    [1538.8 1513.82 1488.75 1463.99]};

    case 'famp6'
        opt.thick = {[28.2932 33.1042 33.1699 33.1042], 127.6715};
        opt.vcl = {[0.2878 0.65 0.2878 0.65], 0.3266};
        opt.dip = [0, -5];
        opt.fDip = 46.0685;
        opt.zf = [200, 200];
        opt.zmax = {[1440.6 1417.5 1392.5 1367.5], 1400};

    otherwise
        error('Unsupported window "%s".', char(window))
end
end
