function results = gom_perm_distribution_sensitivity(outputDir, varargin)
% Sensitivity of permeability distributions for the six GOM paper windows.
%
% This function uses the six throw windows corresponding to the WRR paper
% and quantifies how the estimated permeability distributions change with
% the number of realizations. For each window, it:
%   1. Generates a reference set with ReferenceNsim realizations.
%   2. Either:
%         - uses the first N realizations from that same reference stream
%           for each requested test sample size ('nested' mode), or
%         - reruns each test sample size independently ('independent' mode).
%   3. Compares each test distribution against the reference distribution.
%   4. Saves distribution figures and summary tables.
%
% The default comparison metric is the mean absolute error (MAE) of the
% bin probabilities in log10-permeability space. Hellinger distance and
% total variation distance are also saved as complementary metrics.
%
% Usage:
%   results = gom_perm_distribution_sensitivity()
%   results = gom_perm_distribution_sensitivity('D:\tmp\gom_sensitivity')
%
% Name-value options:
%   'ReferenceNsim' - reference sample size. Default: 1000
%   'TestNsims'     - sample sizes to compare. Default: [20 50 100 200 500]
%   'Windows'       - paper windows. Default: {'famp1',...,'famp6'}
%   'CorrCoef'      - copula correlation coefficient. Default: 0.6
%   'BaseSeed'      - base RNG seed. Default: 1729
%   'Mode'          - 'nested' (default) or 'independent'
%   'ShowProgress'  - print progress. Default: true
%   'BinEdges'      - histogram bin edges in log10(mD). Default: linspace(-7, 3, 25)
%
% Notes:
%   - Run MRST startup.m before calling this function.
%   - 'nested' mode is faster and isolates convergence along one stochastic
%     sample stream.
%   - 'independent' mode is more faithful to separate reruns, but is much
%     more computationally expensive.

if nargin < 1 || isempty(outputDir)
    outputDir = fullfile(pwd, 'gom_perm_distribution_sensitivity');
end

parser = inputParser;
parser.addParameter('ReferenceNsim', 1000, @(x) isnumeric(x) && isscalar(x) && x >= 1);
parser.addParameter('TestNsims', [20 50 100 200 500], @(x) isnumeric(x) && isvector(x) && all(x >= 1));
parser.addParameter('Windows', {'famp1', 'famp2', 'famp3', 'famp4', 'famp5', 'famp6'}, ...
                    @(x) iscell(x) || isstring(x));
parser.addParameter('CorrCoef', 0.6, @(x) isnumeric(x) && isscalar(x));
parser.addParameter('BaseSeed', 1729, @(x) isnumeric(x) && isscalar(x));
parser.addParameter('Mode', 'nested', @(x) ischar(x) || isstring(x));
parser.addParameter('ShowProgress', true, @(x) islogical(x) && isscalar(x));
parser.addParameter('BinEdges', linspace(-7, 3, 25), @(x) isnumeric(x) && isvector(x) && numel(x) >= 3);
parser.parse(varargin{:});
opt = parser.Results;
opt.Mode = char(lower(string(opt.Mode)));
assert(ismember(opt.Mode, {'nested', 'independent'}), ...
       'Mode must be ''nested'' or ''independent''.')

testNsims = sort(unique(opt.TestNsims(:)'));
assert(all(testNsims < opt.ReferenceNsim), ...
       'All TestNsims values must be smaller than ReferenceNsim.')

windows = cellstr(string(opt.Windows));
compNames = {'kxx', 'kyy', 'kzz'};
compLabels = {'$k_{xx}$', '$k_{yy}$', '$k_{zz}$'};
binEdges = opt.BinEdges(:)';
binCenters = binEdges(1:end-1) + diff(binEdges)/2;

assert(exist('mrstModule', 'file') == 2, ...
       ['MRST is not on the MATLAB path. Run startup.m in your MRST ' ...
        'folder before calling gom_perm_distribution_sensitivity.'])
mrstModule add mrst-gui coarsegrid upscaling incomp mpfa mimetic
mrstVerbose off

figDir = fullfile(outputDir, 'figures');
tableDir = fullfile(outputDir, 'tables');
dataDir = fullfile(outputDir, 'data');
ensureFolder(outputDir);
ensureFolder(figDir);
ensureFolder(tableDir);
ensureFolder(dataDir);

U.useAcceleration = 1;
U.method = 'tpfa';
U.coarseDims = [1 1 1];
U.flexible = true;
U.exportJutulInputs = false;

nWindows = numel(windows);
nTests = numel(testNsims);
nComp = numel(compNames);
nBins = numel(binCenters);

mae = nan(nWindows, nTests, nComp);
hellinger = nan(nWindows, nTests, nComp);
totalVariation = nan(nWindows, nTests, nComp);
referencePerms = cell(nWindows, 1);
referenceHist = nan(nWindows, nComp, nBins);
testHist = nan(nWindows, nTests, nComp, nBins);

for iw = 1:nWindows
    window = windows{iw};
    if opt.ShowProgress
        fprintf('\nWindow %s: generating %d-reference realizations...\n', ...
                window, opt.ReferenceNsim);
    end

    windowOpt = getWindowOptions(window);
    mySect = buildFaultedSection(windowOpt);

    rng(opt.BaseSeed + iw - 1, 'twister');
    permsRef = runWindowPermSamples(mySect, windowOpt, opt.ReferenceNsim, ...
                                    opt.CorrCoef, U, opt.ShowProgress);
    permsRef = sanitizePerms(permsRef, window, 'reference');
    referencePerms{iw} = permsRef;
    referenceHist(iw, :, :) = getHistogramProbabilities(permsRef, binEdges);

    saveReferenceFigure(window, permsRef, squeeze(referenceHist(iw, :, :)), ...
                        binCenters, figDir, compLabels, opt.ReferenceNsim);

    windowTestHist = nan(nTests, nComp, nBins);
    for it = 1:nTests
        nCurrent = testNsims(it);
        if strcmp(opt.Mode, 'nested')
            assert(size(permsRef, 1) >= nCurrent, ...
                   ['After filtering invalid reference realizations, there are not ' ...
                    'enough samples left for N = ' num2str(nCurrent) ...
                    ' in window ' window '.'])
            permsTest = permsRef(1:nCurrent, :);
        else
            if opt.ShowProgress
                fprintf('  Independent rerun for N = %d...\n', nCurrent);
            end
            rng(getIndependentSeed(opt.BaseSeed, iw, it), 'twister');
            permsTest = runWindowPermSamples(mySect, windowOpt, nCurrent, ...
                                             opt.CorrCoef, U, false);
            permsTest = sanitizePerms(permsTest, window, ['N' num2str(nCurrent)]);
        end
        pRef = squeeze(referenceHist(iw, :, :));
        pTest = getHistogramProbabilities(permsTest, binEdges);
        windowTestHist(it, :, :) = pTest;
        [mae(iw, it, :), hellinger(iw, it, :), totalVariation(iw, it, :)] = ...
            compareProbabilityDistributions(pRef, pTest);
    end
    testHist(iw, :, :, :) = windowTestHist;

    saveComparisonFigure(window, squeeze(referenceHist(iw, :, :)), ...
                         windowTestHist, binCenters, testNsims, ...
                         squeeze(mae(iw, :, :)), ...
                         squeeze(hellinger(iw, :, :)), opt.ReferenceNsim, ...
                         figDir, compLabels);

    save(fullfile(dataDir, [window '_distribution_data.mat']), ...
         'permsRef', 'windowTestHist', 'binEdges', 'binCenters');
end

maeOverall = mean(mae, 3, 'omitnan');
hellingerOverall = mean(hellinger, 3, 'omitnan');
tvOverall = mean(totalVariation, 3, 'omitnan');

maeOverallTable = makeMetricTable(maeOverall, windows, testNsims);
hellingerOverallTable = makeMetricTable(hellingerOverall, windows, testNsims);
tvOverallTable = makeMetricTable(tvOverall, windows, testNsims);
maeTripletTable = makeTripletMetricTable(mae, windows, testNsims, compNames);
hellingerTripletTable = makeTripletMetricTable(hellinger, windows, testNsims, compNames);
tvTripletTable = makeTripletMetricTable(totalVariation, windows, testNsims, compNames);

fprintf('\nOverall MAE Table (mean across kxx, kyy, kzz)\n');
disp(maeOverallTable)
fprintf('\nMAE Triplet Table [kxx | kyy | kzz]\n');
disp(maeTripletTable)
fprintf('\nOverall Hellinger Table (mean across kxx, kyy, kzz)\n');
disp(hellingerOverallTable)
fprintf('\nHellinger Triplet Table [kxx | kyy | kzz]\n');
disp(hellingerTripletTable)
fprintf('\nOverall Total Variation Table (mean across kxx, kyy, kzz)\n');
disp(tvOverallTable)
fprintf('\nTotal Variation Triplet Table [kxx | kyy | kzz]\n');
disp(tvTripletTable)

writetable(maeOverallTable, fullfile(tableDir, 'mae_overall.csv'), 'WriteRowNames', true);
writetable(hellingerOverallTable, fullfile(tableDir, 'hellinger_overall.csv'), 'WriteRowNames', true);
writetable(tvOverallTable, fullfile(tableDir, 'total_variation_overall.csv'), 'WriteRowNames', true);
writetable(maeTripletTable, fullfile(tableDir, 'mae_triplet.csv'), 'WriteRowNames', true);
writetable(hellingerTripletTable, fullfile(tableDir, 'hellinger_triplet.csv'), 'WriteRowNames', true);
writetable(tvTripletTable, fullfile(tableDir, 'total_variation_triplet.csv'), 'WriteRowNames', true);

maeTables = struct();
hellingerTables = struct();
tvTables = struct();
for ic = 1:nComp
    maeTables.(compNames{ic}) = makeMetricTable(mae(:, :, ic), windows, testNsims);
    hellingerTables.(compNames{ic}) = makeMetricTable(hellinger(:, :, ic), windows, testNsims);
    tvTables.(compNames{ic}) = makeMetricTable(totalVariation(:, :, ic), windows, testNsims);

    writetable(maeTables.(compNames{ic}), ...
               fullfile(tableDir, ['mae_' compNames{ic} '.csv']), ...
               'WriteRowNames', true);
    writetable(hellingerTables.(compNames{ic}), ...
               fullfile(tableDir, ['hellinger_' compNames{ic} '.csv']), ...
               'WriteRowNames', true);
    writetable(tvTables.(compNames{ic}), ...
               fullfile(tableDir, ['total_variation_' compNames{ic} '.csv']), ...
               'WriteRowNames', true);
end

saveSummaryHeatmap(maeOverall, windows, testNsims, figDir, ...
                   sprintf('Overall MAE vs. %d-realization reference', opt.ReferenceNsim), ...
                   'mae_summary');

results = struct();
results.Config = opt;
results.Windows = windows;
results.TestNsims = testNsims;
results.ReferenceNsim = opt.ReferenceNsim;
results.ComponentNames = compNames;
results.BinEdges = binEdges;
results.BinCenters = binCenters;
results.ReferencePerms = referencePerms;
results.ReferenceHist = referenceHist;
results.TestHist = testHist;
results.MAE = mae;
results.Hellinger = hellinger;
results.TotalVariation = totalVariation;
results.MAEOverallTable = maeOverallTable;
results.HellingerOverallTable = hellingerOverallTable;
results.TotalVariationOverallTable = tvOverallTable;
results.MAETripletTable = maeTripletTable;
results.HellingerTripletTable = hellingerTripletTable;
results.TotalVariationTripletTable = tvTripletTable;
results.MAETables = maeTables;
results.HellingerTables = hellingerTables;
results.TotalVariationTables = tvTables;

save(fullfile(outputDir, 'gom_perm_distribution_sensitivity_results.mat'), ...
     'results', '-v7.3');
end


function mySect = buildFaultedSection(windowOpt)
% Build the faulted section object once for a given window.

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
                            'maxPerm', windowOpt.maxPerm, ...
                            'totThick', windowOpt.totThick);
else
    mySect = FaultedSection(footwall, hangingwall, windowOpt.fDip, ...
                            'maxPerm', windowOpt.maxPerm);
end
mySect = mySect.getMatPropDistr();
end


function perms = runWindowPermSamples(mySect, windowOpt, Nsim, rho, U, showProgress)
% Generate only the permeability outputs, not the full realization objects.

perms = nan(Nsim, 3);
progressStep = max(1, min(100, floor(Nsim/10)));
for n = 1:Nsim
    perms(n, :) = runSingleWindowPermRealization(mySect, windowOpt, rho, U);

    if showProgress && (mod(n, progressStep) == 0 || n == Nsim)
        fprintf('  Realization %d / %d completed.\n', n, Nsim);
    end
end
end


function perm = runSingleWindowPermRealization(mySect, windowOpt, rho, U)
% Run one realization and return only the upscaled permeability in mD.

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
    myFaultSection = myFaultSection.placeMaterials(mySect, smear, G);
    myFault = myFault.assignExtrudedVals(G, myFaultSection, k);
end

[myFault, ~] = myFault.upscaleProps(G, Urun);
perm = myFault.Perm ./ (milli*darcy);
end


function perms = sanitizePerms(perms, window, label)
% Remove any realization with a non-positive permeability component.

mask = all(perms > 0, 2) & all(isfinite(perms), 2);
numRemoved = sum(~mask);
if numRemoved > 0
    warning('%s: removed %d invalid permeability realizations from %s.', ...
            window, numRemoved, label)
end
perms = perms(mask, :);
end


function probs = getHistogramProbabilities(perms, binEdges)
% Histogram probabilities in log10(mD) space.

nComp = size(perms, 2);
nBins = numel(binEdges) - 1;
probs = zeros(nComp, nBins);
logPerm = log10(perms);
for ic = 1:nComp
    probs(ic, :) = histcounts(logPerm(:, ic), binEdges, ...
                              'Normalization', 'probability');
end
end


function [mae, hellinger, totalVariation] = compareProbabilityDistributions(pRef, pTest)
% Compare discrete probability distributions component-wise.

delta = abs(pTest - pRef);
mae = mean(delta, 2);
totalVariation = 0.5 * sum(delta, 2);
hellinger = sqrt(0.5 * sum((sqrt(pTest) - sqrt(pRef)).^2, 2));
end


function tbl = makeMetricTable(values, windows, testNsims)
% Build a summary table.

varNames = matlab.lang.makeValidName("N" + string(testNsims));
tbl = array2table(values, 'VariableNames', cellstr(varNames), ...
                  'RowNames', windows);
end


function tbl = makeTripletMetricTable(values, windows, testNsims, compNames)
% Build a table with one string entry per cell: kxx | kyy | kzz.

varNames = matlab.lang.makeValidName("N" + string(testNsims));
tripletStrings = strings(numel(windows), numel(testNsims));
for i = 1:numel(windows)
    for j = 1:numel(testNsims)
        tripletStrings(i, j) = sprintf('%s=%.4f | %s=%.4f | %s=%.4f', ...
            compNames{1}, values(i, j, 1), ...
            compNames{2}, values(i, j, 2), ...
            compNames{3}, values(i, j, 3));
    end
end

tripletCells = cellstr(tripletStrings);
tbl = cell2table(tripletCells, ...
                 'VariableNames', cellstr(varNames), ...
                 'RowNames', windows);
end


function saveReferenceFigure(window, permsRef, probsRef, binCenters, figDir, compLabels, refNsim)
% Save the reference distribution figure for one window.

fh = figure('Visible', 'off', 'Color', 'w');
tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
for ic = 1:3
    nexttile
    stairs(binCenters, probsRef(ic, :), 'k-', 'LineWidth', 1.5)
    grid on
    xlabel(['$\log_{10}$(' compLabels{ic} ' [mD])'], 'Interpreter', 'latex')
    ylabel('Probability [-]', 'Interpreter', 'latex')
    title(sprintf('%s, N = %d', window, refNsim), 'Interpreter', 'none')
    xlim([binCenters(1) binCenters(end)])
    ylim([0, 1.05 * max(probsRef(ic, :), [], 'all') + eps])
end

summaryText = sprintf(['Reference sample summary\n' ...
                       'Median [mD]: %.3g | %.3g | %.3g\n' ...
                       'Samples used: %d'], ...
                       median(permsRef(:, 1)), median(permsRef(:, 2)), ...
                       median(permsRef(:, 3)), size(permsRef, 1));
annotation(fh, 'textbox', [0.02 0.86 0.3 0.12], 'String', summaryText, ...
           'FitBoxToText', 'on', 'EdgeColor', 'none');

saveFigurePair(fh, fullfile(figDir, [window '_reference_distribution']));
close(fh)
end


function saveComparisonFigure(window, probsRef, probsTest, binCenters, testNsims, maeVals, hellVals, refNsim, figDir, compLabels)
% Save the comparison figure for one window.

nTests = numel(testNsims);
colors = lines(nTests);

fh = figure('Visible', 'off', 'Color', 'w');
tiledlayout(nTests, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
for it = 1:nTests
    for ic = 1:3
        nexttile
        stairs(binCenters, probsRef(ic, :), 'k-', 'LineWidth', 1.5)
        hold on
        stairs(binCenters, squeeze(probsTest(it, ic, :)), '-', ...
               'Color', colors(it, :), 'LineWidth', 1.3)
        hold off
        grid on
        xlabel(['$\log_{10}$(' compLabels{ic} ' [mD])'], 'Interpreter', 'latex')
        ylabel('Probability [-]', 'Interpreter', 'latex')
        title(sprintf('N = %d | MAE = %.4f | H = %.4f', ...
              testNsims(it), maeVals(it, ic), hellVals(it, ic)))
        xlim([binCenters(1) binCenters(end)])
        ylim([0, 1.05 * max([probsRef(ic, :), squeeze(probsTest(it, ic, :))']) + eps])
        if it == 1 && ic == 1
            legend({'Reference', 'Subset'}, 'Location', 'best')
        end
    end
end

sgtitle(sprintf('%s distribution sensitivity vs. N = %d reference', ...
        window, refNsim), 'Interpreter', 'none')
saveFigurePair(fh, fullfile(figDir, [window '_distribution_comparison']));
close(fh)
end


function seed = getIndependentSeed(baseSeed, windowId, testId)
% Deterministic but distinct seed for each independent rerun.

seed = baseSeed + 1000*windowId + 100*testId;
end


function saveSummaryHeatmap(values, windows, testNsims, figDir, figureTitle, stem)
% Save a summary heatmap-like figure for one metric.

fh = figure('Visible', 'off', 'Color', 'w');
imagesc(values)
colormap(parula)
colorbar
title(figureTitle, 'Interpreter', 'none')
xticks(1:numel(testNsims))
xticklabels(cellstr("N = " + string(testNsims)))
yticks(1:numel(windows))
yticklabels(windows)
xlabel('Sample size')
ylabel('Window')

for i = 1:size(values, 1)
    for j = 1:size(values, 2)
        text(j, i, sprintf('%.4f', values(i, j)), ...
             'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'middle', ...
             'Color', 'w', 'FontSize', 9)
    end
end

saveFigurePair(fh, fullfile(figDir, stem));
close(fh)
end


function saveFigurePair(figHandle, basePath)
% Save both PNG and FIG versions of a figure.

savefig(figHandle, [basePath '.fig']);
exportgraphics(figHandle, [basePath '.png'], 'Resolution', 200);
end


function ensureFolder(folderPath)
% Create folder if needed.

if ~exist(folderPath, 'dir')
    mkdir(folderPath);
end
end


function opt = getWindowOptions(window)
% Window-specific paper inputs.

opt.window = window;
opt.maxPerm = 175; % [mD], max perm of Amp B interval (sand layers)

switch lower(window)
    case 'famp1' % bottom throw window
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

    case 'famp6' % top throw window
        opt.thick = {[28.2932 33.1042 33.1699 33.1042], 127.6715};
        opt.vcl = {[0.2878 0.65 0.2878 0.65], 0.3266};
        opt.dip = [0, -5];
        opt.fDip = 46.0685;
        opt.zf = [200, 200];
        opt.zmax = {[1440.6 1417.5 1392.5 1367.5], 1400};

    otherwise
        error(['Unsupported window "' window '". Choose one of: ' ...
               'famp1, famp2, famp3, famp4, famp5, famp6.'])
end
end
