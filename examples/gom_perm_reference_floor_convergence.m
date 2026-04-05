function results = gom_perm_reference_floor_convergence(outputDir, varargin)
% Rigorous convergence study with an empirical reference-distance floor.
%
% This driver extends the original GOM distribution sensitivity study by
% no longer treating one large reference ensemble as exact truth. Instead,
% it builds multiple independent reference ensembles, measures the
% reference-vs-reference distance floor, and then compares repeated small
% runs against that floor.
%
% For each selected window and each permeability component (kxx, kyy, kzz),
% the workflow is:
%   1. Generate NumReferences independent reference ensembles of size
%      ReferenceNsim.
%   2. Compute all pairwise reference-reference distances in log10(k) space
%      using MAE, Hellinger, and Wasserstein distance.
%   3. Define the reference floor from those pairwise distances:
%         - center = median(pairwise distances)
%         - spread = [min(pairwise distances), max(pairwise distances)]
%   4. For each N in TestNsims, run NumRepeats independent small ensembles.
%   5. For each repeated small ensemble, compute its distance to every
%      reference and collapse those distances to one run-level score using
%      the median across references.
%   6. Summarize the repeated run-level scores by median, P10, and P90.
%   7. Fit log10(D) = a + b*log10(Nsim) to the median convergence curve.
%
% Raw permeability values for all references and all repeated small runs
% are saved as soon as each ensemble finishes, and the driver can resume
% from those saved checkpoints after an interruption. Summary tables and
% figures are saved alongside them.
%
% Usage:
%   results = gom_perm_reference_floor_convergence()
%   results = gom_perm_reference_floor_convergence('D:\codex_gom\run1')
%
% Name-value options:
%   'ReferenceNsim' - size of each reference ensemble. Default: 20000
%   'NumReferences' - number of independent references. Default: 3
%   'TestNsims'     - repeated small-run sample sizes.
%                     Default: [20 50 100 200 300 400 500 750 1000 1500 2000]
%   'NumRepeats'    - independent repeats for each small-run size.
%                     Default: 30
%   'Windows'       - paper windows. Default: {'famp1',...,'famp6'}
%   'CorrCoef'      - copula correlation coefficient. Default: 0.6
%   'BaseSeed'      - deterministic seed base. Default: 1729
%   'UseParallel'   - run realizations with parfor. Default: false
%   'NumWorkers'    - requested pool size when auto-starting. Default: []
%   'Resume'        - reuse saved raw-ensemble checkpoints when present.
%                     Default: true
%   'ShowProgress'  - print progress. Default: true
%   'MakePlots'     - save convergence figures. Default: true
%   'BinEdges'      - histogram bin edges in log10(mD).
%                     Default: linspace(-7, 3, 25)
%
% Notes:
%   - Run MRST startup.m before calling this function.
%   - Checkpoints are stored per window under data/<window>/references and
%     data/<window>/small_runs.
%   - MAE and Hellinger are computed on histogram probabilities in
%     log10(mD) space using the shared BinEdges.
%   - Wasserstein distance is computed directly from the raw log10(k)
%     samples and does not depend on histogram bins.

if nargin < 1 || isempty(outputDir)
    outputDir = fullfile(pwd, 'gom_perm_reference_floor_convergence');
end

parser = inputParser;
parser.addParameter('ReferenceNsim', 20000, @(x) isnumeric(x) && isscalar(x) && x >= 2);
parser.addParameter('NumReferences', 3, @(x) isnumeric(x) && isscalar(x) && x >= 2 && mod(x, 1) == 0);
parser.addParameter('TestNsims', [20 50 100 200 300 400 500 750 1000 1500 2000], ...
                    @(x) isnumeric(x) && isvector(x) && all(x >= 2));
parser.addParameter('NumRepeats', 30, @(x) isnumeric(x) && isscalar(x) && x >= 1 && mod(x, 1) == 0);
parser.addParameter('Windows', {'famp1', 'famp2', 'famp3', 'famp4', 'famp5', 'famp6'}, ...
                    @(x) iscell(x) || isstring(x));
parser.addParameter('CorrCoef', 0.6, @(x) isnumeric(x) && isscalar(x));
parser.addParameter('BaseSeed', 1729, @(x) isnumeric(x) && isscalar(x));
parser.addParameter('UseParallel', false, @(x) islogical(x) && isscalar(x));
parser.addParameter('NumWorkers', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 1));
parser.addParameter('Resume', true, @(x) islogical(x) && isscalar(x));
parser.addParameter('ShowProgress', true, @(x) islogical(x) && isscalar(x));
parser.addParameter('MakePlots', true, @(x) islogical(x) && isscalar(x));
parser.addParameter('BinEdges', linspace(-7, 3, 25), @(x) isnumeric(x) && isvector(x) && numel(x) >= 3);
parser.parse(varargin{:});
opt = parser.Results;

testNsims = sortUniqueNumericVector(opt.TestNsims(:)');
assert(all(testNsims < opt.ReferenceNsim), ...
       'All TestNsims values must be smaller than ReferenceNsim.')
windows = cellstr(string(opt.Windows));

metricNames = {'MAE', 'Hellinger', 'Wasserstein'};
metricStems = {'mae', 'hellinger', 'wasserstein'};
compNames = {'kxx', 'kyy', 'kzz'};
compLabels = {'k_{xx}', 'k_{yy}', 'k_{zz}'};
binEdges = opt.BinEdges(:)';
numMetrics = numel(metricNames);
numComp = numel(compNames);
numWindows = numel(windows);
numTests = numel(testNsims);
numRefs = opt.NumReferences;

assert(exist('mrstModule', 'file') == 2, ...
       ['MRST is not on the MATLAB path. Run startup.m in your MRST ' ...
        'folder before calling gom_perm_reference_floor_convergence.'])
mrstModule add mrst-gui coarsegrid upscaling incomp mpfa mimetic
mrstVerbose off
if opt.UseParallel
    ensurePredictParallelPool(opt.NumWorkers);
end

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

referenceFloorBlocks = cell(numWindows, 1);
repeatScoreBlocks = cell(numWindows, 1);
summaryBlocks = cell(numWindows, 1);
fitBlocks = cell(numWindows, 1);
ensembleMetaBlocks = cell(numWindows, 1);
windowDataFiles = strings(numWindows, 1);
windowFigureFiles = strings(numWindows, 1);

for iw = 1:numWindows
    window = windows{iw};
    if opt.ShowProgress
        fprintf('\n=== Window %s ===\n', window);
    end

    windowDir = fullfile(dataDir, window);
    referenceDir = fullfile(windowDir, 'references');
    smallRunDir = fullfile(windowDir, 'small_runs');
    ensureFolder(windowDir);
    ensureFolder(referenceDir);
    ensureFolder(smallRunDir);

    windowOpt = getWindowOptions(window);
    mySect = buildFaultedSection(windowOpt);

    referencePerms = cell(numRefs, 1);
    referenceMeta = cell(numRefs, 1);
    referenceLogPerms = cell(numRefs, 1);
    referenceHist = nan(numRefs, numComp, numel(binEdges)-1);
    for ir = 1:numRefs
        runLabel = sprintf('%s reference R%d', window, ir);
        refSeed = makeReferenceSeed(opt.BaseSeed, iw, ir);
        refFile = fullfile(referenceDir, sprintf('reference_R%d.mat', ir));
        expected = struct('Kind', 'Reference', 'Window', window, ...
                          'TargetN', opt.ReferenceNsim, ...
                          'SeedBase', refSeed, ...
                          'CorrCoef', opt.CorrCoef, ...
                          'ReferenceId', ir, 'Nsim', NaN, 'Repeat', NaN);
        if opt.Resume
            [referencePerms{ir}, referenceMeta{ir}, loadedFromCheckpoint] = ...
                loadEnsembleCheckpoint(refFile, expected);
        else
            loadedFromCheckpoint = false;
        end
        if loadedFromCheckpoint
            if opt.ShowProgress
                fprintf('  %s: resumed from %s\n', runLabel, refFile);
            end
        else
            [referencePerms{ir}, referenceMeta{ir}] = runWindowPermSamplesExact( ...
                mySect, windowOpt, opt.ReferenceNsim, opt.CorrCoef, U, ...
                opt.UseParallel, refSeed, opt.ShowProgress, runLabel);
            saveEnsembleCheckpoint(refFile, referencePerms{ir}, ...
                                   referenceMeta{ir}, expected);
            if opt.ShowProgress
                fprintf('  %s: saved checkpoint %s\n', runLabel, refFile);
            end
        end
        referenceLogPerms{ir} = log10(referencePerms{ir});
        referenceHist(ir, :, :) = getHistogramProbabilitiesFromLog(referenceLogPerms{ir}, binEdges);
    end

    refPairs = nchoosek(1:numRefs, 2);
    numPairs = size(refPairs, 1);
    pairLabels = strings(numPairs, 1);
    referencePairDistances = nan(numPairs, numComp, numMetrics);
    for ip = 1:numPairs
        ia = refPairs(ip, 1);
        ib = refPairs(ip, 2);
        pairLabels(ip) = sprintf('R%d_R%d', ia, ib);
        referencePairDistances(ip, :, :) = computeDistancesFromPrepared( ...
            referenceLogPerms{ia}, squeeze(referenceHist(ia, :, :)), ...
            referenceLogPerms{ib}, squeeze(referenceHist(ib, :, :)));
    end

    floorMedian = squeeze(median(referencePairDistances, 1, 'omitnan'));
    floorMin = squeeze(min(referencePairDistances, [], 1));
    floorMax = squeeze(max(referencePairDistances, [], 1));

    smallPerms = cell(numTests, opt.NumRepeats);
    smallMeta = cell(numTests, opt.NumRepeats);
    smallToReferenceDistances = nan(numTests, opt.NumRepeats, numRefs, numComp, numMetrics);
    runScores = nan(numTests, opt.NumRepeats, numComp, numMetrics);

    for it = 1:numTests
        nCurrent = testNsims(it);
        if opt.ShowProgress
            fprintf('Window %s: Nsim = %d, %d repeats...\n', ...
                    window, nCurrent, opt.NumRepeats);
        end
        for irpt = 1:opt.NumRepeats
            runLabel = sprintf('%s N%d repeat %d', window, nCurrent, irpt);
            runSeed = makeSmallRunSeed(opt.BaseSeed, iw, it, irpt);
            runFile = fullfile(smallRunDir, ...
                               sprintf('N%d_repeat%02d.mat', nCurrent, irpt));
            expected = struct('Kind', 'SmallRun', 'Window', window, ...
                              'TargetN', nCurrent, ...
                              'SeedBase', runSeed, ...
                              'CorrCoef', opt.CorrCoef, ...
                              'ReferenceId', NaN, 'Nsim', nCurrent, ...
                              'Repeat', irpt);
            if opt.Resume
                [smallPerms{it, irpt}, smallMeta{it, irpt}, loadedFromCheckpoint] = ...
                    loadEnsembleCheckpoint(runFile, expected);
            else
                loadedFromCheckpoint = false;
            end
            if loadedFromCheckpoint
                if opt.ShowProgress
                    fprintf('  %s: resumed from %s\n', runLabel, runFile);
                end
            else
                [smallPerms{it, irpt}, smallMeta{it, irpt}] = runWindowPermSamplesExact( ...
                    mySect, windowOpt, nCurrent, opt.CorrCoef, U, ...
                    opt.UseParallel, runSeed, false, runLabel);
                saveEnsembleCheckpoint(runFile, smallPerms{it, irpt}, ...
                                       smallMeta{it, irpt}, expected);
                if opt.ShowProgress
                    fprintf('  %s: saved checkpoint %s\n', runLabel, runFile);
                end
            end

            logSmall = log10(smallPerms{it, irpt});
            histSmall = getHistogramProbabilitiesFromLog(logSmall, binEdges);
            for ir = 1:numRefs
                smallToReferenceDistances(it, irpt, ir, :, :) = computeDistancesFromPrepared( ...
                    referenceLogPerms{ir}, squeeze(referenceHist(ir, :, :)), ...
                    logSmall, histSmall);
            end
            runScores(it, irpt, :, :) = squeeze(median(smallToReferenceDistances(it, irpt, :, :, :), 3, 'omitnan'));
        end
    end

    summaryMedian = nan(numTests, numComp, numMetrics);
    summaryP10 = nan(numTests, numComp, numMetrics);
    summaryP90 = nan(numTests, numComp, numMetrics);
    fitIntercept = nan(numComp, numMetrics);
    fitSlope = nan(numComp, numMetrics);
    fitR2 = nan(numComp, numMetrics);
    fitNumUsed = nan(numComp, numMetrics);

    for im = 1:numMetrics
        for ic = 1:numComp
            for it = 1:numTests
                scores = squeeze(runScores(it, :, ic, im));
                summaryMedian(it, ic, im) = median(scores, 'omitnan');
                summaryP10(it, ic, im) = prctile(scores, 10);
                summaryP90(it, ic, im) = prctile(scores, 90);
            end

            [fitIntercept(ic, im), fitSlope(ic, im), fitR2(ic, im), fitNumUsed(ic, im)] = ...
                fitLogLogConvergence(testNsims, summaryMedian(:, ic, im));
        end
    end

    if opt.MakePlots
        windowFigureFiles(iw) = fullfile(figDir, [window '_rigorous_convergence.png']);
        saveWindowConvergenceFigure(window, testNsims, metricNames, compNames, ...
                                    compLabels, summaryMedian, summaryP10, ...
                                    summaryP90, floorMedian, floorMin, floorMax, ...
                                    fitIntercept, fitSlope, fitR2, figDir);
    end

    referenceFloorBlocks{iw} = makeReferenceFloorLongTable(window, pairLabels, ...
        metricNames, compNames, referencePairDistances, floorMedian, floorMin, floorMax);
    repeatScoreBlocks{iw} = makeRepeatScoreLongTable(window, testNsims, ...
        metricNames, compNames, runScores);
    summaryBlocks{iw} = makeSummaryLongTable(window, testNsims, metricNames, ...
        compNames, summaryMedian, summaryP10, summaryP90, floorMedian, ...
        floorMin, floorMax);
    fitBlocks{iw} = makeFitLongTable(window, metricNames, compNames, ...
        fitIntercept, fitSlope, fitR2, fitNumUsed);
    ensembleMetaBlocks{iw} = makeEnsembleMetaLongTable(window, referenceMeta, ...
        smallMeta, testNsims);

    windowDataFiles(iw) = fullfile(windowDir, [window '_rigorous_convergence_data.mat']);
    save(windowDataFiles(iw), 'window', 'binEdges', 'metricNames', 'metricStems', ...
         'compNames', 'testNsims', 'referencePerms', 'referenceMeta', ...
         'referencePairDistances', 'pairLabels', 'floorMedian', 'floorMin', ...
         'floorMax', 'smallPerms', 'smallMeta', 'smallToReferenceDistances', ...
         'runScores', 'summaryMedian', 'summaryP10', 'summaryP90', ...
         'fitIntercept', 'fitSlope', 'fitR2', 'fitNumUsed', ...
         'referenceDir', 'smallRunDir', '-v7.3');
end

referenceFloorTable = vertcat(referenceFloorBlocks{:});
repeatScoreTable = vertcat(repeatScoreBlocks{:});
summaryTable = vertcat(summaryBlocks{:});
fitTable = vertcat(fitBlocks{:});
ensembleMetaTable = vertcat(ensembleMetaBlocks{:});

writetable(referenceFloorTable, fullfile(tableDir, 'reference_floor_long.csv'));
writetable(repeatScoreTable, fullfile(tableDir, 'repeat_score_long.csv'));
writetable(summaryTable, fullfile(tableDir, 'convergence_summary_long.csv'));
writetable(fitTable, fullfile(tableDir, 'fit_summary_long.csv'));
writetable(ensembleMetaTable, fullfile(tableDir, 'ensemble_metadata_long.csv'));

results = struct();
results.Config = opt;
results.Windows = windows;
results.MetricNames = metricNames;
results.ComponentNames = compNames;
results.TestNsims = testNsims;
results.BinEdges = binEdges;
results.ReferenceFloorTable = referenceFloorTable;
results.RepeatScoreTable = repeatScoreTable;
results.SummaryTable = summaryTable;
results.FitTable = fitTable;
results.EnsembleMetaTable = ensembleMetaTable;
results.WindowDataFiles = cellstr(windowDataFiles);
results.WindowFigureFiles = cellstr(windowFigureFiles);

save(fullfile(outputDir, 'gom_perm_reference_floor_convergence_results.mat'), ...
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


function [perms, meta] = runWindowPermSamplesExact(mySect, windowOpt, targetN, rho, U, useParallel, seedBase, showProgress, label)
% Generate exactly targetN valid permeability samples.

perms = nan(targetN, 3);
numValid = 0;
numAttempts = 0;
numRejected = 0;
batchId = 0;

while numValid < targetN
    batchId = batchId + 1;
    remaining = targetN - numValid;
    batchSeedBase = seedBase + numAttempts;
    batchPerms = runWindowPermSampleBatch(mySect, windowOpt, remaining, rho, U, ...
                                          useParallel, batchSeedBase);
    [validPerms, rejectedThisBatch] = sanitizeBatchPerms(batchPerms);
    takeCount = min(remaining, size(validPerms, 1));
    if takeCount > 0
        perms(numValid + (1:takeCount), :) = validPerms(1:takeCount, :);
        numValid = numValid + takeCount;
    end

    numAttempts = numAttempts + remaining;
    numRejected = numRejected + rejectedThisBatch;

    if showProgress
        fprintf('  %s: batch %d, valid %d / %d, attempts %d, rejected %d\n', ...
                label, batchId, numValid, targetN, numAttempts, numRejected);
    end

    if numAttempts > 10*targetN
        error(['Too many invalid permeability realizations while building ' ...
               label '. Aborting after %d attempts.'], numAttempts)
    end
end

meta = struct();
meta.Label = label;
meta.TargetN = targetN;
meta.NumReturned = size(perms, 1);
meta.NumAttempts = numAttempts;
meta.NumRejected = numRejected;
meta.AcceptanceRatio = targetN / numAttempts;
meta.SeedBase = seedBase;
end


function perms = runWindowPermSampleBatch(mySect, windowOpt, Nsim, rho, U, useParallel, seedBase)
% Generate one batch of permeability realizations.

perms = nan(Nsim, 3);
if useParallel
    parfor n = 1:Nsim
        rng(seedBase + n - 1, 'twister');
        perms(n, :) = runSingleWindowPermRealization(mySect, windowOpt, rho, U);
    end
else
    for n = 1:Nsim
        rng(seedBase + n - 1, 'twister');
        perms(n, :) = runSingleWindowPermRealization(mySect, windowOpt, rho, U);
    end
end
end


function perm = runSingleWindowPermRealization(mySect, windowOpt, rho, U)
% Run one 3D realization and return only the upscaled permeability in mD.

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


function [validPerms, numRejected] = sanitizeBatchPerms(perms)
% Keep only finite positive permeability realizations.

mask = all(perms > 0, 2) & all(isfinite(perms), 2);
validPerms = perms(mask, :);
numRejected = sum(~mask);
end


function probs = getHistogramProbabilitiesFromLog(logPerm, binEdges)
% Histogram probabilities in log10(mD) space.

numComp = size(logPerm, 2);
numBins = numel(binEdges) - 1;
probs = zeros(numComp, numBins);
for ic = 1:numComp
    probs(ic, :) = histcounts(logPerm(:, ic), binEdges, ...
                              'Normalization', 'probability');
end
end


function distances = computeDistancesFromPrepared(logA, probsA, logB, probsB)
% Compute MAE, Hellinger, and Wasserstein for each permeability component.

delta = abs(probsB - probsA);
numComp = size(logA, 2);
distances = zeros(numComp, 3);
distances(:, 1) = mean(delta, 2);
distances(:, 2) = sqrt(0.5 * sum((sqrt(probsB) - sqrt(probsA)).^2, 2));
for ic = 1:numComp
    distances(ic, 3) = wasserstein1dEmpirical(logA(:, ic), logB(:, ic));
end
end


function w = wasserstein1dEmpirical(x, y)
% Exact 1D Wasserstein distance between empirical distributions.

x = sort(x(:));
y = sort(y(:));
if isempty(x) || isempty(y)
    w = NaN;
    return
end

support = mergeSortedNumericVectors(x, y);
if isscalar(support)
    w = 0;
    return
end

[~, xLoc] = ismember(x, support);
[~, yLoc] = ismember(y, support);
xCounts = accumarray(xLoc, 1, [numel(support), 1]);
yCounts = accumarray(yLoc, 1, [numel(support), 1]);

Fx = cumsum(xCounts) / numel(x);
Fy = cumsum(yCounts) / numel(y);
intervalWidths = diff(support);
w = sum(abs(Fx(1:end-1) - Fy(1:end-1)) .* intervalWidths);
end


function values = sortUniqueNumericVector(values)
% Sort a numeric vector and remove repeated entries without unique().

values = sort(values(:)');
if isempty(values)
    return
end

keepMask = [true, diff(values) ~= 0];
values = values(keepMask);
end


function values = mergeSortedNumericVectors(x, y)
% Merge two numeric vectors and remove repeats without unique().

values = sort([x(:); y(:)]);
if isempty(values)
    return
end

keepMask = [true; diff(values) ~= 0];
values = values(keepMask);
end


function [intercept, slope, r2, numUsed] = fitLogLogConvergence(testNsims, medianDistances)
% Fit log10(D) = intercept + slope*log10(Nsim) on positive finite values.

x = testNsims(:);
y = medianDistances(:);
mask = isfinite(x) & isfinite(y) & x > 0 & y > 0;
numUsed = nnz(mask);
if numUsed < 2
    intercept = NaN;
    slope = NaN;
    r2 = NaN;
    return
end

logx = log10(x(mask));
logy = log10(y(mask));
p = polyfit(logx, logy, 1);
slope = p(1);
intercept = p(2);

logyHat = polyval(p, logx);
ssRes = sum((logy - logyHat).^2);
ssTot = sum((logy - mean(logy)).^2);
if ssTot > 0
    r2 = 1 - ssRes/ssTot;
else
    r2 = 1;
end
end


function saveWindowConvergenceFigure(window, testNsims, metricNames, compNames, compLabels, summaryMedian, summaryP10, summaryP90, floorMedian, floorMin, floorMax, fitIntercept, fitSlope, fitR2, figDir)
% Save a 3x3 convergence figure: metrics by rows, components by columns.

metricColors = lines(numel(metricNames));
fh = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1600 1100]);
tiledlayout(numel(metricNames), numel(compNames), 'Padding', 'compact', 'TileSpacing', 'compact');

x = testNsims(:)';
for im = 1:numel(metricNames)
    for ic = 1:numel(compNames)
        nexttile
        medVals = summaryMedian(:, ic, im)';
        p10Vals = summaryP10(:, ic, im)';
        p90Vals = summaryP90(:, ic, im)';
        color = metricColors(im, :);

        positiveBand = all(isfinite([x, p10Vals, p90Vals])) && all(p10Vals > 0) && all(p90Vals > 0);
        if positiveBand
            patch([x, fliplr(x)], [p10Vals, fliplr(p90Vals)], color, ...
                  'FaceAlpha', 0.18, 'EdgeColor', 'none');
            hold on
        else
            hold on
        end

        floorLow = floorMin(ic, im);
        floorHigh = floorMax(ic, im);
        if isfinite(floorLow) && isfinite(floorHigh) && floorLow > 0 && floorHigh > 0
            patch([x(1), x(end), x(end), x(1)], ...
                  [floorLow, floorLow, floorHigh, floorHigh], ...
                  [0.3 0.3 0.3], 'FaceAlpha', 0.12, 'EdgeColor', 'none');
        end

        plot(x, medVals, '-o', 'Color', color, 'LineWidth', 1.7, ...
             'MarkerFaceColor', color, 'MarkerSize', 4);

        floorCenter = floorMedian(ic, im);
        if isfinite(floorCenter) && floorCenter > 0
            yline(floorCenter, 'k--', 'LineWidth', 1.1);
        end

        if isfinite(fitIntercept(ic, im)) && isfinite(fitSlope(ic, im))
            fitVals = 10.^(fitIntercept(ic, im) + fitSlope(ic, im)*log10(x));
            plot(x, fitVals, ':', 'Color', [0.1 0.1 0.1], 'LineWidth', 1.3);
        end

        grid on
        set(gca, 'XScale', 'log', 'YScale', 'log')
        xlabel('Nsim')
        ylabel('Distance')
        title(sprintf('%s | %s\nb = %.3f, R^2 = %.3f', ...
              metricNames{im}, compLabels{ic}, fitSlope(ic, im), fitR2(ic, im)));

        positiveVals = [medVals, p10Vals, p90Vals, floorLow, floorHigh, floorCenter];
        positiveVals = positiveVals(isfinite(positiveVals) & positiveVals > 0);
        if ~isempty(positiveVals)
            ylim([0.8*min(positiveVals), 1.25*max(positiveVals)])
        end
        xlim([min(x), max(x)])

        if im == 1 && ic == 1
            legend({'P10-P90', 'Reference floor range', 'Median score', ...
                    'Reference floor median', 'Log-log fit'}, ...
                   'Location', 'best');
        end
        hold off
    end
end

sgtitle(sprintf('%s rigorous convergence study in log_{10}(k) space', window), ...
        'Interpreter', 'none')
annotation(fh, 'textbox', [0.01 0.96 0.35 0.03], ...
           'String', ['Distances are computed separately for ' ...
                      'log_{10}(k_{xx}), log_{10}(k_{yy}), and log_{10}(k_{zz}).'], ...
           'EdgeColor', 'none');

saveFigurePair(fh, fullfile(figDir, [window '_rigorous_convergence']));
close(fh)
end


function tbl = makeReferenceFloorLongTable(window, pairLabels, metricNames, compNames, pairDistances, floorMedian, floorMin, floorMax)
% Build one long table block for the reference-distance floor.

rows = {};
for ip = 1:numel(pairLabels)
    for im = 1:numel(metricNames)
        for ic = 1:numel(compNames)
            rows(end+1, :) = {window, metricNames{im}, compNames{ic}, ...
                              pairLabels(ip), pairDistances(ip, ic, im), ...
                              floorMedian(ic, im), floorMin(ic, im), ...
                              floorMax(ic, im)}; %#ok<AGROW>
        end
    end
end

tbl = cell2table(rows, 'VariableNames', ...
    {'Window', 'Metric', 'Component', 'ReferencePair', 'Distance', ...
     'FloorMedian', 'FloorMin', 'FloorMax'});
end


function tbl = makeRepeatScoreLongTable(window, testNsims, metricNames, compNames, runScores)
% Build one long table block for all repeated run-level scores.

rows = {};
for it = 1:numel(testNsims)
    for irpt = 1:size(runScores, 2)
        for im = 1:numel(metricNames)
            for ic = 1:numel(compNames)
                rows(end+1, :) = {window, testNsims(it), irpt, metricNames{im}, ...
                                  compNames{ic}, runScores(it, irpt, ic, im)}; %#ok<AGROW>
            end
        end
    end
end

tbl = cell2table(rows, 'VariableNames', ...
    {'Window', 'Nsim', 'Repeat', 'Metric', 'Component', 'Score'});
end


function tbl = makeSummaryLongTable(window, testNsims, metricNames, compNames, summaryMedian, summaryP10, summaryP90, floorMedian, floorMin, floorMax)
% Build one long summary table block for the convergence curves.

rows = {};
for it = 1:numel(testNsims)
    for im = 1:numel(metricNames)
        for ic = 1:numel(compNames)
            rows(end+1, :) = {window, testNsims(it), metricNames{im}, ...
                              compNames{ic}, summaryMedian(it, ic, im), ...
                              summaryP10(it, ic, im), summaryP90(it, ic, im), ...
                              floorMedian(ic, im), floorMin(ic, im), ...
                              floorMax(ic, im)}; %#ok<AGROW>
        end
    end
end

tbl = cell2table(rows, 'VariableNames', ...
    {'Window', 'Nsim', 'Metric', 'Component', 'MedianScore', ...
     'P10Score', 'P90Score', 'FloorMedian', 'FloorMin', 'FloorMax'});
end


function tbl = makeFitLongTable(window, metricNames, compNames, fitIntercept, fitSlope, fitR2, fitNumUsed)
% Build one fit-summary table block.

rows = {};
for im = 1:numel(metricNames)
    for ic = 1:numel(compNames)
        rows(end+1, :) = {window, metricNames{im}, compNames{ic}, ...
                          fitIntercept(ic, im), fitSlope(ic, im), ...
                          fitR2(ic, im), fitNumUsed(ic, im)}; %#ok<AGROW>
    end
end

tbl = cell2table(rows, 'VariableNames', ...
    {'Window', 'Metric', 'Component', 'FitIntercept', ...
     'FitSlope', 'FitR2', 'NumFitPoints'});
end


function tbl = makeEnsembleMetaLongTable(window, referenceMeta, smallMeta, testNsims)
% Build one long table block with acceptance / rejection metadata.

rows = {};
for ir = 1:numel(referenceMeta)
    meta = referenceMeta{ir};
    rows(end+1, :) = {window, 'Reference', ir, NaN, NaN, ...
                      meta.TargetN, meta.NumAttempts, meta.NumRejected, ...
                      meta.AcceptanceRatio, meta.SeedBase}; %#ok<AGROW>
end

for it = 1:numel(testNsims)
    for irpt = 1:size(smallMeta, 2)
        meta = smallMeta{it, irpt};
        rows(end+1, :) = {window, 'SmallRun', NaN, testNsims(it), irpt, ...
                          meta.TargetN, meta.NumAttempts, meta.NumRejected, ...
                          meta.AcceptanceRatio, meta.SeedBase}; %#ok<AGROW>
    end
end

tbl = cell2table(rows, 'VariableNames', ...
    {'Window', 'EnsembleType', 'ReferenceId', 'Nsim', 'Repeat', ...
     'TargetN', 'NumAttempts', 'NumRejected', 'AcceptanceRatio', 'SeedBase'});
end


function seed = makeReferenceSeed(baseSeed, windowId, refId)
% Deterministic seed for each reference ensemble.

seed = baseSeed + 100000000*windowId + 1000000*refId;
end


function seed = makeSmallRunSeed(baseSeed, windowId, testId, repeatId)
% Deterministic seed for each repeated small ensemble.

seed = baseSeed + 100000000*windowId + 1000000*(100 + testId) + 10000*repeatId;
end


function [perms, meta, loaded] = loadEnsembleCheckpoint(filePath, expected)
% Load a saved raw-ensemble checkpoint if it matches the current request.

perms = [];
meta = struct();
loaded = false;

if ~exist(filePath, 'file')
    return
end

try
    S = load(filePath, 'perms', 'meta', 'checkpointInfo');
catch ME
    warning('Could not load checkpoint %s (%s). Regenerating it.', filePath, ME.message)
    return
end

if ~isfield(S, 'perms') || ~isfield(S, 'meta') || ~isfield(S, 'checkpointInfo')
    warning('Checkpoint %s is missing required variables. Regenerating it.', filePath)
    return
end

if ~isCheckpointCompatible(S.checkpointInfo, expected)
    return
end

if size(S.perms, 1) ~= expected.TargetN || size(S.perms, 2) ~= 3
    warning('Checkpoint %s has an unexpected permeability array size. Regenerating it.', filePath)
    return
end

if any(~isfinite(S.perms(:))) || any(S.perms(:) <= 0)
    warning('Checkpoint %s contains invalid permeability values. Regenerating it.', filePath)
    return
end

perms = S.perms;
meta = S.meta;
loaded = true;
end


function tf = isCheckpointCompatible(checkpointInfo, expected)
% Check whether a checkpoint matches the current ensemble request.

requiredFields = {'Kind', 'Window', 'TargetN', 'SeedBase', 'CorrCoef', ...
                  'ReferenceId', 'Nsim', 'Repeat'};
for i = 1:numel(requiredFields)
    if ~isfield(checkpointInfo, requiredFields{i})
        tf = false;
        return
    end
end

tf = strcmpi(string(checkpointInfo.Kind), string(expected.Kind)) && ...
     strcmpi(string(checkpointInfo.Window), string(expected.Window)) && ...
     isequaln(checkpointInfo.TargetN, expected.TargetN) && ...
     isequaln(checkpointInfo.SeedBase, expected.SeedBase) && ...
     isequaln(checkpointInfo.CorrCoef, expected.CorrCoef) && ...
     isequaln(checkpointInfo.ReferenceId, expected.ReferenceId) && ...
     isequaln(checkpointInfo.Nsim, expected.Nsim) && ...
     isequaln(checkpointInfo.Repeat, expected.Repeat);
end


function saveEnsembleCheckpoint(filePath, perms, meta, checkpointInfo)
% Save a raw-ensemble checkpoint atomically to support restart/resume.

checkpointInfo.CompletedOn = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
checkpointInfo.NumReturned = size(perms, 1);

folderPath = fileparts(filePath);
ensureFolder(folderPath);
tmpFilePath = [filePath '.tmp'];
if exist(tmpFilePath, 'file')
    delete(tmpFilePath);
end

save(tmpFilePath, 'perms', 'meta', 'checkpointInfo', '-v7.3');
movefile(tmpFilePath, filePath, 'f');
end


function saveFigurePair(figHandle, basePath)
% Save both PNG and FIG versions of a figure.

savefig(figHandle, [basePath '.fig']);
exportgraphics(figHandle, [basePath '.png'], 'Resolution', 200);
end


function ensureFolder(folderPath)
% Create a folder if needed.

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
