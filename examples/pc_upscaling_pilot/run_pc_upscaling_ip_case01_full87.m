%RUN_PC_UPSCALING_IP_CASE01_FULL87 Invasion-percolation Pc upscaling.
%
% This script uses the same replayed PREDICT realizations as the full-87
% median-sand-ratio Pc pilot, but computes Pc curves with a connectivity
% invasion-percolation calculation equivalent to the original t = 1
% threshold sweep:
%
%   geology: s05_c012
%   scenario: medium sand, nonuniform
%   geologic case: case_012_zf0500_svcl010_cvcl060
%   Level-3 cases: 01, 03, 04, and 07
%
% The material Pc functions are kept deterministic and consistent with the
% calibrated ordinary test: sand is Leverett-scaled from the reference sand
% curve, and clay is scaled from the GoM clay Pce(log10(k)) model at the
% median uncertainty quantile. The key change is the Pc upscaling mechanism:
% gas saturation only increases after a connected invasion path forms.
%
% For a quick smoke test, run:
%   setenv('PC_IP_MAX_ROWS','1')
%   run_pc_upscaling_ip_case01_full87
%
% For the full case, clear that variable:
%   setenv('PC_IP_MAX_ROWS','')
%   run_pc_upscaling_ip_case01_full87

clear; clc;

scriptDir = fileparts(mfilename('fullpath'));
examplesDir = fileparts(scriptDir);
repoRoot = fileparts(examplesDir);

cfg = struct();
cfg.geologyId = "s05_c012";
cfg.level3CaseIds = [1 3 4 7];
cfg.caseToken = "cases_01_03_04_07";
cfg.windows = ["famp1", "famp2", "famp3", "famp4", "famp5", "famp6"];
cfg.sgGrid = linspace(0.02, 0.68, 80);
cfg.sourceRoot = fullfile('D:', 'codex_gom', 'UQ_workflow', ...
    'pc_upscaling_median_examples_full87');
cfg.replaySummaryCsv = fullfile(cfg.sourceRoot, 'tables', ...
    'replay_summary_with_full87_context_s05_c012_cases_01_03_04_07.csv');
cfg.calibratedOrdinaryRoot = fullfile('D:', 'codex_gom', 'UQ_workflow', ...
    'pc_upscaling_calibrated_median_examples_full87');
cfg.upscalingZip = fullfile(repoRoot, 'upscaling.zip');
cfg.upscalingRoot = fullfile('D:', 'codex_gom', 'tmp_upscaling_zip_inspect');
cfg.mrstRoot = fullfile('C:', 'Users', 'Shaow', 'OneDrive', 'MIT', ...
    'mrst-2025a', 'SINTEF-AppliedCompSci-MRST-75749fa');
cfg.outputRoot = fullfile('D:', 'codex_gom', 'UQ_workflow', ...
    'pc_upscaling_ip_median_examples_full87');

maxRowsText = strtrim(string(getenv('PC_IP_MAX_ROWS')));
if maxRowsText ~= ""
    cfg.maxRows = str2double(maxRowsText);
else
    cfg.maxRows = inf;
end
if isfinite(cfg.maxRows)
    cfg.outputRoot = cfg.outputRoot + "_smoke" + string(cfg.maxRows);
end

cfg.curveDir = fullfile(cfg.outputRoot, 'curves');
cfg.tableDir = fullfile(cfg.outputRoot, 'tables');
cfg.figureDir = fullfile(cfg.outputRoot, 'figures');
ensureFolder(cfg.curveDir);
ensureFolder(cfg.tableDir);
ensureFolder(cfg.figureDir);

initializeIpPaths(cfg);
cfg.originalDeckFile = resolveOriginalDeckFile(cfg);

fprintf('\n=== Load replay summary for full IP cases %s Pc upscaling ===\n', ...
    cfg.caseToken)
assert(exist(cfg.replaySummaryCsv, 'file') == 2, ...
    'Missing replay summary: %s', cfg.replaySummaryCsv);
replaySummaryAll = readtable(cfg.replaySummaryCsv, 'TextType', 'string');
caseMask = replaySummaryAll.GeologyId == cfg.geologyId & ...
    ismember(replaySummaryAll.Level3CaseId, cfg.level3CaseIds);
replaySummary = replaySummaryAll(caseMask, :);
replaySummary.WindowOrder = windowOrder(replaySummary.Window);
replaySummary = sortrows(replaySummary, {'Level3CaseId', 'SliceIndex', 'WindowOrder'});

expectedRows = 87 * numel(cfg.windows) * numel(cfg.level3CaseIds);
assert(height(replaySummary) == expectedRows, ...
    'Expected %d rows for cases %s, found %d.', ...
    expectedRows, cfg.caseToken, height(replaySummary));
if isfinite(cfg.maxRows)
    replaySummary = replaySummary(1:min(cfg.maxRows, height(replaySummary)), :);
end
fprintf('Using %d replayed rows for cases %s from: %s\n', ...
    height(replaySummary), cfg.caseToken, cfg.replaySummaryCsv);

fprintf('\n=== Prepare deterministic material Pc scaling ===\n')
pcOpt = ipPcOptions(cfg.originalDeckFile, cfg.sgGrid);
fprintf('Reference deck: %s\n', cfg.originalDeckFile);
fprintf('IP mode uses kzz scaling, %s connectivity, t = %.2f, clay Pce uncertainty quantile %.2f.\n', ...
    pcOpt.ipAlgorithm, pcOpt.t, pcOpt.clayPceUncertaintyQuantile);

curveLongCsv = fullfile(cfg.curveDir, ...
    sprintf('pc_curve_points_s05_c012_%s_ip_full87.csv', cfg.caseToken));
curveSummaryCsv = fullfile(cfg.tableDir, ...
    sprintf('pc_curve_summary_s05_c012_%s_ip_full87.csv', cfg.caseToken));
curveMatFile = fullfile(cfg.curveDir, ...
    sprintf('pc_curves_s05_c012_%s_ip_full87.mat', cfg.caseToken));

fprintf('\n=== Compute full invasion-percolation Pc curves ===\n')
if exist(curveMatFile, 'file') == 2
    fprintf('Loading cached IP curve MAT: %s\n', curveMatFile);
    cached = load(curveMatFile, 'curveMat');
    curveMat = cached.curveMat;
else
    [curveLong, curveSummary, curveMat] = computeIpPcCurves( ...
        replaySummary, pcOpt);
    writetable(curveLong, curveLongCsv);
    writetable(curveSummary, curveSummaryCsv);
    save(curveMatFile, 'curveMat', 'pcOpt', 'cfg', '-v7.3');
    fprintf('Saved IP curve points: %s\n', curveLongCsv);
    fprintf('Saved IP curve summary: %s\n', curveSummaryCsv);
    fprintf('Saved IP curve MAT: %s\n', curveMatFile);
end

fprintf('\n=== Select IP medoid Pc curves ===\n')
results = analyzeIpMedoids(curveMat, cfg);
medoidSummaryCsv = fullfile(cfg.tableDir, ...
    sprintf('pc_medoid_summary_s05_c012_%s_ip_full87.csv', cfg.caseToken));
distanceSummaryCsv = fullfile(cfg.tableDir, ...
    sprintf('pc_distance_summary_s05_c012_%s_ip_full87.csv', cfg.caseToken));
writetable(results.MedoidSummary, medoidSummaryCsv);
writetable(results.DistanceSummary, distanceSummaryCsv);
save(fullfile(cfg.tableDir, ...
    sprintf('pc_medoid_results_s05_c012_%s_ip_full87.mat', cfg.caseToken)), ...
    'results', '-v7.3');
fprintf('Saved IP medoid summary: %s\n', medoidSummaryCsv);
fprintf('Saved IP distance summary: %s\n', distanceSummaryCsv);

fprintf('\n=== Generate IP figures and ordinary-vs-IP comparison ===\n')
makeIpFigures(curveMat, results, cfg);
compareIpAndOrdinaryIfAvailable(curveMat, results, cfg);

fprintf('\nFull invasion-percolation cases %s Pc upscaling complete.\n', ...
    cfg.caseToken)
fprintf('Output root: %s\n', cfg.outputRoot);


function initializeIpPaths(cfg)
% Add MRST and original upscaling helper paths used by upscalePcReg.

assert(exist(cfg.upscalingRoot, 'dir') == 7, ...
    'Missing extracted upscaling root: %s', cfg.upscalingRoot);
if exist(fullfile(cfg.mrstRoot, 'startup.m'), 'file') == 2
    run(fullfile(cfg.mrstRoot, 'startup.m'));
else
    warning('MRST startup not found. Continuing with current MATLAB path: %s', ...
        cfg.mrstRoot);
end
addpath(cfg.upscalingRoot);
addpath(fullfile(cfg.upscalingRoot, 'upscaling'));
end


function pcOpt = ipPcOptions(deckFile, sgGrid)
% Options for deterministic calibrated material curves plus IP upscaling.

pcOpt = struct();
pcOpt.sgGrid = sgGrid;
pcOpt.minPoro = 1.0e-4;
pcOpt.minPermMD = 1.0e-9;
pcOpt.mDInM2 = 9.869233e-16;
pcOpt.refPermSandSI = 7.60393535652603e-13;
pcOpt.refPoroSand = 0.289875;
pcOpt.clayPceRmse = 0.2953;
pcOpt.clayPceUncertaintyQuantile = 0.5;
pcOpt.contactAngleDeg = 30;
pcOpt.scalingPermComponent = "kzz";
pcOpt.t = 1;
pcOpt.ipAlgorithm = "fast_threshold_connectivity";
pcOpt.deckFile = deckFile;
pcOpt.referenceCurves = readSgofReferenceCurves(deckFile);
end


function deckFile = resolveOriginalDeckFile(cfg)
% Locate the original upscaling deck, extracting upscaling.zip if needed.

deckRelPath = fullfile('eclipse_data_files', ...
    'gom_forUps_theta30_PVDO_incompRock.DATA');

candidate = fullfile(cfg.upscalingRoot, deckRelPath);
if exist(candidate, 'file') == 2
    deckFile = candidate;
    return
end

extractRoot = fullfile(cfg.outputRoot, 'reference_upscaling_zip');
candidate = fullfile(extractRoot, deckRelPath);
if exist(candidate, 'file') ~= 2
    assert(exist(cfg.upscalingZip, 'file') == 2, ...
        'Missing original upscaling deck and upscaling.zip: %s', cfg.upscalingZip);
    ensureFolder(extractRoot);
    unzip(cfg.upscalingZip, extractRoot);
end

assert(exist(candidate, 'file') == 2, ...
    'Could not locate extracted deck file: %s', candidate);
deckFile = candidate;
end


function curves = readSgofReferenceCurves(deckFile)
% Read the two original SGOF Pc tables from the Eclipse deck.

assert(exist(deckFile, 'file') == 2, 'Missing deck file: %s', deckFile);
lines = string(readlines(deckFile));
sgofLine = find(strtrim(lines) == "SGOF", 1, 'first');
assert(~isempty(sgofLine), 'No SGOF keyword found in %s.', deckFile);

tables = cell(1, 2);
current = [];
tableIndex = 1;
for i = (sgofLine + 1):numel(lines)
    raw = strtrim(lines(i));
    if raw == "" || startsWith(raw, "--")
        continue
    end
    if raw == "/"
        if ~isempty(current)
            tables{tableIndex} = current;
            tableIndex = tableIndex + 1;
            current = [];
            if tableIndex > 2
                break
            end
        end
        continue
    end

    values = sscanf(erase(raw, "/"), '%f');
    if numel(values) == 4
        current(end+1, :) = values(:)'; %#ok<AGROW>
    elseif ~isempty(current)
        break
    end
end

assert(~isempty(tables{1}) && ~isempty(tables{2}), ...
    'Could not parse two SGOF tables from %s.', deckFile);
curves.sand = tableToCurve(tables{1});
curves.clay = tableToCurve(tables{2});
end


function curve = tableToCurve(T)
% Convert one SGOF numeric table to an interpolation-ready Pc curve.

curve.sg = T(:, 1);
curve.krg = T(:, 2);
curve.krog = T(:, 3);
curve.pcBar = T(:, 4);
curve.pcPa = curve.pcBar * 1.0e5;
[curve.pcPaUnique, ia] = unique(curve.pcPa, 'stable');
curve.sgAtPcUnique = curve.sg(ia);
end


function [curveLong, curveSummary, curveMat] = computeIpPcCurves(replaySummary, pcOpt)
% Compute invasion-percolation Pc curves for every replay row.

n = height(replaySummary);
sgGrid = pcOpt.sgGrid(:)';
pcPa = nan(n, numel(sgGrid));
summaryRows = cell(n, 34);
longRows = cell(n * numel(sgGrid), 18);
longIdx = 0;

for i = 1:n
    outputFile = char(replaySummary.OutputFile(i));
    fprintf('IP Pc curve %3d/%3d: case %02d slice %02d %s\n', ...
        i, n, replaySummary.Level3CaseId(i), ...
        replaySummary.SliceIndex(i), char(replaySummary.Window(i)));
    S = load(outputFile, 'replay');
    curve = ipPcCurveFromReplay(S.replay, pcOpt, replaySummary.Window(i));
    pcPa(i, :) = curve.pcPa;

    summaryRows(i, :) = { ...
        i, replaySummary.SourceRow(i), replaySummary.GeologyId(i), ...
        replaySummary.ScenarioName(i), replaySummary.CaseLabel(i), ...
        replaySummary.Level3CaseId(i), replaySummary.Level3CaseName(i), ...
        replaySummary.Window(i), replaySummary.SliceIndex(i), ...
        replaySummary.AssignedState(i), replaySummary.SamplingPool(i), ...
        replaySummary.SelectedSampleIndex(i), replaySummary.ReplaySeed(i), ...
        curve.poreVolume, curve.numCells, curve.numRegions, ...
        curve.numClayRegions, curve.clayPoreVolumeFraction, ...
        curve.meanLog10KzzMD, curve.medianLog10KzzMD, ...
        curve.p05Log10KzzMD, curve.p95Log10KzzMD, ...
        curve.pcAtSg20Pa, curve.pcAtSg50Pa, curve.pcAtSg65Pa, ...
        curve.bulkSgMax, curve.rawNumPoints, curve.percolationPcPa, ...
        curve.percolationSg, curve.pcMinPa, curve.pcMaxPa, ...
        pcOpt.scalingPermComponent, pcOpt.t, ...
        pcOpt.clayPceUncertaintyQuantile};

    for j = 1:numel(sgGrid)
        longIdx = longIdx + 1;
        longRows(longIdx, :) = { ...
            i, replaySummary.SourceRow(i), replaySummary.GeologyId(i), ...
            replaySummary.Level3CaseId(i), replaySummary.Level3CaseName(i), ...
            replaySummary.Window(i), replaySummary.SliceIndex(i), ...
            replaySummary.AssignedState(i), replaySummary.SamplingPool(i), ...
            replaySummary.SelectedSampleIndex(i), replaySummary.ReplaySeed(i), ...
            sgGrid(j), curve.pcPa(j), curve.pcPa(j) / 1.0e5, ...
            log10(max(curve.pcPa(j), realmin)), curve.clayPoreVolumeFraction, ...
            curve.medianLog10KzzMD, curve.percolationPcPa / 1.0e5};
    end
end

curveSummary = cell2table(summaryRows, 'VariableNames', ...
    {'CurveId', 'ReplaySourceRow', 'GeologyId', 'ScenarioName', ...
     'CaseLabel', 'Level3CaseId', 'Level3CaseName', 'Window', ...
     'SliceIndex', 'AssignedState', 'SamplingPool', 'SelectedSampleIndex', ...
     'ReplaySeed', 'PoreVolume', 'NumCells', 'NumRegions', ...
     'NumClayRegions', 'ClayPoreVolumeFraction', 'MeanLog10KzzMD', ...
     'MedianLog10KzzMD', 'P05Log10KzzMD', 'P95Log10KzzMD', ...
     'PcAtSg20Pa', 'PcAtSg50Pa', 'PcAtSg65Pa', 'BulkSgMax', ...
     'RawNumPoints', 'PercolationPcPa', 'PercolationSg', ...
     'PcMinPa', 'PcMaxPa', 'ScalingPermComponent', 'IpT', ...
     'ClayPceUncertaintyQuantile'});

curveLong = cell2table(longRows(1:longIdx, :), 'VariableNames', ...
    {'CurveId', 'ReplaySourceRow', 'GeologyId', 'Level3CaseId', ...
     'Level3CaseName', 'Window', 'SliceIndex', 'AssignedState', ...
     'SamplingPool', 'SelectedSampleIndex', 'ReplaySeed', ...
     'GasSaturation', 'PcPa', 'PcBar', 'Log10PcPa', ...
     'ClayPoreVolumeFraction', 'MedianLog10KzzMD', ...
     'PercolationPcBar'});

curveMat = struct();
curveMat.sgGrid = sgGrid;
curveMat.pcPa = pcPa;
curveMat.pcBar = pcPa / 1.0e5;
curveMat.summary = curveSummary;
end


function curve = ipPcCurveFromReplay(replay, pcOpt, windowName)
% Compute one invasion-percolation upscaled Pc curve from replay data.

[G, rock, fluid, opt, diagnostics] = buildIpInputsFromReplay( ...
    replay, pcOpt, windowName);
[pcRaw, sgRaw, ipDiagnostics] = fastInvasionPercolationPcReg( ...
    G, fluid, rock, opt);

valid = isfinite(sgRaw(:)) & isfinite(pcRaw(:)) & pcRaw(:) >= 0;
sgRaw = sgRaw(valid);
pcRaw = pcRaw(valid);
[sgRaw, order] = sort(sgRaw(:));
pcRaw = pcRaw(order);
[sgUnique, ia] = unique(sgRaw, 'stable');
pcUnique = pcRaw(ia);
sgUnique = sgUnique(:);
pcUnique = pcUnique(:);

if numel(sgUnique) < 2
    error('IP curve has fewer than two unique saturation points.');
end

pcAtSg = interp1(sgUnique, pcUnique, pcOpt.sgGrid, 'linear', 'extrap');
pcAtSg = max(pcAtSg, realmin);

positive = find(pcUnique > 0 & sgUnique > 0, 1, 'first');
if isempty(positive)
    percolationPc = NaN;
    percolationSg = NaN;
else
    percolationPc = pcUnique(positive);
    percolationSg = sgUnique(positive);
end

curve = diagnostics;
curve.rawSg = sgUnique;
curve.rawPcPa = pcUnique;
curve.pcPa = pcAtSg;
curve.rawNumPoints = numel(sgUnique);
curve.percolationPcPa = percolationPc;
curve.percolationSg = percolationSg;
curve.firstBottomConnectedPcPa = ipDiagnostics.firstBottomConnectedPcPa;
curve.firstBottomConnectedSg = ipDiagnostics.firstBottomConnectedSg;
curve.pcAtSg20Pa = interp1(pcOpt.sgGrid, pcAtSg, 0.20, 'linear', 'extrap');
curve.pcAtSg50Pa = interp1(pcOpt.sgGrid, pcAtSg, 0.50, 'linear', 'extrap');
curve.pcAtSg65Pa = interp1(pcOpt.sgGrid, pcAtSg, 0.65, 'linear', 'extrap');
curve.bulkSgMax = max(sgUnique);
curve.pcMinPa = min(pcUnique);
curve.pcMaxPa = max(pcUnique);
end


function [pcVal, sVal, diagnostics] = fastInvasionPercolationPcReg(G, fluid, rock, opt)
% Connectivity-threshold implementation of t=1 invasion percolation.
%
% A cell is invaded at pressure pc if it is connected to the inlet boundary
% through cells with entry pressures <= pc. The bulk saturation is kept at
% zero until an invaded path first reaches the outlet boundary, matching the
% original upscalePcReg inv-per convention.

reg = rock.regions.saturation;
idReg = unique(reg);
nreg = numel(idReg);

fluid.pcInv = cell(1, max(reg));
pcMin = inf;
pcMax = 0;
pcv2 = inf;
sgmax = zeros(1, nreg);
pce = zeros(G.cells.num, 1);
pcAtSgMax = zeros(1, max(idReg));

for n = 1:nreg
    regId = idReg(n);
    if strcmp(opt.sg, 'sandClay')
        if fluid.isclay(n)
            sgmin = fluid.krPts.g(2, 1);
            sgmax(n) = fluid.krPts.g(2, 3);
        else
            sgmin = fluid.krPts.g(1, 1);
            sgmax(n) = fluid.krPts.g(1, 3) - 0.01;
        end
    else
        sgmin = fluid.krPts.g(n, 1);
        sgmax(n) = fluid.krPts.g(n, 3);
    end

    sgvals = linspace(sgmin, sgmax(n), pow2(6)-1)';
    sgvals = [sgvals(1); sgvals(1)+1e-3; sgvals(2:end)];
    pcvals = fluid.pcOG{regId}(sgvals);
    pcvals = makeStrictlyIncreasing(pcvals);
    pcv2 = min(pcv2, pcvals(2));
    fluid.pcInv{regId} = @(pcOG) interp1(pcvals, sgvals, pcOG, ...
        'linear', 'extrap');
    pcMin = min(pcMin, fluid.pcOG{regId}(sgmin));
    pcAtSgMax(regId) = fluid.pcOG{regId}(sgmax(n));
    if fluid.isclay(n)
        pcMax = max(pcMax, pcAtSgMax(regId));
    end
    pce(reg == regId) = fluid.pcOG{regId}(1e-3);
end

if ~(isfinite(pcMax) && pcMax > pcv2)
    pcMax = max(pcAtSgMax(pcAtSgMax > 0));
end
pcVal = logspace(log10(pcv2), log10(0.99 * pcMax), pow2(6)-2);
pcVal = [0, 0.98 * pcv2, pcVal];

topCells = (1:(G.cartDims(1) * G.cartDims(2)))';
bottomCells = ((G.cells.num - G.cartDims(1) * G.cartDims(2) + 1):G.cells.num)';
adj = buildCellAdjacency(G);
volume = sum(G.cells.volumes .* rock.poro);
sVal = zeros(numel(pcVal), 1);
sgCells = zeros(G.cells.num, 1);
idrem = false(numel(pcVal), 1);
idPercolation = [];
sLast = 0;
firstBottomPc = NaN;
firstBottomSg = NaN;

for k = 2:numel(pcVal)
    pcv = pcVal(k);
    openMask = pce <= pcv;
    invaded = connectedOpenCells(adj, openMask, topCells);
    if any(invaded(bottomCells)) && isempty(idPercolation)
        idPercolation = k;
        firstBottomPc = pcv;
    end

    invadedIds = find(invaded);
    for n = 1:nreg
        regId = idReg(n);
        cells = invadedIds(reg(invadedIds) == regId);
        if ~isempty(cells)
            pcvId = min(pcv, pcAtSgMax(regId));
            sgCells(cells) = fluid.pcInv{regId}(pcvId * ones(numel(cells), 1));
        end
    end

    if isempty(idPercolation)
        sVal(k) = 0;
    else
        sVal(k) = sum(sgCells .* G.cells.volumes .* rock.poro, 'omitnan') / volume;
    end

    if abs(sVal(k) - sLast) < 1e-3
        idrem(k) = true;
    else
        sLast = sVal(k);
    end
end

if isempty(idPercolation)
    error('No percolating path found in IP Pc upscaling.');
end
sVal(idPercolation - 1) = 1e-5;
idrem(idPercolation - 1) = false;
idrem(idPercolation) = false;
idrem(end) = false;
pcVal = pcVal(:);
sVal = sVal(:);
pcVal(idrem(:)) = [];
sVal(idrem(:)) = [];
firstBottomSg = sVal(find(sVal > 1e-5, 1, 'first'));

diagnostics = struct();
diagnostics.pcMinPa = pcMin;
diagnostics.pcMaxPa = pcMax;
diagnostics.firstBottomConnectedPcPa = firstBottomPc;
diagnostics.firstBottomConnectedSg = firstBottomSg;
end


function adj = buildCellAdjacency(G)
% Build an undirected cell-neighbor list from MRST face neighbors.

neighbors = G.faces.neighbors;
neighbors = neighbors(all(neighbors > 0, 2), :);
n = G.cells.num;
counts = accumarray([neighbors(:, 1); neighbors(:, 2)], 1, [n, 1]);
adj = cell(n, 1);
for i = 1:n
    adj{i} = zeros(counts(i), 1);
end
fill = zeros(n, 1);
for e = 1:size(neighbors, 1)
    a = neighbors(e, 1);
    b = neighbors(e, 2);
    fill(a) = fill(a) + 1;
    adj{a}(fill(a)) = b;
    fill(b) = fill(b) + 1;
    adj{b}(fill(b)) = a;
end
end


function visited = connectedOpenCells(adj, openMask, inletCells)
% Return open cells connected to the inlet boundary.

n = numel(openMask);
visited = false(n, 1);
queue = zeros(n, 1);
start = inletCells(openMask(inletCells));
if isempty(start)
    return
end
tail = numel(start);
queue(1:tail) = start;
visited(start) = true;
head = 1;
while head <= tail
    c = queue(head);
    head = head + 1;
    nb = adj{c};
    nb = nb(openMask(nb) & ~visited(nb));
    if ~isempty(nb)
        visited(nb) = true;
        queue((tail+1):(tail+numel(nb))) = nb;
        tail = tail + numel(nb);
    end
end
end


function [G, rock, fluid, opt, diagnostics] = buildIpInputsFromReplay( ...
        replay, pcOpt, windowName)
% Build MRST-style G/rock/fluid/opt inputs for original IP upscaling.

G = replay.G;
grid = replay.Grid;
poroAll = max(grid.poro(:), pcOpt.minPoro);
perm = grid.perm;
units = grid.units(:);
isSmear = logical(grid.isSmear(:));
volume = G.cells.volumes(:);
permComponentSI = selectPermComponent(perm, pcOpt);
permComponentSI = max(permComponentSI(:), pcOpt.minPermMD * pcOpt.mDInM2);
log10KzzMD = log10(permComponentSI ./ pcOpt.mDInM2);

valid = isfinite(poroAll) & isfinite(permComponentSI) & ...
    isfinite(volume) & volume > 0 & units > 0;
assert(all(valid), 'IP runner expects valid replay cells for all fault-core cells.');

rock = struct();
rock.poro = poroAll;
rock.perm = perm;
rock.regions = struct();
rock.regions.saturation = units;
rock.regions.rocknum = ones(G.cells.num, 1);
rock.regions.rocknum(isSmear) = 2;

idReg = unique(units(:))';
fluid = struct();
fluid.krPts = struct();
fluid.krPts.g = [ ...
    pcOpt.referenceCurves.sand.sg(1), NaN, pcOpt.referenceCurves.sand.sg(end); ...
    pcOpt.referenceCurves.clay.sg(1), NaN, pcOpt.referenceCurves.clay.sg(end)];
fluid.pcOG = cell(1, max(idReg));
fluid.isclay = false(1, numel(idReg));

for n = 1:numel(idReg)
    id = units == idReg(n);
    smearFrac = mean(double(isSmear(id)), 'omitnan');
    isClay = smearFrac >= 0.5;
    fluid.isclay(n) = isClay;
    region = struct();
    region.isClay = isClay;
    region.permSI = mean(permComponentSI(id), 'omitnan');
    region.poro = mean(poroAll(id), 'omitnan');
    [sgRef, pcRefPa] = scaledRegionPc(region, pcOpt);
    fluid.pcOG{idReg(n)} = @(sg) interp1(sgRef, pcRefPa, sg, ...
        'linear', 'extrap');
end

opt = struct();
opt.pc_mode = 'inv-per';
opt.t = pcOpt.t;
opt.sg = 'sandClay';
opt.window = char(windowName);
opt.fault = 'predict_3D';
opt.dir = 'z';
opt.theta = [pcOpt.contactAngleDeg, pcOpt.contactAngleDeg];

poreWeights = poroAll .* volume;
diagnostics = struct();
diagnostics.poreVolume = sum(poreWeights, 'omitnan');
diagnostics.numCells = G.cells.num;
diagnostics.numRegions = numel(idReg);
diagnostics.numClayRegions = sum(fluid.isclay);
diagnostics.clayPoreVolumeFraction = ...
    sum(poreWeights(isSmear), 'omitnan') ./ max(sum(poreWeights, 'omitnan'), realmin);
diagnostics.meanLog10KzzMD = mean(log10KzzMD, 'omitnan');
diagnostics.medianLog10KzzMD = median(log10KzzMD, 'omitnan');
diagnostics.p05Log10KzzMD = prctile(log10KzzMD, 5);
diagnostics.p95Log10KzzMD = prctile(log10KzzMD, 95);
end


function [sgRef, pcPa] = scaledRegionPc(region, pcOpt)
% Return scaled material Pc curve for one material region.

if region.isClay
    ref = pcOpt.referenceCurves.clay;
    log10KMD = log10(max(region.permSI / pcOpt.mDInM2, pcOpt.minPermMD));
    pceHgBar = 10.^(-0.1992 * log10KMD + 1.407 - pcOpt.clayPceRmse + ...
        pcOpt.clayPceUncertaintyQuantile * 2 * pcOpt.clayPceRmse);
    pceCo2WaterPa = 1.0e5 * pceHgBar * ...
        abs(cosd(pcOpt.contactAngleDeg) * 25 / (cosd(140) * 485));
    entryRefPa = interp1(ref.sg, ref.pcPa, 0.10, 'linear', 'extrap');
    pcPa = ref.pcPa .* (pceCo2WaterPa ./ entryRefPa);
else
    ref = pcOpt.referenceCurves.sand;
    scale = sqrt((pcOpt.refPermSandSI * region.poro) ./ ...
        (pcOpt.refPoroSand * region.permSI));
    pcPa = ref.pcPa .* scale;
end
sgRef = ref.sg;
pcPa = makeStrictlyIncreasing(pcPa);
end


function pcPa = makeStrictlyIncreasing(pcPa)
% Add tiny monotonic increments so inverse interpolation remains stable.

pcPa = pcPa(:);
for i = 2:numel(pcPa)
    if pcPa(i) <= pcPa(i-1)
        pcPa(i) = pcPa(i-1) + max(abs(pcPa(i-1)) * 1e-9, 1e-6);
    end
end
end


function kSI = selectPermComponent(perm, pcOpt)
% Select the permeability component used for Pc scaling.

switch lower(string(pcOpt.scalingPermComponent))
    case "kxx"
        col = 1;
    case "kyy"
        if size(perm, 2) >= 4
            col = 4;
        elseif size(perm, 2) >= 2
            col = 2;
        else
            col = 1;
        end
    case "kzz"
        if size(perm, 2) >= 6
            col = 6;
        elseif size(perm, 2) >= 3
            col = 3;
        else
            col = 1;
        end
    otherwise
        error('Unknown permeability component: %s', pcOpt.scalingPermComponent);
end
kSI = perm(:, col);
end


function results = analyzeIpMedoids(curveMat, cfg)
% Select one medoid IP curve per Level-3 case and window.

summary = curveMat.summary;
windows = cfg.windows;
medoidRows = {};
distanceRows = {};

for c = 1:numel(cfg.level3CaseIds)
    caseId = cfg.level3CaseIds(c);
    for w = 1:numel(windows)
        windowName = windows(w);
        idx = find(summary.Level3CaseId == caseId & summary.Window == windowName);
        if isfinite(cfg.maxRows)
            if isempty(idx)
                continue
            end
        else
            assert(numel(idx) == 87, ...
                'Expected 87 curves for case %02d, %s; found %d.', ...
                caseId, windowName, numel(idx));
        end

        curves = curveMat.pcPa(idx, :);
        logCurves = log10(max(curves, realmin));
        distances = pairwiseRmsDistance(logCurves);
        meanDistance = mean(distances, 2, 'omitnan');
        [minMeanDistance, localMedoid] = min(meanDistance);
        medoidCurveId = idx(localMedoid);
        upper = distances(triu(true(size(distances)), 1));

        medoidRows(end+1, :) = { ...
            summary.GeologyId(idx(1)), caseId, ...
            summary.Level3CaseName(idx(1)), windowName, medoidCurveId, ...
            summary.SliceIndex(medoidCurveId), ...
            summary.SelectedSampleIndex(medoidCurveId), ...
            summary.ReplaySeed(medoidCurveId), ...
            minMeanDistance, median(upper, 'omitnan'), ...
            prctile(upper, 95), summary.PcAtSg20Pa(medoidCurveId), ...
            summary.PcAtSg50Pa(medoidCurveId), ...
            summary.PcAtSg65Pa(medoidCurveId), ...
            summary.PercolationPcPa(medoidCurveId), ...
            summary.PercolationSg(medoidCurveId), ...
            summary.ClayPoreVolumeFraction(medoidCurveId), ...
            summary.MedianLog10KzzMD(medoidCurveId)};

        for j = 1:numel(idx)
            distanceRows(end+1, :) = { ...
                summary.GeologyId(idx(j)), caseId, ...
                summary.Level3CaseName(idx(j)), windowName, ...
                summary.SliceIndex(idx(j)), summary.SelectedSampleIndex(idx(j)), ...
                summary.ReplaySeed(idx(j)), idx(j), medoidCurveId, ...
                distances(j, localMedoid), meanDistance(j)};
        end
    end
end

results = struct();
results.MedoidSummary = cell2table(medoidRows, 'VariableNames', ...
    {'GeologyId', 'Level3CaseId', 'Level3CaseName', 'Window', ...
     'MedoidCurveId', 'MedoidSliceIndex', 'MedoidSelectedSampleIndex', ...
     'MedoidReplaySeed', 'MedoidMeanRmsLog10PcDistance', ...
     'MedianPairwiseRmsLog10PcDistance', 'P95PairwiseRmsLog10PcDistance', ...
     'MedoidPcAtSg20Pa', 'MedoidPcAtSg50Pa', 'MedoidPcAtSg65Pa', ...
     'MedoidPercolationPcPa', 'MedoidPercolationSg', ...
     'MedoidClayPoreVolumeFraction', 'MedoidMedianLog10KzzMD'});
results.DistanceSummary = cell2table(distanceRows, 'VariableNames', ...
    {'GeologyId', 'Level3CaseId', 'Level3CaseName', 'Window', ...
     'SliceIndex', 'SelectedSampleIndex', 'ReplaySeed', 'CurveId', ...
     'MedoidCurveId', 'RmsLog10PcDistanceToMedoid', ...
     'MeanRmsLog10PcDistanceToAllCurves'});
end


function D = pairwiseRmsDistance(logCurves)
% Pairwise RMS distance between rows of a log-curve matrix.

n = size(logCurves, 1);
D = zeros(n, n);
for i = 1:n
    for j = (i+1):n
        diffv = logCurves(i, :) - logCurves(j, :);
        d = sqrt(mean(diffv.^2, 'omitnan'));
        D(i, j) = d;
        D(j, i) = d;
    end
end
end


function makeIpFigures(curveMat, results, cfg)
% Plot all IP full-87 curves and medoids, separated by Level-3 case.

summary = curveMat.summary;
sg = curveMat.sgGrid;
windows = cfg.windows;

for c = 1:numel(cfg.level3CaseIds)
    caseId = cfg.level3CaseIds(c);
    caseRows = find(summary.Level3CaseId == caseId);
    if isempty(caseRows)
        continue
    end
    caseName = displayCaseName(summary.Level3CaseName(caseRows(1)));
    fig = figure('Color', 'w', 'Position', [80 80 1700 900]);
    tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    for w = 1:numel(windows)
        nexttile;
        windowName = windows(w);
        idx = find(summary.Level3CaseId == caseId & summary.Window == windowName);
        if isempty(idx)
            title(sprintf('%s | no smoke data', upper(char(windowName))), ...
                'FontSize', 16, 'FontWeight', 'bold');
            axis off;
            continue
        end
        medoidMask = results.MedoidSummary.Level3CaseId == caseId & ...
            results.MedoidSummary.Window == windowName;
        medoidId = results.MedoidSummary.MedoidCurveId(medoidMask);
        semilogy(sg(:), curveMat.pcBar(idx, :)', '-', ...
            'Color', [0.72 0.74 0.78], 'LineWidth', 0.8);
        hold on;
        semilogy(sg(:), curveMat.pcBar(medoidId, :)', '-', ...
            'Color', [0.86 0.22 0.16], 'LineWidth', 2.8);
        grid on;
        title(sprintf('%s | invasion percolation', upper(char(windowName))), ...
            'FontSize', 16, 'FontWeight', 'bold');
        xlabel('Gas saturation');
        ylabel('Pc [bar]');
        xlim([min(sg), max(sg)]);
        ylim([1e-2, 1e3]);
        set(gca, 'FontSize', 13, 'LineWidth', 1.0);
    end
    sgtitle({sprintf('Case %02d: %s', caseId, caseName), ...
        'Full-87 invasion-percolation Pc curves by window', ...
        'Grey = 87 slices; red = medoid curve'}, ...
        'FontSize', 22, 'FontWeight', 'bold', 'Interpreter', 'none');
    saveFigureBoth(fig, cfg.figureDir, ...
        sprintf('s05_c012_case%02d_ip_full87_pc_curves_with_medoids', caseId));
end

fig = figure('Color', 'w', 'Position', [120 120 1750 950]);
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for c = 1:numel(cfg.level3CaseIds)
    caseId = cfg.level3CaseIds(c);
    nexttile;
    rows = results.MedoidSummary(results.MedoidSummary.Level3CaseId == caseId, :);
    if isempty(rows)
        title(sprintf('Case %02d: no smoke data', caseId), ...
            'FontSize', 15, 'FontWeight', 'bold');
        axis off;
        continue
    end
    rows.WindowOrder = windowOrder(rows.Window);
    rows = sortrows(rows, 'WindowOrder');
    barData = [rows.MedoidPcAtSg20Pa, rows.MedoidPcAtSg50Pa, ...
               rows.MedoidPcAtSg65Pa, rows.MedoidPercolationPcPa] ./ 1.0e5;
    bar(categorical(rows.Window), barData);
    grid on;
    ylabel('Medoid Pc [bar]');
    title(sprintf('Case %02d: %s', caseId, displayCaseName(rows.Level3CaseName(1))), ...
        'FontSize', 15, 'FontWeight', 'bold', 'Interpreter', 'none');
    set(gca, 'FontSize', 12, 'LineWidth', 1.0);
    if c == 1
        legend({'Sg = 0.20', 'Sg = 0.50', 'Sg = 0.65', 'IP threshold'}, ...
            'Location', 'northwest');
    end
end
sgtitle('Validated invasion-percolation medoid Pc levels by case and window', ...
    'FontSize', 22, 'FontWeight', 'bold');
saveFigureBoth(fig, cfg.figureDir, ...
    sprintf('s05_c012_%s_ip_medoid_pc_levels', cfg.caseToken));
end


function compareIpAndOrdinaryIfAvailable(ipCurveMat, ipResults, cfg)
% Compare IP medoid curves to calibrated ordinary medoid curves if present.

ordinaryMatFile = fullfile(cfg.calibratedOrdinaryRoot, 'curves', ...
    'pc_curves_s05_c012_cases_01_03_04_07_calibrated_full87.mat');
ordinaryResultsFile = fullfile(cfg.calibratedOrdinaryRoot, 'tables', ...
    'pc_medoid_results_s05_c012_cases_01_03_04_07_calibrated_full87.mat');
if exist(ordinaryMatFile, 'file') ~= 2 || exist(ordinaryResultsFile, 'file') ~= 2
    warning('Calibrated ordinary results not found; skipping IP-vs-ordinary comparison.');
    return
end

O = load(ordinaryMatFile, 'curveMat');
R = load(ordinaryResultsFile, 'results');
ordinaryMat = O.curveMat;
ordinaryResults = R.results;

availableWindows = unique(string(ipResults.MedoidSummary.Window));
if numel(intersect(availableWindows, cfg.windows)) < numel(cfg.windows)
    warning('IP medoid results are partial; skipping IP-vs-ordinary comparison.');
    return
end

rows = cell(numel(cfg.level3CaseIds) * numel(cfg.windows), 16);
row = 0;
for c = 1:numel(cfg.level3CaseIds)
    caseId = cfg.level3CaseIds(c);
    fig = figure('Color', 'w', 'Position', [80 80 1700 900]);
    tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    caseName = "";
    for w = 1:numel(cfg.windows)
        windowName = cfg.windows(w);
        ipMask = ipResults.MedoidSummary.Level3CaseId == caseId & ...
            ipResults.MedoidSummary.Window == windowName;
        ipMedoidId = ipResults.MedoidSummary.MedoidCurveId(ipMask);
        ordMask = ordinaryResults.MedoidSummary.Level3CaseId == caseId & ...
            ordinaryResults.MedoidSummary.Window == windowName;
        ordMedoidId = ordinaryResults.MedoidSummary.MedoidCurveId(ordMask);
        if isempty(ipMedoidId) || isempty(ordMedoidId)
            warning('Missing IP or ordinary medoid for case %02d %s; skipping.', ...
                caseId, windowName);
            continue
        end

        ipCurve = ipCurveMat.pcPa(ipMedoidId, :);
        ordCurve = ordinaryMat.pcPa(ordMedoidId, :);
        ipNorm = normalizeAtSg(ipCurveMat.sgGrid, ipCurve, 0.50);
        ordNorm = normalizeAtSg(ordinaryMat.sgGrid, ordCurve, 0.50);
        ordInterp = interp1(ordinaryMat.sgGrid, ordNorm, ipCurveMat.sgGrid, ...
            'linear', 'extrap');
        shapeDiff = sqrt(mean((log10(max(ipNorm, realmin)) - ...
            log10(max(ordInterp, realmin))).^2, 'omitnan'));

        ipSummary = ipCurveMat.summary(ipMedoidId, :);
        ordSummary = ordinaryMat.summary(ordMedoidId, :);
        caseName = displayCaseName(ipSummary.Level3CaseName);
        row = row + 1;
        rows(row, :) = {caseId, ipSummary.Level3CaseName, windowName, ...
            ipSummary.SliceIndex, ipSummary.SelectedSampleIndex, ...
            ipSummary.PcAtSg20Pa / 1.0e5, ipSummary.PcAtSg50Pa / 1.0e5, ...
            ipSummary.PcAtSg65Pa / 1.0e5, ipSummary.PercolationPcPa / 1.0e5, ...
            ordSummary.SliceIndex, ordSummary.SelectedSampleIndex, ...
            ordSummary.PcAtSg20Pa / 1.0e5, ordSummary.PcAtSg50Pa / 1.0e5, ...
            ordSummary.PcAtSg65Pa / 1.0e5, shapeDiff, ...
            ipSummary.SliceIndex == ordSummary.SliceIndex};

        nexttile;
        semilogy(ipCurveMat.sgGrid(:), ordInterp(:), '--', ...
            'Color', [0.13 0.38 0.67], 'LineWidth', 2.4);
        hold on;
        semilogy(ipCurveMat.sgGrid(:), ipNorm(:), '-', ...
            'Color', [0.86 0.22 0.16], 'LineWidth', 2.6);
        grid on;
        title(upper(char(windowName)), 'FontSize', 16, 'FontWeight', 'bold');
        xlabel('Gas saturation');
        ylabel('Pc / Pc(Sg=0.50)');
        xlim([min(ipCurveMat.sgGrid), max(ipCurveMat.sgGrid)]);
        set(gca, 'FontSize', 13, 'LineWidth', 1.0);
        if w == 1
            legend({'calibrated ordinary', 'invasion percolation'}, ...
                'Location', 'best');
        end
    end
    sgtitle({sprintf('Case %02d: %s', caseId, caseName), ...
        'Medoid Pc curve shape comparison', ...
        'Both curves normalized by Pc at Sg = 0.50'}, ...
        'FontSize', 21, 'FontWeight', 'bold', 'Interpreter', 'none');
    saveFigureBoth(fig, cfg.figureDir, ...
        sprintf('s05_c012_case%02d_ip_vs_calibrated_ordinary_medoid_pc_shape', caseId));
end
comparisonTable = cell2table(rows(1:row, :), 'VariableNames', ...
    {'Level3CaseId', 'Level3CaseName', 'Window', ...
     'IpMedoidSliceIndex', 'IpMedoidSelectedSampleIndex', ...
     'IpMedoidPcSg20Bar', 'IpMedoidPcSg50Bar', 'IpMedoidPcSg65Bar', ...
     'IpMedoidPercolationPcBar', 'OrdinaryMedoidSliceIndex', ...
     'OrdinaryMedoidSelectedSampleIndex', 'OrdinaryMedoidPcSg20Bar', ...
     'OrdinaryMedoidPcSg50Bar', 'OrdinaryMedoidPcSg65Bar', ...
     'RmsLog10NormalizedPcShapeDifference', 'SameMedoidSlice'});
writetable(comparisonTable, fullfile(cfg.tableDir, ...
    sprintf('pc_medoid_comparison_ip_vs_calibrated_ordinary_s05_c012_%s_full87.csv', ...
    cfg.caseToken)));
end


function y = normalizeAtSg(sg, pc, sgRef)
% Normalize a Pc curve by its value at a reference saturation.

ref = interp1(sg, pc, sgRef, 'linear', 'extrap');
y = pc ./ max(ref, realmin);
y = max(y, realmin);
end


function label = displayCaseName(rawName)
% Convert internal Level-3 case names to presentation-friendly labels.

name = lower(strtrim(string(rawName)));
switch name
    case "independent_draw_1"
        label = 'Independent draw 1';
    case "strong_fault_wide_low"
        label = 'Strong fault-wide low';
    case "strong_fault_wide_high"
        label = 'Strong fault-wide high';
    case "strong_grouped_g3_low_g4_high"
        label = 'Strong grouped G3 low / G4 high';
    otherwise
        label = char(strrep(string(rawName), "_", " "));
end
end


function order = windowOrder(windowNames)
% Numeric sorting key for famp1 ... famp6.

text = string(windowNames);
order = nan(size(text));
for i = 1:numel(text)
    digits = regexp(char(text(i)), '\d+', 'match', 'once');
    order(i) = str2double(digits);
end
end


function ensureFolder(folderPath)
% Create folder if it does not already exist.

if exist(folderPath, 'dir') ~= 7
    mkdir(folderPath);
end
end


function saveFigureBoth(fig, folderPath, baseName)
% Save one figure as PNG and PDF.

ensureFolder(folderPath);
try
    set(fig, 'DefaultAxesToolbarVisible', 'off');
catch
end
axesList = findall(fig, 'Type', 'axes');
for i = 1:numel(axesList)
    try
        disableDefaultInteractivity(axesList(i));
    catch
    end
    try
        axtoolbar(axesList(i), 'none');
    catch
    end
end
drawnow;
pngFile = fullfile(folderPath, baseName + ".png");
pdfFile = fullfile(folderPath, baseName + ".pdf");
exportgraphics(fig, pngFile, 'Resolution', 220);
exportgraphics(fig, pdfFile, 'ContentType', 'vector');
fprintf('Saved figure: %s\n', pngFile);
fprintf('Saved figure: %s\n', pdfFile);
end
