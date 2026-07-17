function validate_ip_threshold_connectivity_against_original()
%VALIDATE_IP_THRESHOLD_CONNECTIVITY_AGAINST_ORIGINAL Compare fast and original IP upscaling.
%
% This validation script checks whether the optimized threshold-connectivity
% implementation used for full Case 01 Pc upscaling reproduces the original
% upscalePcReg(..., pc_mode = 'inv-per') behavior for tractable MRST grids.
%
% The original implementation is too slow for the full 100 x 10 x 100 replay
% grids, so this script uses smaller synthetic grids with controlled material
% layouts. The validation is still apples-to-apples: both methods receive the
% same G, rock, fluid, and opt structures.

repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
outputRoot = fullfile('D:\codex_gom', 'UQ_workflow', ...
    'pc_upscaling_ip_validation');
figureDir = fullfile(outputRoot, 'figures');
tableDir = fullfile(outputRoot, 'tables');
ensureFolder(outputRoot);
ensureFolder(figureDir);
ensureFolder(tableDir);

initializePaths(repoRoot);

opt = struct();
opt.pc_mode = 'inv-per';
opt.sg = [];
opt.t = 1;
opt.nval = 101;
opt.fault = 'test';

cases = buildValidationCases();
summaryRows = cell(numel(cases), 12);

fig = figure('Color', 'w', 'Position', [80 80 1500 850]);
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:numel(cases)
    testCase = cases(i);
    fprintf('\n=== Validation case %d/%d: %s ===\n', ...
        i, numel(cases), testCase.name);

    [G, rock, fluid] = buildSyntheticProblem(testCase);

    tic;
    [pcOriginal, sgOriginal] = upscalePcReg(G, fluid, rock, opt, false);
    originalSeconds = toc;

    tic;
    [pcFast, sgFast, rawFast] = fastThresholdConnectivityUpscalePcReg( ...
        G, fluid, rock, opt);
    fastSeconds = toc;

    sgCommon = linspace(0, min(max(sgOriginal), max(sgFast)), 101);
    pcOriginalCommon = interp1(sgOriginal(:), pcOriginal(:), sgCommon, ...
        'linear', 'extrap');
    pcFastCommon = interp1(sgFast(:), pcFast(:), sgCommon, ...
        'linear', 'extrap');
    pcOriginalCommon = max(pcOriginalCommon, realmin);
    pcFastCommon = max(pcFastCommon, realmin);

    rmsLog10Pc = sqrt(mean((log10(pcFastCommon) - ...
        log10(pcOriginalCommon)).^2, 'omitnan'));
    maxAbsLog10Pc = max(abs(log10(pcFastCommon) - ...
        log10(pcOriginalCommon)), [], 'omitnan');

    p50Original = interp1(sgOriginal(:), pcOriginal(:), 0.50, ...
        'linear', 'extrap') / 1e5;
    p50Fast = interp1(sgFast(:), pcFast(:), 0.50, ...
        'linear', 'extrap') / 1e5;
    p20Original = interp1(sgOriginal(:), pcOriginal(:), 0.20, ...
        'linear', 'extrap') / 1e5;
    p20Fast = interp1(sgFast(:), pcFast(:), 0.20, ...
        'linear', 'extrap') / 1e5;

    summaryRows(i, :) = {testCase.name, mat2str(testCase.cartDims), ...
        G.cells.num, originalSeconds, fastSeconds, ...
        rawFast.firstBottomConnectedPcPa / 1e5, ...
        rawFast.firstBottomConnectedSg, p20Original, p20Fast, ...
        p50Original, p50Fast, rmsLog10Pc};

    nexttile;
    semilogy(sgOriginal, pcOriginal / 1e5, '-', ...
        'Color', [0.09 0.29 0.55], 'LineWidth', 2.4);
    hold on;
    semilogy(sgFast, pcFast / 1e5, '--', ...
        'Color', [0.86 0.22 0.16], 'LineWidth', 2.4);
    semilogy(sgCommon, abs(pcFastCommon - pcOriginalCommon) / 1e5, ':', ...
        'Color', [0.2 0.2 0.2], 'LineWidth', 1.8);
    grid on;
    title(sprintf('%s | RMS log_{10} diff = %.3g', ...
        strrep(testCase.name, '_', ' '), rmsLog10Pc), ...
        'FontSize', 15, 'FontWeight', 'bold');
    xlabel('Gas saturation');
    ylabel('Pc [bar]');
    set(gca, 'FontSize', 12, 'LineWidth', 1.0);
    if i == 1
        legend({'original upscalePcReg', ...
            'fast threshold connectivity', 'absolute difference'}, ...
            'Location', 'best');
    end
end

summaryTable = cell2table(summaryRows, 'VariableNames', ...
    {'ValidationCase', 'CartDims', 'NumCells', 'OriginalSeconds', ...
     'FastSeconds', 'FastFirstConnectedPcBar', 'FastFirstConnectedSg', ...
     'OriginalPcSg20Bar', 'FastPcSg20Bar', 'OriginalPcSg50Bar', ...
     'FastPcSg50Bar', 'RmsLog10PcDifference'});
writetable(summaryTable, fullfile(tableDir, ...
    'ip_threshold_connectivity_validation_summary.csv'));

sgtitle({'Validation of optimized t = 1 invasion-percolation implementation', ...
    'Same grid, rock, fluid, and opt structures; smaller grids keep original loop tractable'}, ...
    'FontSize', 20, 'FontWeight', 'bold');
saveFigureBoth(fig, figureDir, ...
    'ip_threshold_connectivity_vs_original_validation');

fprintf('\nValidation complete.\nOutput root: %s\n', outputRoot);
disp(summaryTable);
end


function cases = buildValidationCases()
% Define synthetic validation cases with different connected-path patterns.

cases = struct([]);
cases(1).name = 'single_barrier_with_channel';
cases(1).cartDims = [10 1 10];
cases(1).pattern = 'barrier_channel';

cases(2).name = 'layered_clay_lens';
cases(2).cartDims = [12 2 10];
cases(2).pattern = 'layered_lens';

cases(3).name = 'random_patchy_smear';
cases(3).cartDims = [10 2 12];
cases(3).pattern = 'random_patchy';

cases(4).name = 'nearly_blocked_high_entry_path';
cases(4).cartDims = [12 1 12];
cases(4).pattern = 'nearly_blocked';
end


function [G, rock, fluid] = buildSyntheticProblem(testCase)
% Build a small MRST grid and assign deterministic synthetic materials.

dims = testCase.cartDims;
G = cartGrid(dims, dims);
G = computeGeometry(G);
[I, J, K] = gridLogicalIndices(G);

reg = ones(G.cells.num, 1);
switch testCase.pattern
    case 'barrier_channel'
        reg(K == 5 | K == 6) = 2;
        reg(I == 5 & (K == 5 | K == 6)) = 1;

    case 'layered_lens'
        reg(K >= 4 & K <= 6) = 2;
        reg(I >= 5 & I <= 8 & K >= 4 & K <= 6) = 3;
        reg(I <= 2 & K >= 4 & K <= 6) = 1;

    case 'random_patchy'
        rng(71231);
        patch = rand(G.cells.num, 1);
        reg(patch > 0.62) = 2;
        reg(patch > 0.84) = 3;
        reg(I == ceil(dims(1) / 2)) = 1;

    case 'nearly_blocked'
        reg(K >= 4 & K <= 9) = 2;
        reg(I >= 4 & I <= 9 & K >= 4 & K <= 9) = 3;
        reg(I == 2 & K >= 4 & K <= 9) = 1;

    otherwise
        error('Unknown validation pattern: %s', testCase.pattern);
end

poroByReg = [0.28, 0.16, 0.08];
rock = struct();
rock.poro = poroByReg(reg(:))';
rock.poro = rock.poro(:);
rock.regions = struct();
rock.regions.saturation = reg(:);

fluid = struct();
fluid.isclay = [false, true, true];
fluid.krPts = struct();
fluid.krPts.g = [0 0 0.80; 0 0 0.65; 0 0 0.55];
fluid.pcOG = cell(1, 3);
fluid.pcOG{1} = @(sg) 2.5e3 + 4.0e5 .* max(sg, 0).^1.45;
fluid.pcOG{2} = @(sg) 1.8e5 + 7.0e6 .* max(sg, 0).^1.35;
fluid.pcOG{3} = @(sg) 5.5e5 + 2.2e7 .* max(sg, 0).^1.25;
end


function [I, J, K] = gridLogicalIndices(G)
% Return logical i, j, k indices for each cell in a Cartesian grid.

dims = G.cartDims;
[iGrid, jGrid, kGrid] = ndgrid(1:dims(1), 1:dims(2), 1:dims(3));
I = iGrid(:);
J = jGrid(:);
K = kGrid(:);
end


function [pc, sg, diagnostics] = fastThresholdConnectivityUpscalePcReg(G, fluid, rock, opt)
% Fast t = 1 threshold-connectivity implementation with original output convention.

[pcVal, sVal, diagnostics] = fastThresholdConnectivityRaw(G, fluid, rock, opt);
satMin = sVal(1);
satMax = sVal(end);
if satMin ~= 0
    error('Expected zero minimum saturation.');
end
sPerc = sVal(find(sVal > 1e-5, 1, 'first'));
sg = [satMin, 1e-5, linspace(sPerc, satMax, opt.nval - 2)];
pc = interp1(sVal(:), pcVal(:), sg(:), 'linear', 'extrap');
sg = sg(:);
pc = pc(:);
end


function [pcVal, sVal, diagnostics] = fastThresholdConnectivityRaw(G, fluid, rock, opt)
% Compute raw Pc/Sg points using connected cells above entry-pressure threshold.

reg = rock.regions.saturation;
idReg = unique(reg);
nreg = numel(idReg);
ncreg = sum(reg(:)==(1:max(reg)));
ncreg(ncreg==0) = [];

fluid.pcInv = cell(1, max(reg));
pcMax = 0;
pcv2 = 1e6;
sgmax = zeros(1, nreg);
pcAtSgMax = zeros(1, max(idReg));
pce = zeros(G.cells.num, 1);

for n = 1:nreg
    regId = idReg(n);
    if isempty(opt.sg)
        sgmin = fluid.krPts.g(n, 1);
        sgmax(n) = fluid.krPts.g(n, 3);
        sgvals = linspace(sgmin, sgmax(n), pow2(6)-1)';
    elseif strcmp(opt.sg, 'sandClay')
        if fluid.isclay(n) == 1
            sgmin = fluid.krPts.g(2, 1);
            sgmax(n) = fluid.krPts.g(2, 3);
        else
            sgmin = fluid.krPts.g(1, 1);
            sgmax(n) = fluid.krPts.g(1, 3)-0.01;
        end
        sgvals = linspace(sgmin, sgmax(n), pow2(6)-1)';
    else
        error('Unsupported opt.sg setting.');
    end
    sgvals = [sgvals(1); sgvals(1)+1e-3; sgvals(2:end)];
    pcvals = fluid.pcOG{regId}(sgvals);
    pcv2 = min([pcv2, pcvals(2)]);
    fluid.pcInv{regId} = @(pcOG) interp1(pcvals, sgvals, pcOG);
    pcAtSgMax(regId) = fluid.pcOG{regId}(sgmax(n));
    if fluid.isclay(n) == 1
        pcMax = max([pcMax, pcAtSgMax(regId)]);
    end
    pce(reg == regId) = fluid.pcOG{regId}(1e-3);
end

pcVal = logspace(log10(pcv2), log10(0.99*pcMax), pow2(6)-2);
pcVal = [0 0.98*pcv2 pcVal];

topCells = (1:G.cartDims(1)*G.cartDims(2))';
bottomCells = ((G.cells.num-G.cartDims(1)*G.cartDims(2))+1:G.cells.num)';
adj = buildCellAdjacency(G);
volume = sum(G.cells.volumes .* rock.poro);
sVal = zeros(numel(pcVal), 1);
sgCells = zeros(G.cells.num, 1);
sLast = 0;
idrem = false(numel(pcVal), 1);
idPercolation = [];

for k = 2:numel(pcVal)
    pcv = pcVal(k);
    openMask = pce <= pcv;
    invaded = connectedOpenCells(adj, openMask, topCells);
    if any(invaded(bottomCells)) && isempty(idPercolation)
        idPercolation = k;
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
        sVal(k) = sum(sgCells .* G.cells.volumes .* rock.poro) / volume;
    end

    if abs(sVal(k) - sLast) < 1e-3
        idrem(k) = true;
    else
        sLast = sVal(k);
    end
end

if isempty(idPercolation)
    error('No percolating path found in validation case.');
end
idrem(idPercolation-1) = false;
idrem(idPercolation) = false;
idrem(end) = false;
sVal(idPercolation-1) = 1e-5;
pcVal(idrem) = [];
sVal(idrem) = [];
pcVal = pcVal(:);
sVal = sVal(:);

diagnostics = struct();
diagnostics.firstBottomConnectedPcPa = pcVal(find(sVal > 1e-5, 1, 'first'));
diagnostics.firstBottomConnectedSg = sVal(find(sVal > 1e-5, 1, 'first'));
diagnostics.numRawPoints = numel(pcVal);
diagnostics.ncreg = ncreg;
end


function adj = buildCellAdjacency(G)
% Build cell adjacency from the MRST face-neighbor table.

neighbors = G.faces.neighbors;
neighbors = neighbors(all(neighbors > 0, 2), :);
numCells = G.cells.num;
degree = accumarray(neighbors(:), 1, [numCells, 1]);
adj = cell(numCells, 1);
for c = 1:numCells
    adj{c} = zeros(degree(c), 1);
end
cursor = ones(numCells, 1);
for row = 1:size(neighbors, 1)
    a = neighbors(row, 1);
    b = neighbors(row, 2);
    adj{a}(cursor(a)) = b;
    cursor(a) = cursor(a) + 1;
    adj{b}(cursor(b)) = a;
    cursor(b) = cursor(b) + 1;
end
end


function invaded = connectedOpenCells(adj, openMask, seedCells)
% Return cells connected to seedCells through open cells.

numCells = numel(openMask);
invaded = false(numCells, 1);
queue = zeros(numCells, 1);
head = 1;
tail = 0;
seedCells = seedCells(openMask(seedCells));
for i = 1:numel(seedCells)
    c = seedCells(i);
    if ~invaded(c)
        tail = tail + 1;
        queue(tail) = c;
        invaded(c) = true;
    end
end
while head <= tail
    c = queue(head);
    head = head + 1;
    neighbors = adj{c};
    for j = 1:numel(neighbors)
        nb = neighbors(j);
        if openMask(nb) && ~invaded(nb)
            invaded(nb) = true;
            tail = tail + 1;
            queue(tail) = nb;
        end
    end
end
end


function initializePaths(repoRoot)
% Initialize MRST and add original upscaling code paths.

mrstStartup = fullfile('C:\Users\Shaow\OneDrive\MIT\mrst-2025a', ...
    'SINTEF-AppliedCompSci-MRST-75749fa', 'startup.m');
if exist(mrstStartup, 'file') == 2
    run(mrstStartup);
else
    warning('MRST startup not found: %s', mrstStartup);
end

upscalingRoot = fullfile('D:\codex_gom', 'tmp_upscaling_zip_inspect');
addpath(fullfile(upscalingRoot, 'upscaling'));
addpath(upscalingRoot);
addpath(fullfile(repoRoot, 'examples', 'pc_upscaling_pilot'));
end


function ensureFolder(folderPath)
% Create a folder if needed.

if exist(folderPath, 'dir') ~= 7
    mkdir(folderPath);
end
end


function saveFigureBoth(fig, folderPath, baseName)
% Save a figure as PNG and PDF.

pngPath = fullfile(folderPath, baseName + ".png");
pdfPath = fullfile(folderPath, baseName + ".pdf");
exportgraphics(fig, pngPath, 'Resolution', 300);
exportgraphics(fig, pdfPath, 'ContentType', 'vector');
fprintf('Saved figure: %s\n', pngPath);
fprintf('Saved figure: %s\n', pdfPath);
end
