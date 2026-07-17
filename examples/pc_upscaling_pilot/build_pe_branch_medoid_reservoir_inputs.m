function outputs = build_pe_branch_medoid_reservoir_inputs( ...
        fullSliceInput, outputDir, varargin)
%BUILD_PE_BRANCH_MEDOID_RESERVOIR_INPUTS Reduce Pc heterogeneity by Pe branch.
%
%   OUTPUTS = BUILD_PE_BRANCH_MEDOID_RESERVOIR_INPUTS(FULLSLICEINPUT,
%   OUTPUTDIR) reads one rigorously validated full-slice reservoir-ready
%   structure or MAT file. For every window it:
%     1. detects well-separated entry-pressure branches in log10(Pe);
%     2. selects one actual full-Pc-curve medoid inside each branch;
%     3. assigns every along-strike slice to its branch medoid; and
%     4. maps the existing normalized dynamic-Kr shape to each branch
%        medoid's native saturation endpoint.
%
%   Permeability and porosity remain slice specific. The function never
%   modifies or replaces the full-slice input artifact. It writes a new
%   reservoir-ready MAT file, branch and slice-assignment CSV tables, and a
%   QA table.
%
%   Entry pressure is the first positive Pc value on each native curve.
%   Branches are separated only at adjacent log10(Pe) gaps at least
%   MinLog10PeGap wide. A split is accepted only when both resulting branch
%   segments contain at least MinBranchCount curves. This conservative rule
%   distinguishes physically separated Pe levels without splitting smooth
%   within-level variation.
%
%   Full-curve distance within a branch is
%
%       sqrt(mean((delta log10(Pc(u)))^2) +
%            (EndpointWeight * delta BulkSgMax)^2),
%
%   where u = Sg/BulkSgMax is a common normalized saturation grid. The
%   selected medoid is always an actual upscaled Pc curve.
%
%   Name-value options:
%     MinLog10PeGap    default 1.0 (one order of magnitude)
%     MinBranchCount  default 2
%     MaxBranches     default 3
%     CurveGridPoints default 101
%     EndpointWeight  default 1.0
%     ValidationTolerance default 1e-8
%
%   See also EXPORT_RESERVOIR_READY_PC_KR_CASES.

p = inputParser;
addParameter(p, 'MinLog10PeGap', 1.0, ...
    @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
addParameter(p, 'MinBranchCount', 2, ...
    @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 1);
addParameter(p, 'MaxBranches', 3, ...
    @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 1);
addParameter(p, 'CurveGridPoints', 101, ...
    @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 11);
addParameter(p, 'EndpointWeight', 1.0, ...
    @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 0);
addParameter(p, 'ValidationTolerance', 1.0e-8, ...
    @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 0);
parse(p, varargin{:});
opt = p.Results;
opt.MinBranchCount = round(opt.MinBranchCount);
opt.MaxBranches = round(opt.MaxBranches);
opt.CurveGridPoints = round(opt.CurveGridPoints);

[fullSlice, source] = loadFullSliceInput(fullSliceInput);
validateFullSliceStructure(fullSlice);
ensureFolder(outputDir);
tableDir = fullfile(outputDir, 'tables');
ensureFolder(tableDir);

windows = string(fullSlice.windowLabels(:));
slices = double(fullSlice.sliceIndices(:));
nWindows = numel(windows);
nSlices = numel(slices);
uGrid = linspace(0, 1, opt.CurveGridPoints)';

reducedPc = cell(nWindows, nSlices);
reducedKr = cell(nWindows, nSlices);
branchLabels = zeros(nWindows, nSlices);
branchCounts = zeros(nWindows, 1);
representativePc = cell(nWindows, opt.MaxBranches);
representativeKr = cell(nWindows, opt.MaxBranches);
branchRows = cell(nWindows * opt.MaxBranches, 19);
branchRowId = 0;
assignmentRows = cell(nWindows * nSlices, 18);
assignmentRowId = 0;

maxEndpointMismatch = 0;
maxPcMonotonicDrop = 0;
maxKrgMonotonicDrop = 0;
maxKrwMonotonicRise = 0;
maxKrShapeMismatch = 0;

for w = 1:nWindows
    windowName = windows(w);
    originalCurves = fullSlice.pcCurves(w, :);
    entryPcBar = nan(nSlices, 1);
    bulkSgMax = nan(nSlices, 1);
    replaySourceRows = nan(nSlices, 1);
    curveFeatures = nan(nSlices, opt.CurveGridPoints);

    for s = 1:nSlices
        P = originalCurves{s};
        [entryPcBar(s), curveFeatures(s, :)] = ...
            pcCurveFeature(P, uGrid);
        bulkSgMax(s) = double(P.bulkSgMax);
        replaySourceRows(s) = double(P.replaySourceRow);
    end

    labels = detectPeBranches(log10(entryPcBar), ...
        opt.MinLog10PeGap, opt.MinBranchCount, opt.MaxBranches);
    branchLabels(w, :) = labels(:)';
    nBranches = max(labels);
    branchCounts(w) = nBranches;
    [krU, krgShape, krwShape, shapeId, representativeKrSource, ...
        localShapeMismatch] = extractWindowKrShape( ...
        fullSlice.krCurves(w, :), opt.ValidationTolerance);
    maxKrShapeMismatch = max(maxKrShapeMismatch, localShapeMismatch);

    for b = 1:nBranches
        branchSliceColumns = find(labels == b);
        branchFeature = curveFeatures(branchSliceColumns, :);
        branchEndpoints = bulkSgMax(branchSliceColumns);
        D = fullCurveDistanceMatrix(branchFeature, branchEndpoints, ...
            opt.EndpointWeight);
        centrality = mean(D, 2);
        tieTable = table(centrality, ...
            replaySourceRows(branchSliceColumns), ...
            slices(branchSliceColumns), (1:numel(branchSliceColumns))', ...
            'VariableNames', {'Centrality', 'ReplaySourceRow', ...
            'SliceIndex', 'LocalIndex'});
        tieTable = sortrows(tieTable, ...
            {'Centrality', 'ReplaySourceRow', 'SliceIndex'});
        localMedoid = tieTable.LocalIndex(1);
        medoidColumn = branchSliceColumns(localMedoid);
        medoidPc = originalCurves{medoidColumn};
        medoidEntryPcBar = entryPcBar(medoidColumn);
        medoidSgMax = double(medoidPc.bulkSgMax);
        medoidSwi = 1.0 - medoidSgMax;
        representativePc{w, b} = annotateRepresentativePc( ...
            medoidPc, b, slices(medoidColumn), medoidEntryPcBar);
        representativeKr{w, b} = makeBranchKrCurve( ...
            krU, krgShape, krwShape, medoidSgMax, shapeId, ...
            representativeKrSource, b, slices(medoidColumn), ...
            double(medoidPc.replaySourceRow));

        distanceToMedoid = D(:, localMedoid);
        branchRowId = branchRowId + 1;
        branchRows(branchRowId, :) = { ...
            string(fullSlice.geologyId), double(fullSlice.level3CaseId), ...
            string(fullSlice.level3CaseName), windowName, b, ...
            numel(branchSliceColumns), numel(branchSliceColumns) / nSlices, ...
            min(entryPcBar(branchSliceColumns)), ...
            median(entryPcBar(branchSliceColumns)), ...
            max(entryPcBar(branchSliceColumns)), ...
            slices(medoidColumn), double(medoidPc.replaySourceRow), ...
            medoidEntryPcBar, medoidSgMax, medoidSwi, ...
            centrality(localMedoid), mean(distanceToMedoid), ...
            max(distanceToMedoid), ...
            minAdjacentBranchGap(log10(entryPcBar), labels, b)};

        for j = 1:numel(branchSliceColumns)
            s = branchSliceColumns(j);
            originalPc = originalCurves{s};
            assignedPc = representativePc{w, b};
            assignedPc.originalSliceIndex = slices(s);
            assignedPc.originalReplaySourceRow = ...
                double(originalPc.replaySourceRow);
            assignedPc.originalEntryPcBar = entryPcBar(s);
            assignedPc.upscaledPorosity = ...
                double(fullSlice.upscaledPorosity(w, s));
            reducedPc{w, s} = assignedPc;
            reducedKr{w, s} = representativeKr{w, b};

            assignmentRowId = assignmentRowId + 1;
            assignmentRows(assignmentRowId, :) = { ...
                string(fullSlice.geologyId), ...
                double(fullSlice.level3CaseId), ...
                string(fullSlice.level3CaseName), windowName, ...
                slices(s), b, entryPcBar(s), medoidEntryPcBar, ...
                abs(log10(entryPcBar(s)) - log10(medoidEntryPcBar)), ...
                double(originalPc.bulkSgMax), medoidSgMax, ...
                abs(double(originalPc.bulkSgMax) - medoidSgMax), ...
                double(originalPc.replaySourceRow), ...
                double(medoidPc.replaySourceRow), slices(medoidColumn), ...
                distanceToMedoid(j), ...
                double(fullSlice.upscaledPorosity(w, s)), ...
                double(fullSlice.effectivePermeability.mD(w, s, 3))};

            endpointMismatch = validateReducedCurvePair( ...
                reducedPc{w, s}, reducedKr{w, s}, ...
                opt.ValidationTolerance);
            maxEndpointMismatch = max(maxEndpointMismatch, endpointMismatch);
            maxPcMonotonicDrop = max(maxPcMonotonicDrop, ...
                maximumDrop(double(reducedPc{w, s}.pcBar)));
            maxKrgMonotonicDrop = max(maxKrgMonotonicDrop, ...
                maximumDrop(double(reducedKr{w, s}.krg)));
            maxKrwMonotonicRise = max(maxKrwMonotonicRise, ...
                maximumRise(double(reducedKr{w, s}.krw)));
        end
    end
end

branchSummary = cell2table(branchRows(1:branchRowId, :), ...
    'VariableNames', { ...
    'GeologyId', 'Level3CaseId', 'Level3CaseName', 'Window', ...
    'PeBranchId', 'SliceCount', 'SliceFraction', 'MinEntryPcBar', ...
    'MedianEntryPcBar', 'MaxEntryPcBar', 'MedoidSliceIndex', ...
    'MedoidReplaySourceRow', 'MedoidEntryPcBar', 'MedoidBulkSgMax', ...
    'MedoidEffectiveSwi', 'MedoidCentrality', ...
    'MeanCurveDistanceToMedoid', 'MaxCurveDistanceToMedoid', ...
    'NearestAdjacentBranchGapLog10Pe'});
sliceAssignments = cell2table(assignmentRows, 'VariableNames', { ...
    'GeologyId', 'Level3CaseId', 'Level3CaseName', 'Window', ...
    'SliceIndex', 'PeBranchId', 'OriginalEntryPcBar', ...
    'RepresentativeEntryPcBar', 'AbsoluteLog10PeError', ...
    'OriginalBulkSgMax', 'RepresentativeBulkSgMax', ...
    'AbsoluteBulkSgMaxError', 'OriginalReplaySourceRow', ...
    'RepresentativeReplaySourceRow', 'RepresentativeSliceIndex', ...
    'FullCurveDistanceToMedoid', 'UpscaledPorosity', ...
    'EffectiveKzzMD'});

reduced = fullSlice;
reduced.schemaVersion = "1.4";
reduced.pcRepresentation = "pe_branch_medoid";
reduced.pcCurves = reducedPc;
reduced.krCurves = reducedKr;
reduced.peBranchModel = struct( ...
    'branchLabels', branchLabels, ...
    'branchCounts', branchCounts, ...
    'branchSummary', branchSummary, ...
    'sliceAssignments', sliceAssignments, ...
    'representativePcCurves', {representativePc}, ...
    'representativeKrCurves', {representativeKr}, ...
    'normalizedCurveGrid', uGrid, ...
    'settings', opt);
reduced.saturationRegions = build_saturation_region_metadata(reduced);
reduced.provenance.peBranchMedoidSource = source;
reduced.provenance.peBranchMedoidCreatedAt = ...
    string(datetime('now', 'TimeZone', 'UTC'));

totalBranches = sum(branchCounts);
qa = table(string(reduced.geologyId), double(reduced.level3CaseId), ...
    string(reduced.level3CaseName), nWindows, nSlices, ...
    nWindows * nSlices, totalBranches, min(branchCounts), ...
    max(branchCounts), max(sliceAssignments.AbsoluteLog10PeError), ...
    mean(sliceAssignments.AbsoluteLog10PeError), ...
    max(sliceAssignments.AbsoluteBulkSgMaxError), ...
    mean(sliceAssignments.AbsoluteBulkSgMaxError), ...
    max(sliceAssignments.FullCurveDistanceToMedoid), ...
    mean(sliceAssignments.FullCurveDistanceToMedoid), ...
    maxEndpointMismatch, maxPcMonotonicDrop, ...
    maxKrgMonotonicDrop, maxKrwMonotonicRise, maxKrShapeMismatch, true, ...
    'VariableNames', {'GeologyId', 'Level3CaseId', 'Level3CaseName', ...
    'WindowCount', 'SliceCount', 'FaultCellCount', ...
    'UniquePcKrPairCount', 'MinBranchesPerWindow', ...
    'MaxBranchesPerWindow', 'MaxAbsoluteLog10PeError', ...
    'MeanAbsoluteLog10PeError', 'MaxAbsoluteBulkSgMaxError', ...
    'MeanAbsoluteBulkSgMaxError', 'MaxFullCurveDistanceToMedoid', ...
    'MeanFullCurveDistanceToMedoid', 'MaxEndpointMismatch', ...
    'MaxPcMonotonicDrop', 'MaxKrgMonotonicDrop', ...
    'MaxKrwMonotonicRise', 'MaxNormalizedKrShapeMismatch', 'Passed'});

token = sprintf('%s_case%02d', char(reduced.geologyId), ...
    double(reduced.level3CaseId));
matFile = fullfile(outputDir, sprintf( ...
    'reservoir_ready_pe_branch_medoid_%s.mat', token));
branchCsv = fullfile(tableDir, sprintf( ...
    'pe_branch_summary_%s.csv', token));
assignmentCsv = fullfile(tableDir, sprintf( ...
    'pe_branch_slice_assignments_%s.csv', token));
qaCsv = fullfile(tableDir, sprintf('pe_branch_qa_%s.csv', token));

reservoirReady = reduced;
save(matFile, 'reservoirReady', '-v7.3');
writetable(branchSummary, branchCsv);
writetable(sliceAssignments, assignmentCsv);
writetable(qa, qaCsv);

outputs = struct( ...
    'matFile', string(matFile), ...
    'branchSummaryCsv', string(branchCsv), ...
    'sliceAssignmentsCsv', string(assignmentCsv), ...
    'qaCsv', string(qaCsv), ...
    'branchSummary', branchSummary, ...
    'sliceAssignments', sliceAssignments, ...
    'qaTable', qa);

fprintf('Saved Pe-branch-medoid reservoir inputs: %s\n', matFile);
fprintf('Reduced 522 slice Pc/Kr pairs to %d unique branch pairs.\n', ...
    totalBranches);
fprintf('Saved branch QA: %s\n', qaCsv);
end


function [S, source] = loadFullSliceInput(value)
% Load a full-slice reservoir-ready structure or MAT file.

if isstruct(value)
    if isfield(value, 'reservoirReady')
        S = value.reservoirReady;
    else
        S = value;
    end
    source = "in-memory reservoirReady struct";
else
    file = string(value);
    assert(isscalar(file) && isfile(file), ...
        'PeBranch:MissingInput', ...
        'Full-slice reservoir-ready MAT file not found: %s', file);
    loaded = load(file, 'reservoirReady');
    assert(isfield(loaded, 'reservoirReady'), ...
        'PeBranch:MissingVariable', ...
        'MAT file lacks reservoirReady: %s', file);
    S = loaded.reservoirReady;
    source = file;
end
end


function validateFullSliceStructure(S)
% Require the fields needed for a loss-controlled Pc/Kr reduction.

required = {'schemaVersion', 'geologyId', 'level3CaseId', ...
    'level3CaseName', 'windowLabels', 'sliceIndices', 'pcCurves', ...
    'krCurves', 'effectivePermeability', 'upscaledPorosity', ...
    'provenance'};
missing = required(~isfield(S, required));
assert(isempty(missing), 'PeBranch:MissingFields', ...
    'Full-slice input is missing fields: %s', strjoin(missing, ', '));
assert(~isfield(S, 'pcRepresentation') || ...
    string(S.pcRepresentation) == "full_slice", ...
    'PeBranch:InputAlreadyReduced', ...
    'Pe-branch reduction must start from the rigorous full-slice artifact.');
nWindows = numel(S.windowLabels);
nSlices = numel(S.sliceIndices);
assert(isequal(size(S.pcCurves), [nWindows, nSlices]) && ...
    isequal(size(S.krCurves), [nWindows, nSlices]) && ...
    isequal(size(S.upscaledPorosity), [nWindows, nSlices]), ...
    'PeBranch:CoverageMismatch', ...
    'Pc, Kr, and porosity arrays must all have window-by-slice coverage.');
end


function labels = detectPeBranches(logPe, minGap, minCount, maxBranches)
% Split sorted log10(Pe) only at large, well-supported adjacent gaps.

logPe = double(logPe(:));
assert(all(isfinite(logPe)), 'PeBranch:InvalidPe', ...
    'Entry pressures must be finite and positive.');
n = numel(logPe);
[sortedPe, order] = sort(logPe);
gaps = diff(sortedPe);
[~, candidateOrder] = sort(gaps, 'descend');
splitAfter = zeros(0, 1);

for k = 1:numel(candidateOrder)
    split = candidateOrder(k);
    if gaps(split) < minGap || numel(splitAfter) + 1 >= maxBranches
        continue
    end
    boundaries = [0; sort(splitAfter); n];
    segmentId = find(split > boundaries(1:end-1) & ...
        split < boundaries(2:end), 1, 'first');
    if isempty(segmentId)
        continue
    end
    leftCount = split - boundaries(segmentId);
    rightCount = boundaries(segmentId + 1) - split;
    if leftCount >= minCount && rightCount >= minCount
        splitAfter(end + 1, 1) = split; %#ok<AGROW>
    end
end

splitAfter = sort(splitAfter);
sortedLabels = ones(n, 1);
for k = 1:numel(splitAfter)
    sortedLabels((splitAfter(k) + 1):end) = k + 1;
end
labels = zeros(n, 1);
labels(order) = sortedLabels;
end


function [entryPcBar, feature] = pcCurveFeature(P, uGrid)
% Interpolate a native Pc curve in normalized saturation and log-Pc space.

sg = double(P.gasSaturation(:));
pcBar = double(P.pcBar(:));
sgMax = double(P.bulkSgMax);
valid = isfinite(sg) & isfinite(pcBar) & pcBar > 0 & ...
    sg >= 0 & sg <= sgMax * (1 + 1.0e-10);
assert(any(valid) && isfinite(sgMax) && sgMax > 0, ...
    'PeBranch:InvalidPcCurve', ...
    'A native Pc curve lacks positive values or a valid BulkSgMax.');
sg = sg(valid);
pcBar = pcBar(valid);
[sg, order] = sort(sg);
pcBar = pcBar(order);
[sg, uniqueIndex] = unique(sg, 'stable');
pcBar = pcBar(uniqueIndex);
entryPcBar = pcBar(1);
u = sg ./ sgMax;
if u(1) > 0
    u = [0; u];
    pcBar = [entryPcBar; pcBar];
end
if u(end) < 1
    u = [u; 1];
    pcBar = [pcBar; pcBar(end)];
end
feature = interp1(u, log10(pcBar), uGrid, 'linear');
feature = feature(:)';
assert(all(isfinite(feature)), 'PeBranch:CurveInterpolation', ...
    'Pc curve interpolation produced non-finite values.');
end


function D = fullCurveDistanceMatrix(features, endpoints, endpointWeight)
% Pairwise full-curve distance in log-Pc shape plus endpoint saturation.

n = size(features, 1);
D = zeros(n, n);
for i = 1:n
    for j = (i + 1):n
        shapeTerm = mean((features(i, :) - features(j, :)).^2);
        endpointTerm = (endpointWeight * ...
            (endpoints(i) - endpoints(j)))^2;
        d = sqrt(shapeTerm + endpointTerm);
        D(i, j) = d;
        D(j, i) = d;
    end
end
end


function P = annotateRepresentativePc(P, branchId, sliceId, entryPcBar)
% Add explicit branch provenance to a representative Pc curve.

P.representation = "pe_branch_medoid";
P.peBranchId = branchId;
P.representativeSliceIndex = sliceId;
P.representativeReplaySourceRow = double(P.replaySourceRow);
P.entryPcBar = entryPcBar;
end


function [u, krg, krw, shapeId, representativeSource, maxMismatch] = ...
        extractWindowKrShape(curves, tolerance)
% Verify and return the one normalized dynamic-Kr shape used by a window.

reference = curves{1};
u = double(reference.gasSaturation(:)) ./ double(reference.bulkSgMax);
krg = double(reference.krg(:));
krw = double(reference.krw(:));
shapeId = string(reference.shapeId);
representativeSource = double(reference.representativeReplaySourceRow);
maxMismatch = 0;

for s = 1:numel(curves)
    K = curves{s};
    localU = double(K.gasSaturation(:)) ./ double(K.bulkSgMax);
    localKrg = interp1(localU, double(K.krg(:)), u, 'linear');
    localKrw = interp1(localU, double(K.krw(:)), u, 'linear');
    mismatch = max([max(abs(localKrg - krg)), ...
        max(abs(localKrw - krw))]);
    maxMismatch = max(maxMismatch, mismatch);
end
assert(maxMismatch <= tolerance, 'PeBranch:KrShapeMismatch', ...
    ['Slice Kr curves do not share one normalized window shape; ', ...
     'maximum mismatch is %.3g.'], maxMismatch);
end


function K = makeBranchKrCurve(u, krg, krw, sgMax, shapeId, ...
        representativeSource, branchId, pcSliceId, pcReplaySource)
% Map one normalized dynamic-Kr shape to a branch-medoid Pc endpoint.

sg = u .* sgMax;
K = struct( ...
    'gasSaturation', sg, ...
    'waterSaturation', 1.0 - sg, ...
    'krg', krg, ...
    'krw', krw, ...
    'bulkSgMax', sgMax, ...
    'effectiveSwi', 1.0 - sgMax, ...
    'shapeId', shapeId + sprintf('_pe_branch_%02d', branchId), ...
    'representativeReplaySourceRow', representativeSource, ...
    'representation', "pe_branch_medoid", ...
    'peBranchId', branchId, ...
    'representativePcSliceIndex', pcSliceId, ...
    'representativePcReplaySourceRow', pcReplaySource);
end


function mismatch = validateReducedCurvePair(P, K, tolerance)
% Validate endpoint identity, monotonicity, and physical Kr bounds.

pcSg = double(P.gasSaturation(:));
pc = double(P.pcBar(:));
krSg = double(K.gasSaturation(:));
krSw = double(K.waterSaturation(:));
krg = double(K.krg(:));
krw = double(K.krw(:));
mismatch = max([ ...
    abs(double(P.bulkSgMax) - double(K.bulkSgMax)), ...
    abs(double(P.effectiveSwi) - double(K.effectiveSwi)), ...
    abs(max(pcSg) - double(P.bulkSgMax)), ...
    abs(max(krSg) - double(P.bulkSgMax)), ...
    max(abs(krSw - (1.0 - krSg)))]);
assert(mismatch <= tolerance, 'PeBranch:EndpointMismatch', ...
    'Reduced Pc/Kr endpoint mismatch %.3g exceeds %.3g.', ...
    mismatch, tolerance);
assert(maximumDrop(pc) <= tolerance, 'PeBranch:NonmonotonicPc', ...
    'A reduced Pc curve is nonmonotonic.');
assert(maximumDrop(krg) <= tolerance && ...
    maximumRise(krw) <= tolerance, 'PeBranch:NonmonotonicKr', ...
    'A reduced Kr curve is nonmonotonic.');
assert(all(krg >= -tolerance & krg <= 1 + tolerance) && ...
    all(krw >= -tolerance & krw <= 1 + tolerance), ...
    'PeBranch:KrBounds', 'A reduced Kr curve lies outside [0,1].');
end


function gap = minAdjacentBranchGap(logPe, labels, branchId)
% Return the nearest log10(Pe) gap separating a branch from its neighbor.

branchValues = logPe(labels == branchId);
gaps = nan(2, 1);
gapCount = 0;
if any(labels == branchId - 1)
    gapCount = gapCount + 1;
    gaps(gapCount) = min(branchValues) - ...
        max(logPe(labels == branchId - 1));
end
if any(labels == branchId + 1)
    gapCount = gapCount + 1;
    gaps(gapCount) = min(logPe(labels == branchId + 1)) - ...
        max(branchValues);
end
if gapCount == 0
    gap = NaN;
else
    gap = min(gaps(1:gapCount));
end
end


function value = maximumDrop(values)
% Largest downward step in an expected nondecreasing sequence.

value = max([0; -diff(double(values(:)))]);
end


function value = maximumRise(values)
% Largest upward step in an expected nonincreasing sequence.

value = max([0; diff(double(values(:)))]);
end


function ensureFolder(folderPath)
% Create a folder when it does not already exist.

if ~isfolder(folderPath)
    mkdir(folderPath);
end
end
