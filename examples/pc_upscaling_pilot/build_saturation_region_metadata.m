function regions = build_saturation_region_metadata(reservoirReady)
%BUILD_SATURATION_REGION_METADATA Build explicit downstream SATNUM metadata.
%
%   REGIONS = BUILD_SATURATION_REGION_METADATA(RESERVOIRREADY) converts the
%   window-by-slice Pc/Kr representation into globally numbered saturation
%   regions. The result contains:
%     * SATNUM: a window-by-slice global region map;
%     * regionCount and contiguous regionIds;
%     * regionCurveLinearIndices into RESERVOIRREADY.pcCurves/krCurves;
%     * regionTable and assignmentTable for transparent downstream use;
%     * numerical validation that every cell assigned to a region uses the
%       same physical Pc and Kr curves as that region's lookup curve.
%
%   For full_slice, every window-slice cell is its own region. For
%   pe_branch_medoid, each (window, local Pe branch) pair is one region.
%   Local branch labels are never reused globally across windows.

required = {'windowLabels', 'sliceIndices', 'pcCurves', 'krCurves'};
for i = 1:numel(required)
    assert(isfield(reservoirReady, required{i}), ...
        'SaturationRegions:MissingField', ...
        'reservoirReady is missing required field %s.', required{i});
end

windows = string(reservoirReady.windowLabels(:));
slices = double(reservoirReady.sliceIndices(:));
nWindows = numel(windows);
nSlices = numel(slices);
assert(isequal(size(reservoirReady.pcCurves), [nWindows, nSlices]) && ...
    isequal(size(reservoirReady.krCurves), [nWindows, nSlices]), ...
    'SaturationRegions:CurveCoverage', ...
    'Pc and Kr arrays must have window-by-slice coverage.');
assert(nWindows * nSlices <= double(intmax('uint16')), ...
    'SaturationRegions:RegionIdOverflow', ...
    'The region count exceeds the uint16 SATNUM capacity.');

representation = "full_slice";
if isfield(reservoirReady, 'pcRepresentation')
    representation = lower(string(reservoirReady.pcRepresentation));
end
assert(any(representation == ["full_slice", "pe_branch_medoid"]), ...
    'SaturationRegions:Representation', ...
    'Unsupported Pc representation: %s.', representation);

satnum = zeros(nWindows, nSlices, 'uint16');
maxRegions = nWindows * nSlices;
regionWindowIndex = zeros(maxRegions, 1, 'uint16');
regionWindow = strings(maxRegions, 1);
regionLocalDomain = zeros(maxRegions, 1, 'uint16');
regionPeBranch = nan(maxRegions, 1);
regionRepresentativeSlice = nan(maxRegions, 1);
regionCurveLinearIndex = zeros(maxRegions, 1, 'uint32');
regionAssignedCellCount = zeros(maxRegions, 1, 'uint16');

regionId = 0;
if representation == "full_slice"
    for w = 1:nWindows
        for s = 1:nSlices
            regionId = regionId + 1;
            satnum(w, s) = uint16(regionId);
            regionWindowIndex(regionId) = uint16(w);
            regionWindow(regionId) = windows(w);
            regionLocalDomain(regionId) = uint16(s);
            regionRepresentativeSlice(regionId) = slices(s);
            regionCurveLinearIndex(regionId) = uint32( ...
                sub2ind([nWindows, nSlices], w, s));
            regionAssignedCellCount(regionId) = uint16(1);
        end
    end
else
    assert(isfield(reservoirReady, 'peBranchModel') && ...
        isfield(reservoirReady.peBranchModel, 'branchLabels') && ...
        isfield(reservoirReady.peBranchModel, 'branchCounts') && ...
        isfield(reservoirReady.peBranchModel, 'branchSummary'), ...
        'SaturationRegions:BranchMetadata', ...
        'pe_branch_medoid requires labels, counts, and branchSummary.');
    labels = double(reservoirReady.peBranchModel.branchLabels);
    branchCounts = double(reservoirReady.peBranchModel.branchCounts(:));
    branchSummary = reservoirReady.peBranchModel.branchSummary;
    assert(isequal(size(labels), [nWindows, nSlices]) && ...
        numel(branchCounts) == nWindows, ...
        'SaturationRegions:BranchCoverage', ...
        'Pe-branch metadata does not match window-by-slice coverage.');

    for w = 1:nWindows
        for b = 1:branchCounts(w)
            assignedColumns = find(labels(w, :) == b);
            assert(~isempty(assignedColumns), ...
                'SaturationRegions:EmptyBranch', ...
                '%s branch %d has no assigned slices.', windows(w), b);
            summaryMask = string(branchSummary.Window) == windows(w) & ...
                double(branchSummary.PeBranchId) == b;
            assert(sum(summaryMask) == 1, ...
                'SaturationRegions:BranchSummary', ...
                '%s branch %d does not have one summary row.', windows(w), b);
            representativeSlice = double( ...
                branchSummary.MedoidSliceIndex(summaryMask));
            representativeColumn = find(slices == representativeSlice, 1);
            assert(~isempty(representativeColumn) && ...
                any(assignedColumns == representativeColumn), ...
                'SaturationRegions:RepresentativeSlice', ...
                '%s branch %d representative slice is not in the branch.', ...
                windows(w), b);

            regionId = regionId + 1;
            satnum(w, assignedColumns) = uint16(regionId);
            regionWindowIndex(regionId) = uint16(w);
            regionWindow(regionId) = windows(w);
            regionLocalDomain(regionId) = uint16(b);
            regionPeBranch(regionId) = b;
            regionRepresentativeSlice(regionId) = representativeSlice;
            regionCurveLinearIndex(regionId) = uint32( ...
                sub2ind([nWindows, nSlices], w, representativeColumn));
            regionAssignedCellCount(regionId) = ...
                uint16(numel(assignedColumns));
        end
    end
end

nRegions = regionId;
assert(nRegions >= 1 && all(satnum(:) >= 1) && ...
    isequal(unique(double(satnum(:)))', 1:nRegions), ...
    'SaturationRegions:SATNUMCoverage', ...
    'SATNUM must cover contiguous global IDs from 1 to regionCount.');

regionWindowIndex = regionWindowIndex(1:nRegions);
regionWindow = regionWindow(1:nRegions);
regionLocalDomain = regionLocalDomain(1:nRegions);
regionPeBranch = regionPeBranch(1:nRegions);
regionRepresentativeSlice = regionRepresentativeSlice(1:nRegions);
regionCurveLinearIndex = regionCurveLinearIndex(1:nRegions);
regionAssignedCellCount = regionAssignedCellCount(1:nRegions);
regionIds = uint16((1:nRegions)');

[entryPcBar, bulkSgMax, effectiveSwi, replaySourceRow] = ...
    summarizeRegionCurves(reservoirReady, regionCurveLinearIndex);
regionTable = table(regionIds, regionWindow, regionWindowIndex, ...
    regionLocalDomain, regionPeBranch, regionRepresentativeSlice, ...
    regionCurveLinearIndex, regionAssignedCellCount, entryPcBar, ...
    bulkSgMax, effectiveSwi, replaySourceRow, ...
    'VariableNames', {'SATNUM', 'Window', 'WindowIndex', ...
    'LocalDomainId', 'PeBranchId', 'RepresentativeSliceIndex', ...
    'CurveLinearIndex', 'AssignedCellCount', 'ConnectedEntryPcBar', ...
    'BulkSgMax', 'EffectiveSwi', 'ReplaySourceRow'});

assignmentTable = buildAssignmentTable(windows, slices, satnum, ...
    regionLocalDomain, regionPeBranch, regionCurveLinearIndex);
validation = validateRegionAssignments(reservoirReady, satnum, ...
    regionCurveLinearIndex);

regions = struct( ...
    'schemaVersion', "1.0", ...
    'representation', representation, ...
    'orientation', "rows=window, columns=along-strike slice", ...
    'SATNUM', satnum, ...
    'regionCount', uint16(nRegions), ...
    'regionIds', regionIds, ...
    'regionCurveLinearIndices', regionCurveLinearIndex, ...
    'curveStorage', struct( ...
        'pc', "reservoirReady.pcCurves", ...
        'kr', "reservoirReady.krCurves", ...
        'lookupRule', "curveArray(regionCurveLinearIndices)"), ...
    'regionTable', regionTable, ...
    'assignmentTable', assignmentTable, ...
    'validation', validation);
end


function [entryPcBar, bulkSgMax, effectiveSwi, replaySourceRow] = ...
        summarizeRegionCurves(reservoirReady, curveIndices)
% Extract concise physical descriptors for every global region.

n = numel(curveIndices);
entryPcBar = nan(n, 1);
bulkSgMax = nan(n, 1);
effectiveSwi = nan(n, 1);
replaySourceRow = nan(n, 1);
for r = 1:n
    P = reservoirReady.pcCurves{double(curveIndices(r))};
    entryPcBar(r) = connectedEntryPressure(P);
    bulkSgMax(r) = double(P.bulkSgMax);
    effectiveSwi(r) = double(P.effectiveSwi);
    if isfield(P, 'representativeReplaySourceRow')
        replaySourceRow(r) = double(P.representativeReplaySourceRow);
    elseif isfield(P, 'replaySourceRow')
        replaySourceRow(r) = double(P.replaySourceRow);
    end
end
end


function assignments = buildAssignmentTable(windows, slices, satnum, ...
        regionLocalDomain, regionPeBranch, curveIndices)
% Expand the SATNUM map into a transparent one-row-per-cell table.

[nWindows, nSlices] = size(satnum);
n = nWindows * nSlices;
window = strings(n, 1);
windowIndex = zeros(n, 1, 'uint16');
sliceIndex = zeros(n, 1);
satnumColumn = zeros(n, 1, 'uint16');
localDomainId = zeros(n, 1, 'uint16');
peBranchId = nan(n, 1);
curveLinearIndex = zeros(n, 1, 'uint32');
row = 0;
for w = 1:nWindows
    for s = 1:nSlices
        row = row + 1;
        id = double(satnum(w, s));
        window(row) = windows(w);
        windowIndex(row) = uint16(w);
        sliceIndex(row) = slices(s);
        satnumColumn(row) = satnum(w, s);
        localDomainId(row) = regionLocalDomain(id);
        peBranchId(row) = regionPeBranch(id);
        curveLinearIndex(row) = curveIndices(id);
    end
end
assignments = table(window, windowIndex, sliceIndex, satnumColumn, ...
    localDomainId, peBranchId, curveLinearIndex, ...
    'VariableNames', {'Window', 'WindowIndex', 'SliceIndex', 'SATNUM', ...
    'LocalDomainId', 'PeBranchId', 'CurveLinearIndex'});
end


function validation = validateRegionAssignments(reservoirReady, satnum, ...
        curveIndices)
% Verify that SATNUM lookup reproduces every assigned physical curve.

maxPcSgMismatch = 0;
maxPcBarMismatch = 0;
maxKrSgMismatch = 0;
maxKrgMismatch = 0;
maxKrwMismatch = 0;
for cellId = 1:numel(satnum)
    regionId = double(satnum(cellId));
    sourceId = double(curveIndices(regionId));
    P = reservoirReady.pcCurves{cellId};
    Pref = reservoirReady.pcCurves{sourceId};
    K = reservoirReady.krCurves{cellId};
    Kref = reservoirReady.krCurves{sourceId};
    maxPcSgMismatch = max(maxPcSgMismatch, ...
        vectorMismatch(P.gasSaturation, Pref.gasSaturation));
    maxPcBarMismatch = max(maxPcBarMismatch, ...
        vectorMismatch(P.pcBar, Pref.pcBar));
    maxKrSgMismatch = max(maxKrSgMismatch, ...
        vectorMismatch(K.gasSaturation, Kref.gasSaturation));
    maxKrgMismatch = max(maxKrgMismatch, ...
        vectorMismatch(K.krg, Kref.krg));
    maxKrwMismatch = max(maxKrwMismatch, ...
        vectorMismatch(K.krw, Kref.krw));
end
tolerance = 1.0e-12;
passed = max([maxPcSgMismatch, maxPcBarMismatch, maxKrSgMismatch, ...
    maxKrgMismatch, maxKrwMismatch]) <= tolerance;
assert(passed, 'SaturationRegions:CurveLookupMismatch', ...
    'At least one SATNUM curve lookup does not reproduce its cell curve.');
validation = struct( ...
    'tolerance', tolerance, ...
    'maxPcSgMismatch', maxPcSgMismatch, ...
    'maxPcBarMismatch', maxPcBarMismatch, ...
    'maxKrSgMismatch', maxKrSgMismatch, ...
    'maxKrgMismatch', maxKrgMismatch, ...
    'maxKrwMismatch', maxKrwMismatch, ...
    'passed', passed);
end


function mismatch = vectorMismatch(a, b)
% Return a strict elementwise mismatch for two physical curve vectors.

a = double(a(:));
b = double(b(:));
assert(numel(a) == numel(b), 'SaturationRegions:CurveLengthMismatch', ...
    'Curves assigned to one SATNUM have different lengths.');
mismatch = max(abs(a - b), [], 'omitnan');
if isempty(mismatch)
    mismatch = 0;
end
end


function entryPcBar = connectedEntryPressure(P)
% Read the first connected invasion point after the Sg=1e-5 anchor.

sg = double(P.gasSaturation(:));
pcBar = double(P.pcBar(:));
entryId = find(sg > 1.0e-5 * (1 + 1.0e-8), 1, 'first');
assert(~isempty(entryId) && isfinite(pcBar(entryId)) && pcBar(entryId) > 0, ...
    'SaturationRegions:EntryPressure', ...
    'A Pc curve lacks a valid connected-path entry pressure.');
entryPcBar = pcBar(entryId);
end
