function outputs = regenerate_reservoir_ready_with_saturation_regions( ...
        inputFiles, outputDir)
%REGENERATE_RESERVOIR_READY_WITH_SATURATION_REGIONS Upgrade existing MATs.
%
%   OUTPUTS = REGENERATE_RESERVOIR_READY_WITH_SATURATION_REGIONS(
%   INPUTFILES, OUTPUTDIR) reads validated reservoirReady MAT files, adds
%   explicit global saturation-region metadata, and writes new MAT files
%   without modifying the sources. It also exports human-readable SATNUM,
%   region-definition, assignment, and QA CSV files.
%
%   The regenerated files expose reservoirReady.saturationRegions and are
%   intended as the canonical downstream inputs for MRST integration.

inputFiles = normalizeInputs(inputFiles);
assert(~isempty(inputFiles), 'SaturationRegionsUpgrade:EmptyInput', ...
    'At least one reservoir-ready MAT file is required.');
ensureFolder(outputDir);
tableDir = fullfile(outputDir, 'tables');
ensureFolder(tableDir);

nFiles = numel(inputFiles);
matFiles = strings(nFiles, 1);
qaRows = cell(nFiles, 9);

for i = 1:nFiles
    sourceFile = string(inputFiles{i});
    assert(isscalar(sourceFile) && isfile(sourceFile), ...
        'SaturationRegionsUpgrade:MissingInput', ...
        'Input MAT file not found: %s', sourceFile);
    loaded = load(sourceFile, 'reservoirReady');
    assert(isfield(loaded, 'reservoirReady'), ...
        'SaturationRegionsUpgrade:MissingVariable', ...
        'Input does not contain reservoirReady: %s', sourceFile);
    reservoirReady = loaded.reservoirReady;

    representation = inferRepresentation(reservoirReady);
    reservoirReady.pcRepresentation = representation;
    reservoirReady.schemaVersion = "1.4";
    reservoirReady.saturationRegions = ...
        build_saturation_region_metadata(reservoirReady);
    reservoirReady.provenance.saturationRegionsSource = sourceFile;
    reservoirReady.provenance.saturationRegionsCreatedAt = ...
        string(datetime('now', 'TimeZone', 'UTC'));

    geologyId = string(reservoirReady.geologyId);
    caseId = double(reservoirReady.level3CaseId);
    token = sprintf('%s_case%02d', char(geologyId), caseId);
    targetFile = fullfile(outputDir, sprintf( ...
        'reservoir_ready_%s_%s.mat', char(representation), token));
    assert(~pathsEqual(sourceFile, targetFile), ...
        'SaturationRegionsUpgrade:SourceOverwrite', ...
        'Output must not overwrite the validated source MAT file.');
    save(targetFile, 'reservoirReady', '-v7.3');
    matFiles(i) = string(targetFile);

    prefix = sprintf('%s_%s', char(representation), token);
    writetable(reservoirReady.saturationRegions.regionTable, ...
        fullfile(tableDir, ['saturation_region_definitions_', prefix, '.csv']));
    writetable(reservoirReady.saturationRegions.assignmentTable, ...
        fullfile(tableDir, ['saturation_region_assignments_', prefix, '.csv']));
    writeSatnumMap(reservoirReady, fullfile(tableDir, ...
        ['SATNUM_map_', prefix, '.csv']));

    regions = reservoirReady.saturationRegions;
    qaRows(i, :) = {geologyId, caseId, representation, ...
        numel(reservoirReady.windowLabels), ...
        numel(reservoirReady.sliceIndices), ...
        numel(regions.SATNUM), double(regions.regionCount), ...
        logical(regions.validation.passed), string(targetFile)};
    fprintf('Saved %s with %d explicit saturation regions: %s\n', ...
        representation, double(regions.regionCount), targetFile);
end

qa = cell2table(qaRows, 'VariableNames', { ...
    'GeologyId', 'Level3CaseId', 'PcRepresentation', 'WindowCount', ...
    'SliceCount', 'AssignedCellCount', 'SaturationRegionCount', ...
    'CurveLookupValidationPassed', 'MatFile'});
qaCsv = fullfile(tableDir, 'saturation_region_regeneration_qa.csv');
writetable(qa, qaCsv);
outputs = struct('matFiles', matFiles, 'qaTable', qa, ...
    'qaCsv', string(qaCsv), 'tableDir', string(tableDir));
end


function representation = inferRepresentation(reservoirReady)
% Infer only when legacy full-slice files lack an explicit representation.

if isfield(reservoirReady, 'pcRepresentation')
    representation = lower(string(reservoirReady.pcRepresentation));
elseif isfield(reservoirReady, 'peBranchModel')
    representation = "pe_branch_medoid";
else
    representation = "full_slice";
end
assert(any(representation == ["full_slice", "pe_branch_medoid"]), ...
    'SaturationRegionsUpgrade:Representation', ...
    'Unsupported Pc representation: %s.', representation);
end


function writeSatnumMap(reservoirReady, outputFile)
% Write one row per along-strike slice and one SATNUM column per window.

SATNUM = reservoirReady.saturationRegions.SATNUM;
windowNames = matlab.lang.makeValidName( ...
    cellstr(string(reservoirReady.windowLabels(:))));
mapTable = array2table(double(SATNUM'), 'VariableNames', windowNames);
mapTable = addvars(mapTable, double(reservoirReady.sliceIndices(:)), ...
    'Before', 1, 'NewVariableNames', 'SliceIndex');
writetable(mapTable, outputFile);
end


function inputs = normalizeInputs(value)
% Convert scalar or array path inputs into a cell vector.

if iscell(value)
    inputs = value(:);
elseif isstring(value) || ischar(value)
    inputs = cellstr(string(value(:)));
else
    error('SaturationRegionsUpgrade:UnsupportedInput', ...
        'inputFiles must be paths or a cell array of paths.');
end
end


function tf = pathsEqual(a, b)
% Compare normalized paths without adding a Java runtime dependency.

a = replace(string(a), '/', filesep);
b = replace(string(b), '/', filesep);
tf = strcmpi(char(a), char(b));
end


function ensureFolder(folderPath)
% Create a folder if it does not already exist.

if ~isfolder(folderPath)
    mkdir(folderPath);
end
end
