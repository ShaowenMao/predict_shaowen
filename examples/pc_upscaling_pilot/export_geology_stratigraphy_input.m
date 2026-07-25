function outputs = export_geology_stratigraphy_input(dataRoot, geologyId, outputDir, varargin)
%EXPORT_GEOLOGY_STRATIGRAPHY_INPUT Package one geology for reservoir setup.
%
%   OUTPUTS = EXPORT_GEOLOGY_STRATIGRAPHY_INPUT(DATAROOT, GEOLOGYID,
%   OUTPUTDIR) reads the production collapsed-cell-union PREDICT metadata
%   for all six throw windows and writes one compact geology-only MAT file.
%   The MAT file records both the original thin-layer stratigraphy and the
%   adjacent-lithology-collapsed stratigraphy actually used by PREDICT.
%
%   Name-value options:
%     'ReservoirReadyFiles' - Optional reservoir-ready MAT files to verify
%                             and link to this geology in a CSV manifest.
%     'ExpectedCaseIds'     - Expected Level-3 case IDs in the linked files.
%                             Default: [] (do not constrain case IDs).
%     'Overwrite'           - Replace existing outputs. Default: false.
%
%   The saved top-level variable is GEOLOGYSTRATIGRAPHY. It intentionally
%   excludes fault-core permeability, porosity, Pc, and Kr, which remain in
%   the case-specific reservoirReady MAT files.

parser = inputParser;
parser.addRequired('dataRoot', @(x) ischar(x) || isstring(x));
parser.addRequired('geologyId', @(x) ischar(x) || isstring(x));
parser.addRequired('outputDir', @(x) ischar(x) || isstring(x));
parser.addParameter('ReservoirReadyFiles', strings(0, 1), ...
    @(x) ischar(x) || isstring(x) || iscell(x));
parser.addParameter('ExpectedCaseIds', [], ...
    @(x) isnumeric(x) && isvector(x));
parser.addParameter('Overwrite', false, ...
    @(x) islogical(x) && isscalar(x));
parser.parse(dataRoot, geologyId, outputDir, varargin{:});
opt = parser.Results;

dataRoot = char(string(opt.dataRoot));
geologyId = string(opt.geologyId);
outputDir = char(string(opt.outputDir));
reservoirFiles = string(opt.ReservoirReadyFiles(:));
reservoirFiles(strlength(reservoirFiles) == 0) = [];

assert(isfolder(dataRoot), 'PREDICT data root does not exist: %s', dataRoot)
ensureFolder(outputDir)

[scenarioIndex, caseIndex] = parseGeologyId(geologyId);
scenarioFile = fullfile(dataRoot, 'thickness_scenario_definitions.csv');
caseFile = fullfile(dataRoot, 'geology_case_definitions.csv');
assert(isfile(scenarioFile), 'Missing scenario definition file: %s', scenarioFile)
assert(isfile(caseFile), 'Missing geology-case definition file: %s', caseFile)

scenarioTable = readtable(scenarioFile, 'TextType', 'string');
caseTable = readtable(caseFile, 'TextType', 'string');
scenarioRows = scenarioTable(double(scenarioTable.ScenarioIndex) == scenarioIndex, :);
caseRows = caseTable(double(caseTable.CaseIndex) == caseIndex, :);
assert(height(scenarioRows) == 6, ...
    'Expected six scenario rows for %s; found %d.', geologyId, height(scenarioRows))
assert(height(caseRows) == 1, ...
    'Expected one geology-case row for %s; found %d.', geologyId, height(caseRows))

windowLabels = "famp" + string((1:6).');
assert(all(ismember(windowLabels, scenarioRows.Window)), ...
    'Scenario %d does not define all six windows.', scenarioIndex)

scenarioLabel = scenarioRows.ScenarioLabel(1);
scenarioName = scenarioRows.ScenarioName(1);
caseLabel = caseRows.CaseLabel(1);
faultingDepth = double(caseRows.FaultingDepth(1));
sandVcl = double(caseRows.SandVcl(1));
clayVcl = double(caseRows.ClayVcl(1));

windows = repmat(struct(), 6, 1);
sourceFiles = strings(6, 1);
ensembleSizes = zeros(6, 1);
seedBases = zeros(6, 1);
correlationCoefficients = zeros(6, 1);
smearOverlapRules = strings(6, 1);
collapseFlags = false(6, 1);
summaryRows = cell(12, 17);
summaryRow = 0;

for w = 1:6
    window = windowLabels(w);
    scenarioWindow = scenarioRows(scenarioRows.Window == window, :);
    assert(height(scenarioWindow) == 1, ...
        'Expected one row for %s / %s.', scenarioLabel, window)

    base = getBaseWindowOptions(window);
    originalPatterns = {char(scenarioWindow.FWPattern), ...
                        char(scenarioWindow.HWPattern)};
    sourceFile = fullfile(dataRoot, 'data', char(scenarioLabel), ...
        char(window), char(caseLabel), 'predict_runs.mat');
    assert(isfile(sourceFile), 'Missing PREDICT result: %s', sourceFile)
    sourceFiles(w) = string(sourceFile);

    loaded = load(sourceFile, 'checkpointInfo');
    assert(isfield(loaded, 'checkpointInfo'), ...
        'Missing checkpointInfo in %s.', sourceFile)
    checkpoint = loaded.checkpointInfo;
    validateCheckpointIdentity(checkpoint, scenarioIndex, scenarioLabel, ...
        scenarioName, caseIndex, caseLabel, window);

    windows(w).windowIndex = w;
    windows(w).windowLabel = window;
    windows(w).faultingDepth_m = faultingDepth;
    windows(w).faultDip_deg = base.faultDip_deg;
    windows(w).maximumSandPermeability_mD = base.maximumSandPermeability_mD;
    if isfield(base, 'totalFaultThickness_m')
        windows(w).totalFaultThickness_m = base.totalFaultThickness_m;
    else
        windows(w).totalFaultThickness_m = [];
    end

    wallNames = {'footwall', 'hangingwall'};
    checkpointPrefixes = {'FW', 'HW'};
    for side = 1:2
        wall = buildWallRecord(originalPatterns{side}, ...
            base.thickness_m{side}, base.burialDepth_m{side}, ...
            sandVcl, clayVcl, base.dip_deg(side));
        validateCheckpointWall(checkpoint, checkpointPrefixes{side}, wall, sourceFile)
        windows(w).(wallNames{side}) = wall;

        summaryRow = summaryRow + 1;
        summaryRows(summaryRow, :) = { ...
            geologyId, scenarioIndex, scenarioLabel, caseIndex, caseLabel, ...
            window, string(wallNames{side}), wall.dip_deg, ...
            string(wall.original.pattern), wall.original.layerCount, ...
            vectorText(wall.original.thickness_m), ...
            vectorText(wall.original.vcl_fraction), ...
            vectorText(wall.original.burialDepth_m), ...
            string(wall.collapsed.pattern), wall.collapsed.layerCount, ...
            vectorText(wall.collapsed.thickness_m), ...
            vectorText(wall.collapsed.burialDepth_m)};
    end

    info = whos('-file', sourceFile, 'perms');
    assert(isscalar(info) && numel(info.size) == 2 && info.size(2) == 3, ...
        'Expected an N-by-3 perms array in %s.', sourceFile)
    ensembleSizes(w) = info.size(1);
    seedBases(w) = double(checkpoint.SeedBase);
    correlationCoefficients(w) = double(checkpoint.CorrCoef);
    smearOverlapRules(w) = string(checkpoint.SmearOverlapRule);
    collapseFlags(w) = logical(checkpoint.CollapseAdjacentLithology);
end

assert(all(ensembleSizes == ensembleSizes(1)), ...
    'The six PREDICT libraries do not have a common ensemble size.')
assert(all(correlationCoefficients == correlationCoefficients(1)), ...
    'The six windows do not have a common correlation coefficient.')
assert(isscalar(unique(smearOverlapRules)), ...
    'The six windows do not have a common smear-overlap rule.')
assert(all(collapseFlags), ...
    'At least one source window was not generated with collapsed lithology.')

geologyStratigraphy = struct();
geologyStratigraphy.schemaVersion = "1.1";
geologyStratigraphy.contentType = "geology_stratigraphy";
geologyStratigraphy.geologyId = geologyId;
geologyStratigraphy.scenario = struct( ...
    'index', scenarioIndex, ...
    'label', scenarioLabel, ...
    'name', scenarioName);
geologyStratigraphy.geologyCase = struct( ...
    'index', caseIndex, ...
    'label', caseLabel, ...
    'faultingDepth_m', faultingDepth, ...
    'sandVcl_fraction', sandVcl, ...
    'clayVcl_fraction', clayVcl);
geologyStratigraphy.windowLabels = windowLabels;
geologyStratigraphy.windows = windows;
geologyStratigraphy.predictConfiguration = struct( ...
    'ensembleSizePerWindow', ensembleSizes, ...
    'seedBasePerWindow', seedBases, ...
    'correlationCoefficient', correlationCoefficients(1), ...
    'smearOverlapRule', smearOverlapRules(1), ...
    'collapseAdjacentLithology', true);
geologyStratigraphy.units = struct( ...
    'length', "m", ...
    'angle', "degree", ...
    'permeability', "mD", ...
    'vcl', "fraction");
geologyStratigraphy.qa = struct( ...
    'windowCount', numel(windowLabels), ...
    'allCheckpointIdentitiesMatch', true, ...
    'allCollapsedInputsMatchCheckpoints', true, ...
    'allThicknessSumsConserved', true, ...
    'reservoirMappingContractValidated', true);

geologyStratigraphy.reservoirMapping = struct( ...
    'schemaVersion', "1.0", ...
    'layerRepresentation', "original", ...
    'windowOrder', windowLabels, ...
    'gridGeometry', "fixed_existing_grid", ...
    'movesGridBoundaries', false, ...
    'collapsesAdjacentLayers', false, ...
    'existingReservoirPropertiesUnchanged', true, ...
    'propertySelectionKey', "lithology", ...
    'allowedLithologies', ["sand", "clay"], ...
    'requiredFieldPaths', [ ...
        "windows(w).footwall.original.thickness_m"; ...
        "windows(w).footwall.original.lithology"; ...
        "windows(w).hangingwall.original.thickness_m"; ...
        "windows(w).hangingwall.original.lithology"]);

hashPayload = buildReservoirMappingHashPayload( ...
    geologyId, windowLabels, windows, ...
    geologyStratigraphy.reservoirMapping.schemaVersion);
geologyStratigraphy.geologyHashAlgorithm = ...
    "SHA-256 over canonical reservoir-stratigraphy mapping payload: " + ...
    "geologyId, ordered windows, and original FW/HW lithology/thickness";
geologyStratigraphy.geologyHash = sha256Text(jsonencode(hashPayload));
geologyStratigraphy.provenance = struct( ...
    'generatedAtUtc', string(datetime('now', 'TimeZone', 'UTC', ...
        'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX')), ...
    'generator', string(mfilename('fullpath')), ...
    'generatorSha256', sha256Text(fileread([mfilename('fullpath') '.m'])), ...
    'sourceDataRoot', string(dataRoot), ...
    'scenarioDefinitionFile', string(scenarioFile), ...
    'geologyCaseDefinitionFile', string(caseFile), ...
    'sourcePredictFiles', sourceFiles, ...
    'sourceGitCommit', currentGitCommit(), ...
    'sourceGitWorkingTreeDirty', isGitWorkingTreeDirty());

summaryTable = cell2table(summaryRows, 'VariableNames', { ...
    'GeologyId', 'ScenarioIndex', 'ScenarioLabel', 'CaseIndex', ...
    'GeologyCaseLabel', 'Window', 'Wall', 'DipDeg', ...
    'OriginalPattern', 'OriginalLayerCount', 'OriginalThicknessM', ...
    'OriginalVclFraction', 'OriginalBurialDepthM', 'CollapsedPattern', ...
    'CollapsedLayerCount', 'CollapsedThicknessM', 'CollapsedBurialDepthM'});

matFile = fullfile(outputDir, sprintf('geology_stratigraphy_%s.mat', geologyId));
summaryFile = fullfile(outputDir, sprintf( ...
    'geology_stratigraphy_summary_%s.csv', geologyId));
linkFile = fullfile(outputDir, sprintf( ...
    'geology_fault_case_links_%s.csv', geologyId));
guardOutput(matFile, opt.Overwrite)
guardOutput(summaryFile, opt.Overwrite)
if ~isempty(reservoirFiles)
    guardOutput(linkFile, opt.Overwrite)
end

linkTable = validateReservoirReadyLinks(reservoirFiles, geologyId, ...
    geologyStratigraphy.geologyHash, opt.ExpectedCaseIds);
save(matFile, 'geologyStratigraphy', '-v7');
writetable(summaryTable, summaryFile)
if ~isempty(linkTable)
    writetable(linkTable, linkFile)
else
    linkFile = '';
end

outputs = struct( ...
    'geologyStratigraphy', geologyStratigraphy, ...
    'summaryTable', summaryTable, ...
    'linkTable', linkTable, ...
    'matFile', string(matFile), ...
    'summaryCsv', string(summaryFile), ...
    'linkCsv', string(linkFile));

fprintf('Saved geology stratigraphy: %s\n', matFile)
fprintf('Geology hash: %s\n', geologyStratigraphy.geologyHash)
fprintf('Validated %d linked reservoir-ready files.\n', height(linkTable))
end


function [scenarioIndex, caseIndex] = parseGeologyId(geologyId)
% Parse the canonical sNN_cNNN geology identifier.

tokens = regexp(char(geologyId), '^s(\d+)_c(\d+)$', 'tokens', 'once');
assert(~isempty(tokens), ...
    'GeologyId must use the canonical sNN_cNNN format; received %s.', geologyId)
scenarioIndex = str2double(tokens{1});
caseIndex = str2double(tokens{2});
end


function payload = buildReservoirMappingHashPayload(geologyId, windowLabels, windows, schemaVersion)
% Restrict the downstream hash to fields that control fixed-grid mapping.

windowPayload = repmat(struct(), numel(windows), 1);
for w = 1:numel(windows)
    windowPayload(w).windowIndex = windows(w).windowIndex;
    windowPayload(w).windowLabel = windows(w).windowLabel;
    windowPayload(w).footwall = struct( ...
        'thickness_m', windows(w).footwall.original.thickness_m, ...
        'lithology', windows(w).footwall.original.lithology);
    windowPayload(w).hangingwall = struct( ...
        'thickness_m', windows(w).hangingwall.original.thickness_m, ...
        'lithology', windows(w).hangingwall.original.lithology);
end
payload = struct( ...
    'schemaVersion', schemaVersion, ...
    'geologyId', geologyId, ...
    'windowLabels', windowLabels, ...
    'windows', windowPayload);
end


function wall = buildWallRecord(pattern, thickness, burialDepth, sandVcl, clayVcl, dip)
% Build original and collapsed wall records using the production rule.

pattern = upper(char(pattern));
thickness = rowVector(thickness);
burialDepth = rowVector(burialDepth);
assert(numel(pattern) == numel(thickness) && ...
       numel(thickness) == numel(burialDepth), ...
    'Pattern, thickness, and burial-depth lengths must match.')
vcl = patternToVcl(pattern, sandVcl, clayVcl);
[collapsedPattern, collapsedThickness, collapsedVcl, ...
    collapsedBurialDepth, sourceLayerIndices, groupIndex] = ...
    collapseAdjacentLayers(pattern, thickness, vcl, burialDepth);

wall = struct();
wall.dip_deg = double(dip);
wall.original = struct( ...
    'pattern', string(pattern), ...
    'layerCount', numel(pattern), ...
    'layerIndex', 1:numel(pattern), ...
    'lithology', patternToNames(pattern), ...
    'thickness_m', thickness, ...
    'vcl_fraction', vcl, ...
    'burialDepth_m', burialDepth, ...
    'collapsedGroupIndex', groupIndex);
wall.collapsed = struct( ...
    'pattern', string(collapsedPattern), ...
    'layerCount', numel(collapsedPattern), ...
    'layerIndex', 1:numel(collapsedPattern), ...
    'lithology', patternToNames(collapsedPattern), ...
    'thickness_m', collapsedThickness, ...
    'vcl_fraction', collapsedVcl, ...
    'burialDepth_m', collapsedBurialDepth, ...
    'sourceLayerIndices', {sourceLayerIndices});
assert(abs(sum(thickness) - sum(collapsedThickness)) < 1e-10, ...
    'Adjacent-layer collapse did not conserve total thickness.')
end


function [patternOut, thicknessOut, vclOut, burialDepthOut, sourceIds, groupIndex] = ...
        collapseAdjacentLayers(pattern, thickness, vcl, burialDepth)
% Reproduce the production adjacent-lithology collapse exactly.

starts = [1, find(diff(double(pattern)) ~= 0) + 1, numel(pattern) + 1];
nGroups = numel(starts) - 1;
patternOut = repmat('S', 1, nGroups);
thicknessOut = zeros(1, nGroups);
vclOut = zeros(1, nGroups);
burialDepthOut = zeros(1, nGroups);
sourceIds = cell(1, nGroups);
groupIndex = zeros(1, numel(pattern));
for g = 1:nGroups
    ids = starts(g):(starts(g + 1) - 1);
    sourceIds{g} = ids;
    groupIndex(ids) = g;
    patternOut(g) = pattern(ids(1));
    thicknessOut(g) = sum(thickness(ids));
    vclOut(g) = vcl(ids(1));
    burialDepthOut(g) = sum(burialDepth(ids) .* thickness(ids)) / ...
        sum(thickness(ids));
end
end


function validateCheckpointIdentity(checkpoint, scenarioIndex, scenarioLabel, ...
        scenarioName, caseIndex, caseLabel, window)
% Ensure the source checkpoint belongs to the requested geology and window.

assert(double(checkpoint.ScenarioIndex) == scenarioIndex)
assert(string(checkpoint.ScenarioLabel) == scenarioLabel)
assert(string(checkpoint.ScenarioName) == scenarioName)
assert(double(checkpoint.CaseIndex) == caseIndex)
assert(string(checkpoint.CaseLabel) == caseLabel)
assert(strcmpi(string(checkpoint.Window), window))
end


function validateCheckpointWall(checkpoint, prefix, wall, sourceFile)
% Check reconstructed original/collapsed wall inputs against saved metadata.

originalPattern = string(checkpoint.(['Original' prefix 'Pattern']));
collapsedPattern = string(checkpoint.([prefix 'Pattern']));
checkpointThickness = numericVector(checkpoint.([prefix 'Thickness']));
checkpointBurialDepth = numericVector(checkpoint.([prefix 'Zmax']));
assert(originalPattern == wall.original.pattern, ...
    'Original %s pattern mismatch in %s.', prefix, sourceFile)
assert(collapsedPattern == wall.collapsed.pattern, ...
    'Collapsed %s pattern mismatch in %s.', prefix, sourceFile)
assert(vectorsMatch(checkpointThickness, wall.collapsed.thickness_m), ...
    'Collapsed %s thickness mismatch in %s.', prefix, sourceFile)
assert(vectorsMatch(checkpointBurialDepth, wall.collapsed.burialDepth_m), ...
    'Collapsed %s burial-depth mismatch in %s.', prefix, sourceFile)
end


function linkTable = validateReservoirReadyLinks(files, geologyId, geologyHash, expectedCaseIds)
% Verify case-specific MAT files and build a portable linkage manifest.

if isempty(files)
    linkTable = table();
    return
end

n = numel(files);
rows = cell(n, 10);
foundCaseIds = zeros(n, 1);
for i = 1:n
    file = char(files(i));
    assert(isfile(file), 'Reservoir-ready MAT file does not exist: %s', file)
    loaded = load(file, 'reservoirReady');
    assert(isfield(loaded, 'reservoirReady'), ...
        'Missing reservoirReady in %s.', file)
    reservoirReady = loaded.reservoirReady;
    assert(isfield(reservoirReady, 'geologyId') && ...
        string(reservoirReady.geologyId) == geologyId, ...
        'GeologyId mismatch in %s.', file)
    assert(isfield(reservoirReady, 'level3CaseId'), ...
        'Missing level3CaseId in %s.', file)
    caseId = double(reservoirReady.level3CaseId);
    foundCaseIds(i) = caseId;
    representation = "unspecified";
    if isfield(reservoirReady, 'pcRepresentation')
        representation = string(reservoirReady.pcRepresentation);
    end
    caseName = "";
    if isfield(reservoirReady, 'level3CaseName')
        caseName = string(reservoirReady.level3CaseName);
    end
    fileInfo = dir(file);
    [parent, name, ext] = fileparts(file);
    rows(i, :) = {geologyId, geologyHash, caseId, caseName, ...
        representation, string([name ext]), string(parent), ...
        double(fileInfo.bytes), true, true};
end

if ~isempty(expectedCaseIds)
    expectedCaseIds = unique(double(expectedCaseIds(:)));
    assert(isequal(unique(foundCaseIds), expectedCaseIds), ...
        'Linked MAT files do not contain exactly the expected Level-3 case IDs.')
end

linkTable = cell2table(rows, 'VariableNames', { ...
    'GeologyId', 'GeologyHash', 'Level3CaseId', 'Level3CaseName', ...
    'PcRepresentation', 'FaultInputFile', 'FaultInputFolder', ...
    'FileSizeBytes', 'GeologyIdVerified', 'ReadableVerified'});
linkTable = sortrows(linkTable, {'Level3CaseId', 'PcRepresentation'});
end


function tf = vectorsMatch(a, b)
% Compare checkpoint vectors after compact metadata formatting.

a = rowVector(a);
b = rowVector(b);
tf = numel(a) == numel(b) && all(abs(a - b) <= 1e-4 .* max(1, abs(b)));
end


function values = numericVector(raw)
% Convert numeric or compact space-delimited checkpoint vectors to doubles.

if isnumeric(raw)
    values = rowVector(raw);
else
    values = sscanf(char(string(raw)), '%f').';
end
end


function names = patternToNames(pattern)
% Convert compact S/C codes to explicit lithology names.

names = strings(1, numel(pattern));
names(pattern == 'S') = "sand";
names(pattern == 'C') = "clay";
end


function vcl = patternToVcl(pattern, sandVcl, clayVcl)
% Convert compact S/C codes to the geology-specific Vcl vector.

vcl = nan(1, numel(pattern));
vcl(pattern == 'S') = sandVcl;
vcl(pattern == 'C') = clayVcl;
assert(all(isfinite(vcl)), 'Patterns may contain only S and C.')
end


function text = vectorText(values)
% Format a short numeric vector for human-readable CSV review.

text = string(strjoin(compose('%.10g', rowVector(values)), ' '));
end


function row = rowVector(values)
% Normalize scalar/vector inputs to a double row vector.

row = double(values(:).');
end


function guardOutput(file, overwrite)
% Prevent accidental replacement unless explicitly requested.

assert(overwrite || ~isfile(file), ...
    'Output already exists; pass Overwrite=true to replace it: %s', file)
end


function ensureFolder(folder)
% Create an output folder when needed.

if ~isfolder(folder)
    mkdir(folder)
end
end


function value = sha256Text(textValue)
% Compute a deterministic SHA-256 digest using MATLAB's Java runtime.

digest = java.security.MessageDigest.getInstance('SHA-256');
digest.update(uint8(unicode2native(char(textValue), 'UTF-8')));
bytes = typecast(digest.digest(), 'uint8');
value = lower(string(reshape(dec2hex(bytes, 2).', 1, [])));
end


function commit = currentGitCommit()
% Record the source commit without making Git a runtime requirement.

repoRoot = currentRepoRoot();
[status, output] = system(sprintf('git -C "%s" rev-parse HEAD', repoRoot));
if status == 0
    commit = string(strtrim(output));
else
    commit = "unavailable";
end
end


function dirty = isGitWorkingTreeDirty()
% Flag uncommitted source state so generated artifacts remain auditable.

repoRoot = currentRepoRoot();
[status, output] = system(sprintf( ...
    'git -C "%s" status --porcelain --untracked-files=all', repoRoot));
dirty = status ~= 0 || strlength(strtrim(string(output))) > 0;
end


function repoRoot = currentRepoRoot()
% Locate the repository root from examples/pc_upscaling_pilot.

thisFile = mfilename('fullpath');
repoRoot = fileparts(fileparts(fileparts(thisFile)));
end


function opt = getBaseWindowOptions(window)
% Return the fixed GOM window geometry used by the production driver.

opt.maximumSandPermeability_mD = 175;
switch lower(char(window))
    case 'famp1'
        opt.thickness_m = {[115.6143 28.8949], ...
            [37.6113 37.6861 37.6113 31.6005]};
        opt.dip_deg = [0, -12.0136];
        opt.faultDip_deg = 41.6345;
        opt.burialDepth_m = {[1912 1861], [1934 1909 1884 1860]};
    case 'famp2'
        opt.thickness_m = {[36.9255 35.8537 36.8537 36.3111], ...
            [36.5042 36.5042 36.4314 36.5042]};
        opt.dip_deg = [0, -13.8951];
        opt.faultDip_deg = 43.2508;
        opt.burialDepth_m = {[1837.5 1812.5 1787.5 1762.5], ...
            [1837.5 1812.5 1787.5 1762.5]};
    case 'famp3'
        opt.thickness_m = {[35.8537 35.8537 35.8537 35.8537], ...
            [35.8537 35.8537 35.8537 35.8537]};
        opt.dip_deg = [0, -9.7683];
        opt.faultDip_deg = 43.8;
        opt.burialDepth_m = {[1738.8 1713.8 1688.8 1663.8], ...
            [1738.8 1713.8 1688.8 1663.8]};
    case 'famp4'
        opt.thickness_m = {[35.8537 35.8537 35.8537 35.9255], ...
            [35.8537 35.8537 35.8537 35.9255]};
        opt.dip_deg = [0, -4.9456];
        opt.faultDip_deg = 44.1811;
        opt.burialDepth_m = {[1638.8 1613.8 1588.8 1563.8], ...
            [1638.8 1613.8 1588.8 1563.8]};
    case 'famp5'
        opt.thickness_m = {[35.8537 35.8537 35.8537 35.8537], ...
            [37.4901 35.2847 35.3553 35.2847]};
        opt.dip_deg = [0, -5.2221];
        opt.faultDip_deg = 45.0685;
        opt.burialDepth_m = {[1538.8 1513.82 1488.75 1463.99], ...
            [1538.8 1513.82 1488.75 1463.99]};
    case 'famp6'
        opt.thickness_m = {[28.2932 33.1042 33.1699 33.1042], 127.6715};
        opt.dip_deg = [0, -5];
        opt.faultDip_deg = 46.0685;
        opt.burialDepth_m = {[1440.6 1417.5 1392.5 1367.5], 1400};
    otherwise
        error('Unsupported throw window: %s', window)
end
end
