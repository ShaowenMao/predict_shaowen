function outputs = build_production_geology_stratigraphy_package( ...
        dataRoot, runRoot, outputRoot, varargin)
%BUILD_PRODUCTION_GEOLOGY_STRATIGRAPHY_PACKAGE Export all geology companions.
%
%   OUTPUTS = BUILD_PRODUCTION_GEOLOGY_STRATIGRAPHY_PACKAGE(DATAROOT,
%   RUNROOT, OUTPUTROOT) creates one hashed stratigraphy MAT file for each
%   geology in the production case-work manifest. Every geology is checked
%   against its six frozen PREDICT checkpoints and linked to exactly ten
%   validated full-slice fault-property MAT files.
%
%   The package contains:
%     geologies/<geologyId>/geology_stratigraphy_<geologyId>.mat
%     geologies/<geologyId>/geology_stratigraphy_summary_<geologyId>.csv
%     geologies/<geologyId>/geology_fault_case_links_<geologyId>.csv
%     geology_stratigraphy_manifest.csv
%     geology_fault_case_links.csv
%     SHA256SUMS
%     geology_stratigraphy.done.json
%
%   Name-value options:
%     'ExpectedGeologyCount' - Required geology count. Default: 162.
%     'ExpectedCaseIds'      - Required cases per geology. Default: 1:10.
%     'Overwrite'            - Replace existing package files. Default:
%                              false.

parser = inputParser;
parser.addRequired('dataRoot', @(x) ischar(x) || isstring(x));
parser.addRequired('runRoot', @(x) ischar(x) || isstring(x));
parser.addRequired('outputRoot', @(x) ischar(x) || isstring(x));
parser.addParameter('ExpectedGeologyCount', 162, ...
    @(x) isnumeric(x) && isscalar(x) && x >= 1 && mod(x, 1) == 0);
parser.addParameter('ExpectedCaseIds', 1:10, ...
    @(x) isnumeric(x) && isvector(x) && ~isempty(x));
parser.addParameter('Overwrite', false, ...
    @(x) islogical(x) && isscalar(x));
parser.parse(dataRoot, runRoot, outputRoot, varargin{:});
opt = parser.Results;

dataRoot = char(string(dataRoot));
runRoot = char(string(runRoot));
outputRoot = char(string(outputRoot));
expectedCaseIds = unique(double(opt.ExpectedCaseIds(:))).';

assert(isfolder(dataRoot), 'PREDICT data root does not exist: %s', dataRoot)
assert(isfolder(runRoot), 'Production run root does not exist: %s', runRoot)

caseWorkFile = fullfile(runRoot, 'case_work_manifest', 'case_work.csv');
completionGateFile = fullfile(runRoot, 'case_completion_gate.json');
caseResultRoot = fullfile(runRoot, 'case_results');
assert(isfile(caseWorkFile), 'Missing case-work manifest: %s', caseWorkFile)
assert(isfile(completionGateFile), ...
    'Missing final case-completion gate: %s', completionGateFile)
assert(isfolder(caseResultRoot), ...
    'Missing production case-result root: %s', caseResultRoot)

completionGate = jsondecode(fileread(completionGateFile));
assert(string(completionGate.status) == "complete", ...
    'The final case-completion gate has not passed.')
assert(double(completionGate.error_count) == 0, ...
    'The final case-completion gate contains errors.')

caseWork = readtable(caseWorkFile, 'TextType', 'string');
requiredVariables = [ ...
    "geology_id", "scenario_index", "scenario_label", "case_id", ...
    "case_name", "case_category", "case_relative_path"];
assert(all(ismember(requiredVariables, string(caseWork.Properties.VariableNames))), ...
    'The case-work manifest does not contain the required columns.')

caseWork.geology_id = string(caseWork.geology_id);
geologyIds = unique(caseWork.geology_id, 'stable');
assert(numel(geologyIds) == opt.ExpectedGeologyCount, ...
    'Expected %d geologies; found %d.', ...
    opt.ExpectedGeologyCount, numel(geologyIds))
expectedCaseCount = opt.ExpectedGeologyCount * numel(expectedCaseIds);
assert(height(caseWork) == expectedCaseCount, ...
    'Expected %d production cases; found %d.', ...
    expectedCaseCount, height(caseWork))
assert(double(completionGate.expected_geology_count) == ...
    opt.ExpectedGeologyCount)
assert(double(completionGate.expected_case_count) == expectedCaseCount)
assert(double(completionGate.result_markers_validated) == expectedCaseCount)

completionFile = fullfile(outputRoot, ...
    'geology_stratigraphy.done.json');
if isfile(completionFile) && ~opt.Overwrite
    error(['A completed package already exists. Validate and reuse it, or ', ...
        'pass Overwrite=true explicitly: %s'], completionFile)
end
ensureFolder(outputRoot)
geologyRoot = fullfile(outputRoot, 'geologies');
ensureFolder(geologyRoot)

manifestRows = cell(numel(geologyIds), 16);
allLinkTables = cell(numel(geologyIds), 1);
packageFiles = strings(0, 1);

for g = 1:numel(geologyIds)
    geologyId = geologyIds(g);
    geologyRows = caseWork(caseWork.geology_id == geologyId, :);
    geologyRows = sortrows(geologyRows, 'case_id');
    foundCaseIds = double(geologyRows.case_id(:)).';
    assert(isequal(foundCaseIds, expectedCaseIds), ...
        '%s does not contain exactly the expected case IDs.', geologyId)

    faultFiles = strings(height(geologyRows), 1);
    for c = 1:height(geologyRows)
        relativeCasePath = strrep( ...
            char(geologyRows.case_relative_path(c)), '/', filesep);
        reservoirFolder = fullfile(caseResultRoot, relativeCasePath, ...
            'kr', 'reservoir_ready');
        listing = dir(fullfile(reservoirFolder, '*.mat'));
        assert(numel(listing) == 1, ...
            'Expected one full-slice fault-property MAT in %s; found %d.', ...
            reservoirFolder, numel(listing))
        faultFiles(c) = string(fullfile(listing(1).folder, listing(1).name));
    end

    geologyOutputDir = fullfile(geologyRoot, char(geologyId));
    ensureFolder(geologyOutputDir)
    geologyOutput = export_geology_stratigraphy_input( ...
        dataRoot, geologyId, geologyOutputDir, ...
        'ReservoirReadyFiles', faultFiles, ...
        'ExpectedCaseIds', expectedCaseIds, ...
        'Overwrite', opt.Overwrite);

    linkTable = geologyOutput.linkTable;
    [found, locations] = ismember( ...
        double(linkTable.Level3CaseId), double(geologyRows.case_id));
    assert(all(found), 'Could not join all %s case links.', geologyId)
    linkTable.CaseCategory = geologyRows.case_category(locations);
    linkTable.CaseRelativePath = geologyRows.case_relative_path(locations);
    linkTable.CaseCompletionGateValidated = true(height(linkTable), 1);
    allLinkTables{g} = linkTable;

    matInfo = dir(char(geologyOutput.matFile));
    assert(isscalar(matInfo), ...
        'Could not inspect generated stratigraphy MAT for %s.', geologyId)
    matSha256 = sha256File(geologyOutput.matFile);
    geology = geologyOutput.geologyStratigraphy;
    manifestRows(g, :) = { ...
        geologyId, ...
        double(geology.scenario.index), ...
        string(geology.scenario.label), ...
        string(geology.scenario.name), ...
        double(geology.geologyCase.index), ...
        string(geology.geologyCase.label), ...
        double(geology.geologyCase.faultingDepth_m), ...
        double(geology.geologyCase.sandVcl_fraction), ...
        double(geology.geologyCase.clayVcl_fraction), ...
        string(geology.geologyHash), ...
        height(linkTable), ...
        string(relativeToRoot(geologyOutput.matFile, outputRoot)), ...
        matSha256, ...
        double(matInfo.bytes), ...
        string(relativeToRoot(geologyOutput.summaryCsv, outputRoot)), ...
        string(relativeToRoot(geologyOutput.linkCsv, outputRoot))};
    packageFiles = [packageFiles; ...
        string(geologyOutput.matFile); ...
        string(geologyOutput.summaryCsv); ...
        string(geologyOutput.linkCsv)]; %#ok<AGROW>
end

manifest = cell2table(manifestRows, 'VariableNames', { ...
    'GeologyId', 'ScenarioIndex', 'ScenarioLabel', 'ScenarioName', ...
    'GeologyCaseIndex', 'GeologyCaseLabel', 'FaultingDepthM', ...
    'SandVclFraction', 'ClayVclFraction', 'GeologyHash', ...
    'LinkedFaultCaseCount', 'StratigraphyMat', ...
    'StratigraphyMatSha256', 'StratigraphyMatBytes', ...
    'LayerSummaryCsv', 'FaultCaseLinkCsv'});
manifest = sortrows(manifest, {'ScenarioIndex', 'GeologyCaseIndex'});
assert(height(manifest) == opt.ExpectedGeologyCount)
assert(numel(unique(manifest.GeologyId)) == opt.ExpectedGeologyCount)
assert(numel(unique(manifest.GeologyHash)) == opt.ExpectedGeologyCount)
assert(all(manifest.LinkedFaultCaseCount == numel(expectedCaseIds)))

faultCaseLinks = vertcat(allLinkTables{:});
faultCaseLinks = sortrows(faultCaseLinks, ...
    {'GeologyId', 'Level3CaseId'});
assert(height(faultCaseLinks) == expectedCaseCount)
assert(all(faultCaseLinks.GeologyIdVerified))
assert(all(faultCaseLinks.ReadableVerified))
assert(all(faultCaseLinks.CaseCompletionGateValidated))

manifestFile = fullfile(outputRoot, 'geology_stratigraphy_manifest.csv');
faultLinkFile = fullfile(outputRoot, 'geology_fault_case_links.csv');
checksumFile = fullfile(outputRoot, 'SHA256SUMS');
guardOutput(manifestFile, opt.Overwrite)
guardOutput(faultLinkFile, opt.Overwrite)
guardOutput(checksumFile, opt.Overwrite)
writetable(manifest, manifestFile)
writetable(faultCaseLinks, faultLinkFile)
packageFiles = [packageFiles; string(manifestFile); string(faultLinkFile)];
writeChecksums(packageFiles, outputRoot, checksumFile)

freezeRoot = fileparts(fileparts(dataRoot));
freezeMetadataFile = fullfile(freezeRoot, 'freeze_metadata.json');
freezeMetadataSha256 = "";
if isfile(freezeMetadataFile)
    freezeMetadataSha256 = sha256File(freezeMetadataFile);
end

completion = struct( ...
    'schema_version', 1, ...
    'status', "complete", ...
    'completed_at_utc', utcTimestamp(), ...
    'content_type', "production_geology_stratigraphy_package", ...
    'source_data_root', string(dataRoot), ...
    'source_run_root', string(runRoot), ...
    'case_work_csv', string(caseWorkFile), ...
    'case_work_csv_sha256', sha256File(caseWorkFile), ...
    'case_completion_gate', string(completionGateFile), ...
    'case_completion_gate_sha256', sha256File(completionGateFile), ...
    'freeze_metadata_sha256', freezeMetadataSha256, ...
    'expected_geology_count', opt.ExpectedGeologyCount, ...
    'generated_geology_count', height(manifest), ...
    'expected_fault_case_count', expectedCaseCount, ...
    'linked_fault_case_count', height(faultCaseLinks), ...
    'cases_per_geology', numel(expectedCaseIds), ...
    'pc_representation', "full_slice", ...
    'geology_manifest_sha256', sha256File(manifestFile), ...
    'fault_case_links_sha256', sha256File(faultLinkFile), ...
    'checksums_sha256', sha256File(checksumFile), ...
    'generator', string(mfilename('fullpath')), ...
    'generator_sha256', sha256File([mfilename('fullpath') '.m']));
writeJson(completionFile, completion)

outputs = struct( ...
    'packageRoot', string(outputRoot), ...
    'manifest', manifest, ...
    'faultCaseLinks', faultCaseLinks, ...
    'manifestFile', string(manifestFile), ...
    'faultCaseLinkFile', string(faultLinkFile), ...
    'checksumFile', string(checksumFile), ...
    'completionFile', string(completionFile));

fprintf('Completed geology-stratigraphy package: %s\n', outputRoot)
fprintf('Generated %d geology MAT files and linked %d fault cases.\n', ...
    height(manifest), height(faultCaseLinks))
end


function writeChecksums(files, root, outputFile)
% Write GNU-compatible checksums using paths relative to the package root.

files = unique(string(files(:)), 'stable');
fileId = fopen(outputFile, 'w');
assert(fileId >= 0, 'Could not open checksum file: %s', outputFile)
cleanup = onCleanup(@() fclose(fileId)); %#ok<NASGU>
for i = 1:numel(files)
    file = char(files(i));
    assert(isfile(file), 'Cannot checksum missing package file: %s', file)
    relativePath = relativeToRoot(file, root);
    fprintf(fileId, '%s  %s\n', sha256File(file), relativePath);
end
end


function relativePath = relativeToRoot(file, root)
% Convert a package path to a portable forward-slash relative path.

file = char(string(file));
root = char(string(root));
prefix = [root filesep];
assert(startsWith(file, prefix), ...
    'Package file is outside the package root: %s', file)
relativePath = strrep(file(numel(prefix) + 1:end), filesep, '/');
end


function value = sha256File(file)
% Stream a file through Java's SHA-256 implementation.

file = char(string(file));
fileId = fopen(file, 'rb');
assert(fileId >= 0, 'Could not open file for hashing: %s', file)
cleanup = onCleanup(@() fclose(fileId)); %#ok<NASGU>
digest = java.security.MessageDigest.getInstance('SHA-256');
while true
    bytes = fread(fileId, 8 * 1024 * 1024, '*uint8');
    if isempty(bytes)
        break
    end
    digest.update(bytes);
end
hashBytes = typecast(digest.digest(), 'uint8');
value = lower(string(reshape(dec2hex(hashBytes, 2).', 1, [])));
end


function writeJson(file, value)
% Write a human-readable JSON completion marker.

text = jsonencode(value, 'PrettyPrint', true);
fileId = fopen(file, 'w');
assert(fileId >= 0, 'Could not write JSON file: %s', file)
cleanup = onCleanup(@() fclose(fileId)); %#ok<NASGU>
fprintf(fileId, '%s\n', text);
end


function value = utcTimestamp()
% Return a compact ISO-8601 UTC timestamp.

value = string(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
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
