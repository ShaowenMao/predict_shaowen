function outputs = finalize_downstream_input_package( ...
        geologyStratigraphyFile, faultPropertyFiles, outputDir, varargin)
%FINALIZE_DOWNSTREAM_INPUT_PACKAGE Link and hash downstream MAT inputs.
%
%   OUTPUTS = FINALIZE_DOWNSTREAM_INPUT_PACKAGE(STRATIGRAPHYFILE,
%   FAULTPROPERTYFILES, OUTPUTDIR) validates that every fault-property MAT
%   belongs to the same geology as STRATIGRAPHYFILE, embeds the deterministic
%   stratigraphy hash into each fault-property MAT, and writes:
%
%     downstream_input_manifest.csv
%     SHA256SUMS
%
%   The input fault-property files are updated atomically in place. Use this
%   function on copied handoff artifacts, not immutable production outputs.
%
%   Name-value options:
%     'ExpectedCaseIds' - Required Level-3 case IDs. Default: [].
%     'Overwrite'       - Replace existing manifests. Default: false.

parser = inputParser;
parser.addRequired('geologyStratigraphyFile', ...
    @(x) ischar(x) || isstring(x));
parser.addRequired('faultPropertyFiles', ...
    @(x) ischar(x) || isstring(x) || iscell(x));
parser.addRequired('outputDir', @(x) ischar(x) || isstring(x));
parser.addParameter('ExpectedCaseIds', [], ...
    @(x) isnumeric(x) && isvector(x));
parser.addParameter('Overwrite', false, ...
    @(x) islogical(x) && isscalar(x));
parser.parse(geologyStratigraphyFile, faultPropertyFiles, outputDir, ...
    varargin{:});
opt = parser.Results;

geologyStratigraphyFile = char(string(opt.geologyStratigraphyFile));
faultPropertyFiles = unique(string(opt.faultPropertyFiles(:)), 'stable');
faultPropertyFiles(strlength(faultPropertyFiles) == 0) = [];
outputDir = char(string(opt.outputDir));

assert(isfile(geologyStratigraphyFile), ...
    'Missing geology-stratigraphy MAT: %s', geologyStratigraphyFile)
assert(~isempty(faultPropertyFiles), ...
    'At least one fault-property MAT is required.')
assert(all(isfile(faultPropertyFiles)), ...
    'At least one fault-property MAT does not exist.')
ensureFolder(outputDir)

manifestFile = fullfile(outputDir, 'downstream_input_manifest.csv');
checksumFile = fullfile(outputDir, 'SHA256SUMS');
guardOutput(manifestFile, opt.Overwrite)
guardOutput(checksumFile, opt.Overwrite)

loadedGeology = load(geologyStratigraphyFile, 'geologyStratigraphy');
assert(isfield(loadedGeology, 'geologyStratigraphy'), ...
    'Missing geologyStratigraphy in %s.', geologyStratigraphyFile)
geology = loadedGeology.geologyStratigraphy;
requiredGeologyFields = ["geologyId", "geologyHash", ...
    "geologyHashAlgorithm"];
missing = requiredGeologyFields(~isfield(geology, requiredGeologyFields));
assert(isempty(missing), ...
    'Geology stratigraphy is missing fields: %s', strjoin(missing, ', '))

geologyId = string(geology.geologyId);
geologyHash = string(geology.geologyHash);
geologyHashAlgorithm = string(geology.geologyHashAlgorithm);
assert(strlength(geologyHash) == 64, ...
    'Expected a 64-character SHA-256 geology hash.')

nFaultFiles = numel(faultPropertyFiles);
caseIds = zeros(nFaultFiles, 1);
caseNames = strings(nFaultFiles, 1);
samplingCaseIds = strings(nFaultFiles, 1);
phases = strings(nFaultFiles, 1);
caseTypes = strings(nFaultFiles, 1);
replicateIds = nan(nFaultFiles, 1);
useForProbabilisticUq = false(nFaultFiles, 1);
isBenchmark = false(nFaultFiles, 1);
isStressTest = false(nFaultFiles, 1);
representations = strings(nFaultFiles, 1);
pairingVerified = false(nFaultFiles, 1);

[~, stratigraphyName, stratigraphyExtension] = ...
    fileparts(geologyStratigraphyFile);
stratigraphyFileName = string([stratigraphyName stratigraphyExtension]);

for i = 1:nFaultFiles
    file = char(faultPropertyFiles(i));
    loadedFault = load(file, 'reservoirReady');
    assert(isfield(loadedFault, 'reservoirReady'), ...
        'Missing reservoirReady in %s.', file)
    reservoirReady = loadedFault.reservoirReady;
    assert(isfield(reservoirReady, 'geologyId') && ...
        string(reservoirReady.geologyId) == geologyId, ...
        'GeologyId mismatch in %s.', file)
    assert(isfield(reservoirReady, 'level3CaseId'), ...
        'Missing level3CaseId in %s.', file)
    assert(isfield(reservoirReady, 'pcRepresentation'), ...
        'Missing pcRepresentation in %s.', file)
    validateRevisedSamplingContract(reservoirReady, file)

    if isfield(reservoirReady, 'geologyHash')
        assert(string(reservoirReady.geologyHash) == geologyHash, ...
            'Embedded geology hash mismatch in %s.', file)
    end

    caseIds(i) = double(reservoirReady.level3CaseId);
    if isfield(reservoirReady, 'level3CaseName')
        caseNames(i) = string(reservoirReady.level3CaseName);
    end
    if isfield(reservoirReady, 'samplingCaseId')
        samplingCaseIds(i) = string(reservoirReady.samplingCaseId);
    end
    if isfield(reservoirReady, 'phase')
        phases(i) = string(reservoirReady.phase);
    end
    if isfield(reservoirReady, 'caseType')
        caseTypes(i) = string(reservoirReady.caseType);
    end
    if isfield(reservoirReady, 'replicateId')
        replicateIds(i) = double(reservoirReady.replicateId);
    end
    if isfield(reservoirReady, 'useForProbabilisticUq')
        useForProbabilisticUq(i) = logical( ...
            reservoirReady.useForProbabilisticUq);
    end
    if isfield(reservoirReady, 'isBenchmark')
        isBenchmark(i) = logical(reservoirReady.isBenchmark);
    end
    if isfield(reservoirReady, 'isStressTest')
        isStressTest(i) = logical(reservoirReady.isStressTest);
    end
    representations(i) = string(reservoirReady.pcRepresentation);

    alreadyLinked = hasMatchingGeologyLink( ...
        reservoirReady, geologyId, geologyHash, stratigraphyFileName);
    if ~alreadyLinked
        if ~isfield(reservoirReady, 'schemaVersion')
            reservoirReady.schemaVersion = "1.6";
        end
        reservoirReady.geologyHash = geologyHash;
        reservoirReady.geologyHashAlgorithm = geologyHashAlgorithm;
        reservoirReady.geologyLink = struct( ...
            'schemaVersion', "1.0", ...
            'geologyId', geologyId, ...
            'geologyHash', geologyHash, ...
            'stratigraphyFile', stratigraphyFileName, ...
            'pairingKey', geologyId + ":" + geologyHash, ...
            'linkedAtUtc', string(datetime('now', 'TimeZone', 'UTC', ...
                'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX')));
        saveFaultPropertiesAtomically(file, reservoirReady)
    end

    verify = load(file, 'reservoirReady');
    pairingVerified(i) = hasMatchingGeologyLink( ...
        verify.reservoirReady, geologyId, geologyHash, ...
        stratigraphyFileName);
    assert(pairingVerified(i), ...
        'Fault/stratigraphy pairing verification failed for %s.', file)
end

if ~isempty(opt.ExpectedCaseIds)
    expectedCaseIds = unique(double(opt.ExpectedCaseIds(:)));
    assert(isequal(unique(caseIds), expectedCaseIds), ...
        'Fault-property MATs do not contain exactly the expected case IDs.')
end

allFiles = [string(geologyStratigraphyFile); faultPropertyFiles];
nFiles = numel(allFiles);
fileTypes = ["geology_stratigraphy"; ...
    repmat("fault_properties", nFaultFiles, 1)];
manifestCaseIds = [NaN; caseIds];
manifestCaseNames = [""; caseNames];
manifestSamplingCaseIds = [""; samplingCaseIds];
manifestPhases = [""; phases];
manifestCaseTypes = [""; caseTypes];
manifestReplicateIds = [NaN; replicateIds];
manifestUseForProbabilisticUq = [false; useForProbabilisticUq];
manifestIsBenchmark = [false; isBenchmark];
manifestIsStressTest = [false; isStressTest];
manifestRepresentations = [""; representations];
manifestPairing = [true; pairingVerified];
relativePaths = strings(nFiles, 1);
fileNames = strings(nFiles, 1);
sizes = zeros(nFiles, 1);
sha256 = strings(nFiles, 1);

for i = 1:nFiles
    file = char(allFiles(i));
    [~, name, extension] = fileparts(file);
    fileNames(i) = string([name extension]);
    relativePaths(i) = packageRelativePath(file, outputDir);
    info = dir(file);
    sizes(i) = double(info.bytes);
    sha256(i) = sha256File(file);
end

manifest = table(fileTypes, repmat(geologyId, nFiles, 1), ...
    repmat(geologyHash, nFiles, 1), manifestCaseIds, ...
    manifestCaseNames, manifestSamplingCaseIds, manifestPhases, ...
    manifestCaseTypes, manifestReplicateIds, ...
    manifestUseForProbabilisticUq, manifestIsBenchmark, ...
    manifestIsStressTest, manifestRepresentations, fileNames, ...
    relativePaths, sizes, sha256, manifestPairing, ...
    'VariableNames', {'FileType', 'GeologyId', 'GeologyHash', ...
    'Level3CaseId', 'Level3CaseName', 'SamplingCaseId', 'Phase', ...
    'CaseType', 'ReplicateId', 'UseForProbabilisticUq', ...
    'IsBenchmark', 'IsStressTest', 'PcRepresentation', ...
    'FileName', 'RelativePath', 'SizeBytes', 'FileSHA256', ...
    'PairingVerified'});

writetable(manifest, manifestFile)
writeSha256Sums(checksumFile, sha256, relativePaths)

outputs = struct( ...
    'manifest', manifest, ...
    'manifestCsv', string(manifestFile), ...
    'sha256Sums', string(checksumFile), ...
    'geologyId', geologyId, ...
    'geologyHash', geologyHash, ...
    'faultPropertyFiles', faultPropertyFiles);

fprintf('Linked %d fault-property MAT files to geology %s.\n', ...
    nFaultFiles, geologyId)
fprintf('Saved downstream manifest: %s\n', manifestFile)
fprintf('Saved SHA-256 checksums: %s\n', checksumFile)
end


function validateRevisedSamplingContract(reservoirReady, file)
% Fail closed when a revised-design file lacks downstream-critical metadata.

if ~isfield(reservoirReady, 'samplingCase') || ...
        ~isfield(reservoirReady.samplingCase, 'metadataExplicit') || ...
        ~logical(reservoirReady.samplingCase.metadataExplicit)
    return
end

required = ["samplingCaseId", "phase", "caseType", "replicateId", ...
    "useForProbabilisticUq", "isBenchmark", "isStressTest", ...
    "windowSliceMap", "predictRealizationIdMap", ...
    "Kxx_mD", "Kyy_mD", "Kzz_mD", ...
    "Kxx_SI", "Kyy_SI", "Kzz_SI", "effectivePorosity", ...
    "pcCurves", "SwiEff", "krCurves", "saturationRegions", ...
    "samplingManifestHash", "replayManifestHash", "configurationHash"];
missing = required(~isfield(reservoirReady, required));
assert(isempty(missing), ...
    'Revised fault-property MAT %s is missing fields: %s', ...
    file, strjoin(missing, ', '))

expectedSize = [6, 87];
maps = {reservoirReady.predictRealizationIdMap, ...
    reservoirReady.Kxx_mD, reservoirReady.Kyy_mD, ...
    reservoirReady.Kzz_mD, reservoirReady.Kxx_SI, ...
    reservoirReady.Kyy_SI, reservoirReady.Kzz_SI, ...
    reservoirReady.effectivePorosity, reservoirReady.SwiEff, ...
    reservoirReady.pcCurves, reservoirReady.krCurves};
assert(all(cellfun(@(x) isequal(size(x), expectedSize), maps)), ...
    'Revised fault-property MAT %s does not have complete 6-by-87 coverage.', ...
    file)
assert(all(reservoirReady.predictRealizationIdMap > 0, 'all'), ...
    'Revised fault-property MAT %s has invalid realization IDs.', file)
assert(all(isfinite(reservoirReady.Kxx_mD), 'all') && ...
    all(isfinite(reservoirReady.Kyy_mD), 'all') && ...
    all(isfinite(reservoirReady.Kzz_mD), 'all') && ...
    all(reservoirReady.Kxx_mD > 0, 'all') && ...
    all(reservoirReady.Kyy_mD > 0, 'all') && ...
    all(reservoirReady.Kzz_mD > 0, 'all'), ...
    'Revised fault-property MAT %s has invalid permeability.', file)
assert(all(reservoirReady.effectivePorosity >= 0 & ...
    reservoirReady.effectivePorosity <= 1, 'all'), ...
    'Revised fault-property MAT %s has invalid porosity.', file)
assert(all(reservoirReady.SwiEff >= 0 & reservoirReady.SwiEff <= 1, 'all'), ...
    'Revised fault-property MAT %s has invalid Swi endpoints.', file)

hashes = [string(reservoirReady.samplingManifestHash), ...
    string(reservoirReady.replayManifestHash), ...
    string(reservoirReady.configurationHash)];
assert(all(cellfun(@isSha256, cellstr(hashes))), ...
    'Revised fault-property MAT %s has incomplete SHA-256 provenance.', file)

assert(isfield(reservoirReady, 'effectivePermeability') && ...
    isfield(reservoirReady.effectivePermeability, 'coordinateContract'), ...
    'Revised fault-property MAT %s lacks the permeability coordinate contract.', ...
    file)
contract = reservoirReady.effectivePermeability.coordinateContract;
assert(isfield(contract, 'downstream_transform_contract') && ...
    string(contract.downstream_transform_contract) == ...
        "fault_local_to_reservoir_grid_signed_yz_v1" && ...
    isfield(contract, 'rotation_required') && ...
    logical(contract.rotation_required) && ...
    isfield(contract, 'pre_rotated') && ~logical(contract.pre_rotated), ...
    'Revised fault-property MAT %s has an invalid coordinate contract.', file)
end


function tf = isSha256(value)
% Return true for one 64-character hexadecimal digest.

tf = ~isempty(regexp(char(string(value)), ...
    '^[0-9a-fA-F]{64}$', 'once'));
end


function tf = hasMatchingGeologyLink( ...
        reservoirReady, geologyId, geologyHash, stratigraphyFileName)
% Confirm the redundant top-level and structured pairing metadata.

tf = isfield(reservoirReady, 'geologyHash') && ...
    string(reservoirReady.geologyHash) == geologyHash && ...
    isfield(reservoirReady, 'geologyLink');
if ~tf
    return
end

link = reservoirReady.geologyLink;
required = ["geologyId", "geologyHash", "stratigraphyFile"];
tf = all(isfield(link, required)) && ...
    string(link.geologyId) == geologyId && ...
    string(link.geologyHash) == geologyHash && ...
    string(link.stratigraphyFile) == stratigraphyFileName;
end


function saveFaultPropertiesAtomically(file, reservoirReady)
% Replace a copied handoff MAT only after a complete temporary save.

folder = fileparts(file);
temporaryFile = [tempname(folder) '.mat'];
cleanup = onCleanup(@() deleteIfPresent(temporaryFile));
save(temporaryFile, 'reservoirReady', '-v7.3')

verify = load(temporaryFile, 'reservoirReady');
assert(isfield(verify, 'reservoirReady'), ...
    'Temporary fault-property MAT could not be verified: %s', ...
    temporaryFile)
[ok, message] = movefile(temporaryFile, file, 'f');
assert(ok, 'Could not publish linked fault-property MAT: %s', message)
clear cleanup
end


function relativePath = packageRelativePath(file, outputDir)
% Return a portable path relative to the package root when possible.

filePath = canonicalPath(file);
rootPath = canonicalPath(outputDir);
rootPrefix = rootPath + "/";
if startsWith(filePath, rootPrefix, 'IgnoreCase', ispc)
    relativePath = extractAfter(filePath, strlength(rootPrefix));
else
    relativePath = filePath;
end
end


function path = canonicalPath(value)
% Normalize a path and use forward separators for portable manifests.

path = string(java.io.File(char(value)).getCanonicalPath());
path = replace(path, "\", "/");
end


function value = sha256File(file)
% Compute SHA-256 incrementally without loading a complete MAT into memory.

fid = fopen(file, 'rb');
assert(fid >= 0, 'Could not open file for hashing: %s', file)
cleanup = onCleanup(@() fclose(fid));
digest = java.security.MessageDigest.getInstance('SHA-256');
while true
    bytes = fread(fid, 1024 * 1024, '*uint8');
    if isempty(bytes)
        break
    end
    digest.update(bytes);
end
hashBytes = typecast(digest.digest(), 'uint8');
value = lower(string(reshape(dec2hex(hashBytes, 2).', 1, [])));
clear cleanup
end


function writeSha256Sums(file, sha256, relativePaths)
% Write a standard checksum file accepted by sha256sum --check.

fid = fopen(file, 'wt');
assert(fid >= 0, 'Could not create checksum file: %s', file)
cleanup = onCleanup(@() fclose(fid));
for i = 1:numel(sha256)
    fprintf(fid, '%s  %s\n', sha256(i), ...
        replace(relativePaths(i), "\", "/"));
end
clear cleanup
end


function guardOutput(file, overwrite)
% Refuse accidental manifest replacement unless explicitly requested.

if isfile(file) && ~overwrite
    error('DownstreamPackage:OutputExists', ...
        'Output already exists: %s. Set Overwrite=true to replace it.', ...
        file)
end
end


function ensureFolder(folder)
% Create the manifest folder when needed.

if ~isfolder(folder)
    mkdir(folder)
end
end


function deleteIfPresent(file)
% Remove an unpublished temporary MAT after an interrupted save.

if isfile(file)
    delete(file)
end
end
