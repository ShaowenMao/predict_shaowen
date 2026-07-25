function fileName = fault_properties_filename(geologyId, caseId, pcRepresentation)
%FAULT_PROPERTIES_FILENAME Build the canonical downstream fault-property name.
%
%   FILENAME = FAULT_PROPERTIES_FILENAME(GEOLOGYID, CASEID,
%   PCREPRESENTATION) returns:
%
%     fault_properties_<geology>_caseNN_pc_full_slice.mat
%
%   or:
%
%     fault_properties_<geology>_caseNN_pc_branch_medoid.mat
%
%   PCREPRESENTATION uses the internal representation names "full_slice"
%   and "pe_branch_medoid".

geologyId = string(geologyId);
pcRepresentation = lower(string(pcRepresentation));

assert(isscalar(geologyId) && strlength(geologyId) > 0, ...
    'FaultPropertiesFilename:GeologyId', ...
    'geologyId must be a nonempty scalar string.');
assert(isscalar(caseId) && isnumeric(caseId) && isfinite(caseId) && ...
    caseId >= 0 && caseId == floor(caseId), ...
    'FaultPropertiesFilename:CaseId', ...
    'caseId must be a nonnegative integer scalar.');

switch pcRepresentation
    case "full_slice"
        representationToken = 'pc_full_slice';
    case "pe_branch_medoid"
        representationToken = 'pc_branch_medoid';
    otherwise
        error('FaultPropertiesFilename:PcRepresentation', ...
            'Unsupported Pc representation: %s.', pcRepresentation);
end

fileName = sprintf('fault_properties_%s_case%02d_%s.mat', ...
    char(geologyId), double(caseId), representationToken);
end
