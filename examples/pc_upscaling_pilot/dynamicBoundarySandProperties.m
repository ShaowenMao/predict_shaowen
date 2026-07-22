function [sandPoro, sandPerm, source] = dynamicBoundarySandProperties( ...
        replay, poroAll, permAll, isSmear)
%DYNAMICBOUNDARYSANDPROPERTIES Define the dynamic model's sand boundary.
%   [SANDPORO, SANDPERM, SOURCE] = DYNAMICBOUNDARYSANDPROPERTIES(REPLAY,
%   POROALL, PERMALL, ISSMEAR) returns one porosity and one six-component
%   symmetric permeability tensor for the artificial sand layer added at
%   the flow boundary by the Appendix-C dynamic Kr workflow.
%
%   When fault-core sand cells exist, the function reproduces the original
%   behavior and averages their realized fine-scale properties. A valid
%   all-smear realization has no such cells, even though its parent
%   stratigraphy contains sand. In that edge case, the function rebuilds a
%   thickness-weighted expected sand tensor from the parent material
%   properties saved during exact PREDICT replay. The fallback changes only
%   the artificial boundary layer and never changes the replayed fault map.

sandCells = ~isSmear;
if any(sandCells)
    sandPoro = mean(poroAll(sandCells), 'omitnan');
    sandPerm = mean(permAll(sandCells, :), 1, 'omitnan');
    source = "fault_core_sand_cells";
    return
end

assert(isfield(replay, 'SectionDetails') && ...
    ~isempty(replay.SectionDetails), ...
    ['All-smear replay lacks the parent material properties needed ', ...
     'for the sand boundary.']);
assert(isfield(replay, 'Disp') && isfield(replay, 'Dip'), ...
    ['All-smear replay lacks fault geometry needed to rotate parent ', ...
     'sand tensors.']);

poroValues = zeros(0, 1);
permValues = zeros(0, 6);
weights = zeros(0, 1);
faultThrow = double(replay.Disp) .* sind(double(replay.Dip));

for sectionId = 1:numel(replay.SectionDetails)
    section = replay.SectionDetails(sectionId);
    assert(isfield(section, 'MatProps') && ~isempty(section.MatProps), ...
        'Replay section %d lacks parent material properties.', sectionId);
    props = section.MatProps;
    nUnits = numel(props.poro);
    sandUnitMask = identifyParentSandUnits(section, props, nUnits);
    assert(any(sandUnitMask), ...
        'Replay section %d contains no identifiable parent sand unit.', ...
        sectionId);

    kAcross = double(props.perm(sandUnitMask));
    anisotropy = double(props.permAnisoRatio(sandUnitMask));
    poro = double(props.poro(sandUnitMask));
    alpha = parentFaultAngle(faultThrow, double(replay.Dip), ...
        double(props.thick));
    tensor = rotateParentSandTensors(kAcross, anisotropy, alpha);
    unitWeights = parentUnitWeights(section, sandUnitMask, nUnits);

    poroValues = [poroValues; poro(:)]; %#ok<AGROW>
    permValues = [permValues; tensor]; %#ok<AGROW>
    weights = [weights; unitWeights(:)]; %#ok<AGROW>
end

assert(all(isfinite(weights) & weights > 0) && sum(weights) > 0, ...
    'Parent-sand boundary weights are nonphysical.');
weights = weights ./ sum(weights);
sandPoro = sum(weights .* poroValues);
sandPerm = sum(weights .* permValues, 1);
assert(isfinite(sandPoro) && sandPoro > 0 && sandPoro <= 1, ...
    'Parent-sand boundary porosity is nonphysical.');
assert(all(isfinite(sandPerm)) && all(sandPerm([1, 4, 6]) > 0), ...
    'Parent-sand boundary permeability is nonphysical.');
source = "parent_sand_material_fallback";
end


function sandUnitMask = identifyParentSandUnits(section, props, nUnits)
% Identify material-property entries associated with parent sand layers.

sandUnitMask = false(1, nUnits);
if isfield(section, 'MatMap') && isstruct(section.MatMap) && ...
        isfield(section.MatMap, 'unitIn') && ...
        isfield(section.MatMap, 'isclayIn')
    unitIds = double(section.MatMap.unitIn(~section.MatMap.isclayIn));
    unitIds = unitIds(isfinite(unitIds) & unitIds >= 1 & unitIds <= nUnits);
    sandUnitMask(unique(round(unitIds))) = true;
elseif isfield(props, 'ssfc')
    % PREDICT stores SSFc only for clay-source units; sand entries are NaN.
    sandUnitMask = isnan(props.ssfc);
end
end


function alpha = parentFaultAngle(faultThrow, dip, thickness)
% Reproduce getFaultAngles without requiring PREDICT helpers on workers.

gamma = 90 - dip;
b = thickness ./ sind(dip) + faultThrow .* cotd(dip);
delta = atand(faultThrow ./ b);
alpha = 90 - gamma - delta;
end


function tensor = rotateParentSandTensors(kAcross, anisotropy, alpha)
% Rotate local across/along permeabilities into [xx xy xz yy yz zz].

c = cosd(alpha);
s = sind(alpha);
kAlong = kAcross .* anisotropy;
kxx = c.^2 .* kAcross + s.^2 .* kAlong;
kxz = c .* s .* (kAcross - kAlong);
kzz = s.^2 .* kAcross + c.^2 .* kAlong;
tensor = [kxx(:), zeros(numel(kxx), 1), kxz(:), ...
    kAlong(:), zeros(numel(kxx), 1), kzz(:)];
end


function weights = parentUnitWeights(section, sandUnitMask, nUnits)
% Use parent-layer thickness where replay metadata provides it.

weights = ones(sum(sandUnitMask), 1);
if isfield(section, 'MatMap') && isstruct(section.MatMap) && ...
        isfield(section.MatMap, 'layerTop') && ...
        isfield(section.MatMap, 'layerBot') && ...
        numel(section.MatMap.layerTop) == nUnits
    allWeights = abs(double(section.MatMap.layerTop(:)) - ...
        double(section.MatMap.layerBot(:)));
    candidateWeights = allWeights(sandUnitMask(:));
    if all(isfinite(candidateWeights) & candidateWeights > 0)
        weights = candidateWeights;
    end
end
end
