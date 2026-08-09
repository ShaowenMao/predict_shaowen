function [globalTensor, compactTensor] = ...
        fault_local_permeability_tensor_contract(localPermeability, thetaDeg)
%FAULT_LOCAL_PERMEABILITY_TENSOR_CONTRACT Reference coordinate transform.
%
%   [K, KC] = FAULT_LOCAL_PERMEABILITY_TENSOR_CONTRACT(LOCAL, THETA)
%   transforms PREDICT-local diagonal permeability components
%
%     LOCAL = [kxx, kyy, kzz]
%             [fault normal, along strike, down dip]
%
%   into reservoir-grid coordinates. Global X is aligned with strike and
%   THETA is the signed rotation in the global Y-Z plane:
%
%     theta = sign(dY/dZ) * (dip - 90 degrees).
%
%   This pure helper documents and tests the handoff contract. Production
%   reservoir import must calculate THETA from the actual paired fault-node
%   trace for each cell; the upstream fault-property MAT remains unrotated.
%
%   K is N-by-3-by-3. KC uses the compact component order
%   [Kxx, Kxy, Kxz, Kyy, Kyz, Kzz]. Units are unchanged.

validateattributes(localPermeability, {'numeric'}, ...
    {'2d', 'ncols', 3, 'finite', 'positive'});
validateattributes(thetaDeg, {'numeric'}, {'vector', 'finite'});
n = size(localPermeability, 1);
if isscalar(thetaDeg)
    thetaDeg = repmat(double(thetaDeg), n, 1);
else
    thetaDeg = double(thetaDeg(:));
    assert(numel(thetaDeg) == n, ...
        'One signed rotation angle is required per permeability row.');
end

kNormal = double(localPermeability(:, 1));
kStrike = double(localPermeability(:, 2));
kDip = double(localPermeability(:, 3));
c = cosd(thetaDeg);
s = sind(thetaDeg);

compactTensor = zeros(n, 6);
compactTensor(:, 1) = kStrike;
compactTensor(:, 4) = c.^2.*kNormal + s.^2.*kDip;
compactTensor(:, 5) = c.*s.*(kNormal - kDip);
compactTensor(:, 6) = s.^2.*kNormal + c.^2.*kDip;

globalTensor = zeros(n, 3, 3);
for i = 1:n
    globalTensor(i, :, :) = [ ...
        compactTensor(i, 1), compactTensor(i, 2), compactTensor(i, 3); ...
        compactTensor(i, 2), compactTensor(i, 4), compactTensor(i, 5); ...
        compactTensor(i, 3), compactTensor(i, 5), compactTensor(i, 6)];
end
end
