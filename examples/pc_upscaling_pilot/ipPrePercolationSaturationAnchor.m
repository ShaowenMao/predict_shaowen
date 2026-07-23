function anchorSg = ipPrePercolationSaturationAnchor(firstConnectedSg)
%IPPREPERCOLATIONSATURATIONANCHOR Place the pre-breakthrough Pc anchor.
%
%   ANCHORSG = IPPREPERCOLATIONSATURATIONANCHOR(FIRSTCONNECTEDSG) returns
%   the legacy 1e-5 gas-saturation anchor whenever it precedes the first
%   connected invasion state. If breakthrough occurs below 1e-5, the
%   anchor is placed halfway to that state instead. This preserves pressure
%   ordering when native Pc points are subsequently sorted by saturation.

validateattributes(firstConnectedSg, {'numeric'}, ...
    {'real', 'finite', 'scalar', 'positive'}, mfilename, ...
    'firstConnectedSg');

legacyAnchorSg = 1.0e-5;
if firstConnectedSg > legacyAnchorSg
    anchorSg = legacyAnchorSg;
else
    anchorSg = 0.5 .* double(firstConnectedSg);
end
end
