function validation = validate_pc_guided_kr_representatives( ...
        fullSummaryCsv, proxySummaryCsv, outputDir)
%VALIDATE_PC_GUIDED_KR_REPRESENTATIVES Compare reduced and full Kr results.
%
% The validation is performed in normalized gas-saturation space so that
% curve-shape error is separated from the slice-specific Pc endpoint. The
% full benchmark's medoid minimizes average two-phase curve RMSE across all
% slices in the same case/window. The proxy is the one dynamic Kr curve
% selected using median effective Swi from Pc upscaling.
%
% Environment-variable defaults are available for batch execution:
%   KR_VALIDATION_FULL_SUMMARY_CSV
%   KR_VALIDATION_PROXY_SUMMARY_CSV
%   KR_VALIDATION_OUTPUT_DIR
%   KR_VALIDATION_RMSE_TOL       (default 0.02)
%   KR_VALIDATION_MAX_ABS_TOL    (default 0.05)

if nargin < 1 || strlength(string(fullSummaryCsv)) == 0
    fullSummaryCsv = getenv('KR_VALIDATION_FULL_SUMMARY_CSV');
end
if nargin < 2 || strlength(string(proxySummaryCsv)) == 0
    proxySummaryCsv = getenv('KR_VALIDATION_PROXY_SUMMARY_CSV');
end
if nargin < 3 || strlength(string(outputDir)) == 0
    outputDir = getenv('KR_VALIDATION_OUTPUT_DIR');
end
assert(exist(fullSummaryCsv, 'file') == 2, ...
    'Full Kr summary not found: %s', fullSummaryCsv);
assert(exist(proxySummaryCsv, 'file') == 2, ...
    'Pc-guided Kr summary not found: %s', proxySummaryCsv);
if strlength(string(outputDir)) == 0
    outputDir = fullfile(fileparts(proxySummaryCsv), 'validation');
end
if exist(outputDir, 'dir') ~= 7
    mkdir(outputDir);
end

rmseTolerance = numericEnv('KR_VALIDATION_RMSE_TOL', 0.02);
maxAbsTolerance = numericEnv('KR_VALIDATION_MAX_ABS_TOL', 0.05);
u = linspace(0, 1, 201);

fullTable = readtable(fullSummaryCsv, 'TextType', 'string');
proxyTable = readtable(proxySummaryCsv, 'TextType', 'string');
required = {'GeologyId', 'Level3CaseId', 'Window', 'SliceIndex', ...
    'SourceRow', 'PcMaxSg', 'IrreducibleWaterSaturation', ...
    'BrineCoreyExponent', 'GasCoreyExponent'};
assert(all(ismember(required, fullTable.Properties.VariableNames)), ...
    'Full Kr summary lacks required validation columns.');
assert(all(ismember(required, proxyTable.Properties.VariableNames)), ...
    'Pc-guided Kr summary lacks required validation columns.');

keys = unique(fullTable(:, {'GeologyId', 'Level3CaseId', ...
    'Level3CaseName', 'Window'}), 'rows', 'stable');
rows = cell(height(keys), 25);

for g = 1:height(keys)
    fullMask = fullTable.GeologyId == keys.GeologyId(g) & ...
        fullTable.Level3CaseId == keys.Level3CaseId(g) & ...
        fullTable.Window == keys.Window(g);
    proxyMask = proxyTable.GeologyId == keys.GeologyId(g) & ...
        proxyTable.Level3CaseId == keys.Level3CaseId(g) & ...
        proxyTable.Window == keys.Window(g);
    fullRows = fullTable(fullMask, :);
    proxyRows = proxyTable(proxyMask, :);
    assert(height(proxyRows) == 1, ...
        'Expected one Pc-guided curve for case %d, window %s; found %d.', ...
        keys.Level3CaseId(g), keys.Window(g), height(proxyRows));

    fullShapes = coreyShapes(fullRows, u);
    proxyShape = coreyShapes(proxyRows, u);
    distance = pairwiseCurveRmse(fullShapes);
    centrality = mean(distance, 2);
    [minimumCentrality, medoidIndex] = min(centrality);
    proxyDistance = sqrt(mean((fullShapes - proxyShape).^2, 2));
    [proxyToNearestFull, nearestProxyIndex] = min(proxyDistance);
    proxyCentrality = mean(proxyDistance);
    centralityTolerance = 1.0e-12;
    proxyRank = 1 + sum(centrality < proxyCentrality - centralityTolerance);
    isMedoidShape = proxyCentrality <= minimumCentrality + centralityTolerance;
    rmseToMedoid = sqrt(mean( ...
        (proxyShape - fullShapes(medoidIndex, :)).^2));
    maxAbsToMedoid = max(abs( ...
        proxyShape - fullShapes(medoidIndex, :)));
    accepted = rmseToMedoid <= rmseTolerance && ...
        maxAbsToMedoid <= maxAbsTolerance;

    rows(g, :) = { ...
        keys.GeologyId(g), keys.Level3CaseId(g), ...
        keys.Level3CaseName(g), keys.Window(g), height(fullRows), ...
        double(proxyRows.SourceRow(1)), double(proxyRows.SliceIndex(1)), ...
        double(proxyRows.IrreducibleWaterSaturation(1)), ...
        double(proxyRows.BrineCoreyExponent(1)), ...
        double(proxyRows.GasCoreyExponent(1)), ...
        double(fullRows.SourceRow(medoidIndex)), ...
        double(fullRows.SliceIndex(medoidIndex)), ...
        double(fullRows.IrreducibleWaterSaturation(medoidIndex)), ...
        double(fullRows.BrineCoreyExponent(medoidIndex)), ...
        double(fullRows.GasCoreyExponent(medoidIndex)), ...
        rmseToMedoid, maxAbsToMedoid, proxyCentrality, ...
        minimumCentrality, proxyCentrality / max(minimumCentrality, eps), ...
        proxyRank, isMedoidShape, proxyToNearestFull, ...
        double(fullRows.SliceIndex(nearestProxyIndex)), accepted};
end

validation = cell2table(rows, 'VariableNames', { ...
    'GeologyId', 'Level3CaseId', 'Level3CaseName', 'Window', ...
    'FullCurveCount', 'ProxySourceRow', 'ProxySliceIndex', ...
    'ProxyEffectiveSwi', 'ProxyBrineCoreyExponent', ...
    'ProxyGasCoreyExponent', 'FullMedoidSourceRow', ...
    'FullMedoidSliceIndex', 'FullMedoidEffectiveSwi', ...
    'FullMedoidBrineCoreyExponent', 'FullMedoidGasCoreyExponent', ...
    'RmseToFullMedoid', 'MaxAbsDifferenceToFullMedoid', ...
    'ProxyMeanDistanceToFullFamily', 'MedoidMeanDistanceToFullFamily', ...
    'CentralityRatio', 'ProxyCentralityRank', 'IsExactMedoidShape', ...
    'DistanceToNearestFullShape', 'NearestFullShapeSliceIndex', 'Accepted'});

outputCsv = fullfile(outputDir, 'pc_guided_kr_validation.csv');
writetable(validation, outputCsv);
fprintf('Saved Pc-guided Kr validation: %s\n', outputCsv);
fprintf('Accepted windows: %d/%d (RMSE <= %.3g and max abs <= %.3g).\n', ...
    sum(validation.Accepted), height(validation), ...
    rmseTolerance, maxAbsTolerance);
disp(validation(:, {'Level3CaseId', 'Window', 'ProxySliceIndex', ...
    'FullMedoidSliceIndex', 'RmseToFullMedoid', ...
    'MaxAbsDifferenceToFullMedoid', 'Accepted'}));
end


function shapes = coreyShapes(T, u)
% Evaluate fitted dynamic Corey curves in normalized saturation space.

n = height(T);
shapes = nan(n, 2 * numel(u));
for i = 1:n
    krg = u .^ double(T.GasCoreyExponent(i));
    krw = (1.0 - u) .^ double(T.BrineCoreyExponent(i));
    shapes(i, :) = [krg, krw];
end
end


function distance = pairwiseCurveRmse(shapes)
% Compute the symmetric pairwise RMSE matrix without extra toolboxes.

n = size(shapes, 1);
distance = zeros(n, n);
for i = 1:n
    for j = (i + 1):n
        value = sqrt(mean((shapes(i, :) - shapes(j, :)).^2));
        distance(i, j) = value;
        distance(j, i) = value;
    end
end
end


function value = numericEnv(name, defaultValue)
% Read one finite scalar environment option.

text = strtrim(string(getenv(name)));
if text == ""
    value = defaultValue;
else
    value = str2double(text);
    assert(isfinite(value), '%s must be a finite scalar.', name);
end
end

