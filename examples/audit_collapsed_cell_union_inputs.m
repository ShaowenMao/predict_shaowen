function auditTable = audit_collapsed_cell_union_inputs(outputFile)
%AUDIT_COLLAPSED_CELL_UNION_INPUTS Check collapsed layer inputs without PREDICT.
%
% This lightweight audit mirrors the layer-collapse input construction used
% by gom_perm_varying_thickness_geology_cases_collapsed_cell_union, but it
% does not run MRST or PREDICT. It verifies every default thickness
% scenario/window pair for:
%   - original pattern length matching the base layer vector,
%   - collapsed pattern,
%   - collapsed thickness vector,
%   - collapsed zmax vector,
%   - conservation of total thickness on each side.

if nargin < 1 || isempty(outputFile)
    outputFile = fullfile('D:', 'codex_gom', ...
        'collapsed_cell_union_input_audit', ...
        'collapsed_cell_union_input_audit.csv');
end

scenarioTable = buildDefaultThicknessScenarioTable();
rows = {};
windows = unique(scenarioTable.Window, 'stable');

for iw = 1:numel(windows)
    window = char(windows(iw));
    baseOpt = getWindowOptions(window);
    scenarioRows = scenarioTable(strcmpi(scenarioTable.Window, window), :);

    for ir = 1:height(scenarioRows)
        row = scenarioRows(ir, :);
        [fwThick, fwZmax, fwPattern] = collapseSide( ...
            baseOpt.thick{1}, baseOpt.zmax{1}, char(row.FWPattern));
        [hwThick, hwZmax, hwPattern] = collapseSide( ...
            baseOpt.thick{2}, baseOpt.zmax{2}, char(row.HWPattern));

        fwConserved = abs(sum(fwThick) - sum(baseOpt.thick{1})) < 1e-9;
        hwConserved = abs(sum(hwThick) - sum(baseOpt.thick{2})) < 1e-9;

        rows(end+1, :) = {row.ScenarioIndex, char(row.ScenarioLabel), ...
            char(row.ScenarioName), window, char(row.FWPattern), ...
            fwPattern, vectorString(baseOpt.thick{1}), vectorString(fwThick), ...
            vectorString(baseOpt.zmax{1}), vectorString(fwZmax), ...
            fwConserved, char(row.HWPattern), hwPattern, ...
            vectorString(baseOpt.thick{2}), vectorString(hwThick), ...
            vectorString(baseOpt.zmax{2}), vectorString(hwZmax), ...
            hwConserved}; %#ok<AGROW>
    end
end

auditTable = cell2table(rows, 'VariableNames', { ...
    'ScenarioIndex', 'ScenarioLabel', 'ScenarioName', 'Window', ...
    'OriginalFWPattern', 'CollapsedFWPattern', ...
    'OriginalFWThickness', 'CollapsedFWThickness', ...
    'OriginalFWZmax', 'CollapsedFWZmax', 'FWThicknessConserved', ...
    'OriginalHWPattern', 'CollapsedHWPattern', ...
    'OriginalHWThickness', 'CollapsedHWThickness', ...
    'OriginalHWZmax', 'CollapsedHWZmax', 'HWThicknessConserved'});

assert(all(auditTable.FWThicknessConserved), ...
    'Footwall thickness was not conserved for at least one row.')
assert(all(auditTable.HWThicknessConserved), ...
    'Hangingwall thickness was not conserved for at least one row.')

folderPath = fileparts(outputFile);
if ~exist(folderPath, 'dir')
    mkdir(folderPath);
end
writetable(auditTable, outputFile);
fprintf('Saved collapsed-input audit table to:\n  %s\n', outputFile);
end


function [thickOut, zmaxOut, patternOut] = collapseSide(thick, zmax, pattern)
thick = thick(:).';
zmax = zmax(:).';
pattern = upper(char(pattern));
assert(numel(pattern) == numel(thick), ...
    'Pattern length does not match layer thickness vector.')
starts = [1, find(diff(double(pattern)) ~= 0) + 1, numel(pattern) + 1];
thickOut = zeros(1, numel(starts)-1);
zmaxOut = zeros(1, numel(starts)-1);
patternOut = repmat('S', 1, numel(starts)-1);
for g = 1:(numel(starts)-1)
    ids = starts(g):(starts(g+1)-1);
    thickOut(g) = sum(thick(ids));
    zmaxOut(g) = sum(zmax(ids).*thick(ids)) ./ sum(thick(ids));
    patternOut(g) = pattern(ids(1));
end
end


function s = vectorString(x)
s = strjoin(compose('%.8g', x(:).'), ' ');
end


function scenarioTable = buildDefaultThicknessScenarioTable()
rows = {
    1, 'scenario_01_low_sand_uniform',       'low sand, uniform',       'famp1', 'SC',   'SCCC';
    1, 'scenario_01_low_sand_uniform',       'low sand, uniform',       'famp2', 'SCCC', 'SCCC';
    1, 'scenario_01_low_sand_uniform',       'low sand, uniform',       'famp3', 'SCCC', 'SCCC';
    1, 'scenario_01_low_sand_uniform',       'low sand, uniform',       'famp4', 'SCCC', 'SCCC';
    1, 'scenario_01_low_sand_uniform',       'low sand, uniform',       'famp5', 'SCCC', 'SCCC';
    1, 'scenario_01_low_sand_uniform',       'low sand, uniform',       'famp6', 'SCCC', 'S';
    2, 'scenario_02_medium_sand_uniform',    'medium sand, uniform',    'famp1', 'SC',   'SCSC';
    2, 'scenario_02_medium_sand_uniform',    'medium sand, uniform',    'famp2', 'SCSC', 'SCSC';
    2, 'scenario_02_medium_sand_uniform',    'medium sand, uniform',    'famp3', 'SCSC', 'SCSC';
    2, 'scenario_02_medium_sand_uniform',    'medium sand, uniform',    'famp4', 'SCSC', 'SCSC';
    2, 'scenario_02_medium_sand_uniform',    'medium sand, uniform',    'famp5', 'SCSC', 'SCSC';
    2, 'scenario_02_medium_sand_uniform',    'medium sand, uniform',    'famp6', 'SCSC', 'S';
    3, 'scenario_03_high_sand_uniform',      'high sand, uniform',      'famp1', 'SC',   'SSSC';
    3, 'scenario_03_high_sand_uniform',      'high sand, uniform',      'famp2', 'SSSC', 'SSSC';
    3, 'scenario_03_high_sand_uniform',      'high sand, uniform',      'famp3', 'SSSC', 'SSSC';
    3, 'scenario_03_high_sand_uniform',      'high sand, uniform',      'famp4', 'SSSC', 'SSSC';
    3, 'scenario_03_high_sand_uniform',      'high sand, uniform',      'famp5', 'SSSC', 'SSSC';
    3, 'scenario_03_high_sand_uniform',      'high sand, uniform',      'famp6', 'SSSC', 'S';
    4, 'scenario_04_low_sand_nonuniform',    'low sand, nonuniform',    'famp1', 'SC',   'SCCS';
    4, 'scenario_04_low_sand_nonuniform',    'low sand, nonuniform',    'famp2', 'SCCS', 'CCCC';
    4, 'scenario_04_low_sand_nonuniform',    'low sand, nonuniform',    'famp3', 'CCCC', 'SCCS';
    4, 'scenario_04_low_sand_nonuniform',    'low sand, nonuniform',    'famp4', 'SCCS', 'CCCC';
    4, 'scenario_04_low_sand_nonuniform',    'low sand, nonuniform',    'famp5', 'CCCC', 'SCCC';
    4, 'scenario_04_low_sand_nonuniform',    'low sand, nonuniform',    'famp6', 'SCCC', 'S';
    5, 'scenario_05_medium_sand_nonuniform', 'medium sand, nonuniform', 'famp1', 'SC',   'SSSC';
    5, 'scenario_05_medium_sand_nonuniform', 'medium sand, nonuniform', 'famp2', 'SSSC', 'SCCC';
    5, 'scenario_05_medium_sand_nonuniform', 'medium sand, nonuniform', 'famp3', 'SCCC', 'SSCC';
    5, 'scenario_05_medium_sand_nonuniform', 'medium sand, nonuniform', 'famp4', 'SSCC', 'SCCC';
    5, 'scenario_05_medium_sand_nonuniform', 'medium sand, nonuniform', 'famp5', 'SCCC', 'SSSC';
    5, 'scenario_05_medium_sand_nonuniform', 'medium sand, nonuniform', 'famp6', 'SSSC', 'S';
    6, 'scenario_06_high_sand_nonuniform',   'high sand, nonuniform',   'famp1', 'SC',   'SSSC';
    6, 'scenario_06_high_sand_nonuniform',   'high sand, nonuniform',   'famp2', 'SSSC', 'SSSS';
    6, 'scenario_06_high_sand_nonuniform',   'high sand, nonuniform',   'famp3', 'SSSS', 'CSSC';
    6, 'scenario_06_high_sand_nonuniform',   'high sand, nonuniform',   'famp4', 'CSSC', 'SSSS';
    6, 'scenario_06_high_sand_nonuniform',   'high sand, nonuniform',   'famp5', 'SSSS', 'CSSC';
    6, 'scenario_06_high_sand_nonuniform',   'high sand, nonuniform',   'famp6', 'CSSC', 'S'};
scenarioTable = cell2table(rows, 'VariableNames', ...
    {'ScenarioIndex', 'ScenarioLabel', 'ScenarioName', ...
     'Window', 'FWPattern', 'HWPattern'});
end


function opt = getWindowOptions(window)
switch lower(window)
    case 'famp1'
        opt.thick = {[115.6143 28.8949], [37.6113 37.6861 37.6113 31.6005]};
        opt.zmax = {[1912 1861], [1934 1909 1884 1860]};
    case 'famp2'
        opt.thick = {[36.9255 35.8537 36.8537 36.3111], ...
                     [36.5042 36.5042 36.4314 36.5042]};
        opt.zmax = {[1837.5 1812.5 1787.5 1762.5], ...
                    [1837.5 1812.5 1787.5 1762.5]};
    case 'famp3'
        opt.thick = {[35.8537 35.8537 35.8537 35.8537], ...
                     [35.8537 35.8537 35.8537 35.8537]};
        opt.zmax = {[1738.8 1713.8 1688.8 1663.8], ...
                    [1738.8 1713.8 1688.8 1663.8]};
    case 'famp4'
        opt.thick = {[35.8537 35.8537 35.8537 35.9255], ...
                     [35.8537 35.8537 35.8537 35.9255]};
        opt.zmax = {[1638.8 1613.8 1588.8 1563.8], ...
                    [1638.8 1613.8 1588.8 1563.8]};
    case 'famp5'
        opt.thick = {[35.8537 35.8537 35.8537 35.8537], ...
                     [37.4901 35.2847 35.3553 35.2847]};
        opt.zmax = {[1538.8 1513.82 1488.75 1463.99], ...
                    [1538.8 1513.82 1488.75 1463.99]};
    case 'famp6'
        opt.thick = {[28.2932 33.1042 33.1699 33.1042], 127.6715};
        opt.zmax = {[1440.6 1417.5 1392.5 1367.5], 1400};
    otherwise
        error('Unsupported window "%s".', window)
end
end
