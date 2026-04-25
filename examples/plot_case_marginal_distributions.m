function outputFiles = plot_case_marginal_distributions(dataDir, caseLabel, outputDir)
% Plot marginal permeability distributions for one case across all windows.
%
% Usage:
%   plot_case_marginal_distributions()
%   plot_case_marginal_distributions('D:\Github\predict_shaowen\examples\data', ...
%       'case_009_zf0050_svcl030_cvcl060')

if nargin < 1 || isempty(dataDir)
    dataDir = fullfile(fileparts(mfilename('fullpath')), 'data');
end
if nargin < 2 || isempty(caseLabel)
    caseLabel = 'case_009_zf0050_svcl030_cvcl060';
end
if nargin < 3 || isempty(outputDir)
    outputDir = fullfile(fileparts(mfilename('fullpath')), 'visualizations');
end

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

windows = {'famp1', 'famp2', 'famp3', 'famp4', 'famp5', 'famp6'};
componentNames = {'K_x', 'K_y', 'K_z'};
colors = lines(numel(windows));

permsByWindow = cell(numel(windows), 1);
for iw = 1:numel(windows)
    matFile = fullfile(dataDir, windows{iw}, caseLabel, 'predict_runs.mat');
    if ~exist(matFile, 'file')
        error('Missing result file: %s', matFile)
    end

    S = load(matFile, 'perms');
    if ~isfield(S, 'perms') || size(S.perms, 2) ~= 3
        error('File %s does not contain a valid perms array.', matFile)
    end

    perms = S.perms;
    good = all(isfinite(perms), 2) & all(perms > 0, 2);
    permsByWindow{iw} = perms(good, :);
end

logPermsByWindow = cellfun(@log10, permsByWindow, 'UniformOutput', false);

fig = figure('Color', 'w', 'Position', [100 50 1150 1350]);
tiledlayout(fig, numel(windows), 3, 'TileSpacing', 'compact', ...
            'Padding', 'compact');

componentEdges = cell(1, 3);
for icomp = 1:3
    allVals = [];
    for iw = 1:numel(windows)
        allVals = [allVals; logPermsByWindow{iw}(:, icomp)]; %#ok<AGROW>
    end

    xMin = floor(min(allVals) * 2) / 2;
    xMax = ceil(max(allVals) * 2) / 2;
    componentEdges{icomp} = linspace(xMin, xMax, 35);
end

for iw = 1:numel(windows)
    for icomp = 1:3
        ax = nexttile;
        vals = logPermsByWindow{iw}(:, icomp);
        histogram(ax, vals, componentEdges{icomp}, ...
                  'Normalization', 'probability', ...
                  'FaceColor', colors(iw, :), ...
                  'EdgeColor', 'none', ...
                  'FaceAlpha', 0.85);

        grid(ax, 'on')
        box(ax, 'on')
        xlim(ax, componentEdges{icomp}([1 end]))

        if iw == 1
            title(ax, componentNames{icomp}, 'Interpreter', 'none')
        end
        if icomp == 1
            ylabel(ax, {windows{iw}; 'Probability'})
        end
        if iw == numel(windows)
            xlabel(ax, sprintf('log_{10}(%s [mD])', componentNames{icomp}))
        end
    end
end

sgtitle(fig, [strrep(caseLabel, '_', '\_') ' marginal histograms'])

pngFile = fullfile(outputDir, [caseLabel '_marginal_histograms.png']);
figFile = fullfile(outputDir, [caseLabel '_marginal_histograms.fig']);
exportgraphics(fig, pngFile, 'Resolution', 220);
savefig(fig, figFile);
close(fig)

summaryTable = buildSummaryTable(windows, permsByWindow);
csvFile = fullfile(outputDir, [caseLabel '_marginal_distribution_summary.csv']);
writetable(summaryTable, csvFile);

outputFiles = struct();
outputFiles.png = pngFile;
outputFiles.fig = figFile;
outputFiles.summaryCsv = csvFile;
end


function summaryTable = buildSummaryTable(windows, permsByWindow)
% Build a compact summary table for the plotted distributions.

rows = {};
componentNames = {'Kx', 'Ky', 'Kz'};
for iw = 1:numel(windows)
    perms = permsByWindow{iw};
    for icomp = 1:3
        vals = perms(:, icomp);
        q = quantile(vals, [0.05 0.25 0.50 0.75 0.95]);
        rows(end+1, :) = {windows{iw}, componentNames{icomp}, numel(vals), ...
                          mean(vals), std(vals), q(1), q(2), q(3), q(4), q(5)}; %#ok<AGROW>
    end
end

summaryTable = cell2table(rows, 'VariableNames', ...
    {'Window', 'Component', 'N', 'Mean_mD', 'Std_mD', ...
     'P05_mD', 'P25_mD', 'Median_mD', 'P75_mD', 'P95_mD'});
end
