function manifest = plot_all_case_marginal_histograms(dataDir, outputDir)
% Plot marginal permeability histograms for every completed case.
%
% Usage:
%   plot_all_case_marginal_histograms()
%   plot_all_case_marginal_histograms('D:\Github\predict_shaowen\examples\data')

if nargin < 1 || isempty(dataDir)
    dataDir = fullfile(fileparts(mfilename('fullpath')), 'data');
end
if nargin < 2 || isempty(outputDir)
    outputDir = fullfile(fileparts(mfilename('fullpath')), ...
                         'visualizations', 'marginal_histograms_all_cases');
end

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

windows = {'famp1', 'famp2', 'famp3', 'famp4', 'famp5', 'famp6'};
caseDirs = dir(fullfile(dataDir, windows{1}, 'case_*'));
caseDirs = caseDirs([caseDirs.isdir]);
caseLabels = sort({caseDirs.name});

if isempty(caseLabels)
    error('No case directories found under %s.', fullfile(dataDir, windows{1}))
end

rows = {};
for icase = 1:numel(caseLabels)
    caseLabel = caseLabels{icase};
    assertCaseComplete(dataDir, windows, caseLabel);

    fprintf('Plotting %s (%d / %d)\n', caseLabel, icase, numel(caseLabels));
    out = plot_case_marginal_distributions(dataDir, caseLabel, outputDir);
    rows(end+1, :) = {caseLabel, out.png, out.fig, out.summaryCsv}; %#ok<AGROW>
end

manifest = cell2table(rows, 'VariableNames', ...
    {'CaseLabel', 'PngFile', 'FigFile', 'SummaryCsv'});
writetable(manifest, fullfile(outputDir, 'marginal_histogram_manifest.csv'));
end


function assertCaseComplete(dataDir, windows, caseLabel)
% Confirm the case exists for every throw window before plotting.

for iw = 1:numel(windows)
    matFile = fullfile(dataDir, windows{iw}, caseLabel, 'predict_runs.mat');
    if ~exist(matFile, 'file')
        error('Missing result file for %s %s: %s', ...
              windows{iw}, caseLabel, matFile)
    end
end
end
