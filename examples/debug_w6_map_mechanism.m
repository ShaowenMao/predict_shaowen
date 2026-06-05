function out = debug_w6_map_mechanism(outputDir, varargin)
%DEBUG_W6_MAP_MECHANISM Inspect why W6 SSSC can be tighter than SCSC.
%
% This diagnostic is intentionally small. It records intermediate map-level
% quantities that the production W6 runs do not save: sand connectivity,
% smear fraction, number of 2D sections with across-fault sand connection,
% and basic smear-domain properties.

if nargin < 1 || isempty(outputDir)
    outputDir = fullfile('D:\codex_gom', 'w6_map_mechanism_debug');
end

parser = inputParser;
parser.addParameter('Nsim', 80, @(x) isnumeric(x) && isscalar(x) && x >= 1);
parser.addParameter('FaultingDepth', 1000, @(x) isnumeric(x) && isscalar(x) && x > 0);
parser.addParameter('SandVcl', 0.1, @(x) isnumeric(x) && isscalar(x) && x >= 0 && x < 0.4);
parser.addParameter('ClayVcl', 0.4, @(x) isnumeric(x) && isscalar(x) && x >= 0.4 && x <= 1);
parser.addParameter('CorrCoef', 0.6, @(x) isnumeric(x) && isscalar(x));
parser.addParameter('BaseSeed', 20260605, @(x) isnumeric(x) && isscalar(x));
parser.parse(varargin{:});
opt = parser.Results;

setupPredictPaths();
assert(exist('mrstModule', 'file') == 2, ...
       ['MRST is not on the MATLAB path. Run startup.m in your MRST ' ...
        'folder before calling this diagnostic.'])
mrstModule add mrst-gui coarsegrid upscaling incomp mpfa mimetic
mrstVerbose off

ensureFolder(outputDir);

U.useAcceleration = 1;
U.method = 'tpfa';
U.coarseDims = [1 1 1];
U.flexible = true;
U.exportJutulInputs = false;

variants = buildVariants(opt);
rows = cell(0, numel(rowNames()));
sectionRows = cell(0, numel(sectionRowNames()));

for iv = 1:numel(variants)
    variant = variants(iv);
    fprintf('\n=== W6 map diagnostic: %s ===\n', variant.Label);
    mySect = buildFaultedSectionForVariant(variant, opt);

    for n = 1:opt.Nsim
        rng(opt.BaseSeed + 1000000 * iv + n - 1, 'twister');
        [perm, stats, segStats] = runOneMapDiagnostic(mySect, variant, opt.CorrCoef, U);
        rows(end+1, :) = makeRow(variant, n, perm, stats); %#ok<AGROW>
        for s = 1:numel(segStats)
            sectionRows(end+1, :) = makeSectionRow(variant, n, s, segStats(s)); %#ok<AGROW>
        end
    end
end

T = cell2table(rows, 'VariableNames', rowNames());
S = cell2table(sectionRows, 'VariableNames', sectionRowNames());
writetable(T, fullfile(outputDir, 'w6_map_mechanism_realization_stats.csv'));
writetable(S, fullfile(outputDir, 'w6_map_mechanism_section_stats.csv'));

summary = groupsummary(T, 'VariantLabel', {'mean', 'median'}, ...
    {'Log10Kxx', 'Log10Kyy', 'Log10Kzz', 'SandFrac3D', 'AcrossFaultSandConnected3D', ...
     'SegmentAcrossFaultConnFraction', 'UpperHalfSandFracMean', 'MeanPsmear'});
writetable(summary, fullfile(outputDir, 'w6_map_mechanism_summary.csv'));

out = struct('realizationStats', T, 'sectionStats', S, 'summary', summary, ...
             'outputDir', outputDir);
save(fullfile(outputDir, 'w6_map_mechanism_debug.mat'), 'out', '-v7.3');
end


function [perm, stats, segStats] = runOneMapDiagnostic(mySect, variant, rho, U)
nSeg = getNSeg(mySect.Vcl, mySect.IsClayVcl, mySect.DepthFaulting);
myFaultSection = Fault2D(mySect, variant.FaultDip);
myFault = Fault3D(myFaultSection, mySect);

if U.flexible
    [myFault, Urun] = myFault.getSegmentationLength(U, nSeg.fcn);
else
    myFault = myFault.getSegmentationLength(U, nSeg.fcn);
    Urun = U;
end

G = [];
segStats = repmat(emptySegStats(), 1, numel(myFault.SegLen));

for k = 1:numel(myFault.SegLen)
    myFaultSection = myFaultSection.getMaterialProperties(mySect, 'corrCoef', rho);
    myFaultSection.MatProps.thick = myFault.Thick;
    if isempty(G)
        G = makeFaultGrid(myFault.Thick, myFault.Disp, myFault.Length, myFault.SegLen, Urun);
    end

    smear = Smear(mySect, myFaultSection, G, 1);
    myFaultSection = myFaultSection.placeMaterials(mySect, smear, G);
    segStats(k) = summarizeSectionMap(myFaultSection.MatMap, smear, mySect);
    myFault = myFault.assignExtrudedVals(G, myFaultSection, k);
end

[myFault, ~] = myFault.upscaleProps(G, Urun);
perm = myFault.Perm ./ (milli*darcy);

[conn, ~] = findSandConn(myFault.Grid.isSmear, Urun.method, 3, G);
stats.SandFrac3D = mean(~myFault.Grid.isSmear);
stats.AcrossFaultSandConnected3D = conn.bc(1);
stats.VerticalSandConnected3D = conn.bc(2);
stats.StrikeSandConnected3D = conn.bc(3);
stats.MaxSandObjectX3D = maxOrNaN(conn.x) / G.cartDims(1);
stats.MaxSandObjectZ3D = maxOrNaN(conn.z) / G.cartDims(3);
stats.MaxSandObjectY3D = maxOrNaN(conn.y) / G.cartDims(2);
stats.NumSegments = numel(myFault.SegLen);
stats.SegmentAcrossFaultConnFraction = mean([segStats.AcrossFaultSandConnected2D]);
stats.UpperHalfSandFracMean = mean([segStats.UpperHalfSandFrac]);
stats.LowerHalfSandFracMean = mean([segStats.LowerHalfSandFrac]);
stats.MeanPsmear = mean([segStats.MeanPsmear], 'omitnan');
stats.MinPsmear = min([segStats.MinPsmear], [], 'omitnan');
end


function s = summarizeSectionMap(M, smear, FS)
[conn, ~] = findSandConn(M.vals, 'tpfa', 2);
nr = size(M.vals, 1);
upperRows = 1:floor(nr/2);
lowerRows = (floor(nr/2)+1):nr;

s = emptySegStats();
s.SandFrac2D = mean(~M.vals(:));
s.UpperHalfSandFrac = mean(~M.vals(upperRows, :), 'all');
s.LowerHalfSandFrac = mean(~M.vals(lowerRows, :), 'all');
s.AcrossFaultSandConnected2D = conn.bc(1);
s.VerticalSandConnected2D = conn.bc(2);
s.MaxSandObjectX2D = maxOrNaN(conn.x) / size(M.vals, 1);
s.MaxSandObjectZ2D = maxOrNaN(conn.z) / size(M.vals, 2);
s.NumMaterialDomains = numel(M.unit);
s.NumClayDomains = sum(M.isclay);
s.MeanPsmear = mean(M.Psmear, 'omitnan');
s.MinPsmear = min(M.Psmear, [], 'omitnan');
s.MaxPsmear = max(M.Psmear, [], 'omitnan');
s.NumContinuousSmearDomains = sum(M.Psmear >= 1);

topClayId = FS.FW.Id(end);
middleClayIds = FS.FW.Id(FS.FW.IsClay & FS.FW.Id ~= topClayId);
s.TopClayDomainCount = sum(M.unit == topClayId);
s.MiddleClayDomainCount = sum(ismember(M.unit, middleClayIds));
s.MiddleClayOwnsAnyDomain = s.MiddleClayDomainCount > 0;
s.TopClayPsmear = mean(smear.Psmear(smear.ParentId == topClayId), 'omitnan');
if isempty(middleClayIds)
    s.MiddleClayPsmear = NaN;
else
    s.MiddleClayPsmear = mean(smear.Psmear(ismember(smear.ParentId, middleClayIds)), 'omitnan');
end
end


function s = emptySegStats()
s = struct();
s.SandFrac2D = NaN;
s.UpperHalfSandFrac = NaN;
s.LowerHalfSandFrac = NaN;
s.AcrossFaultSandConnected2D = false;
s.VerticalSandConnected2D = false;
s.MaxSandObjectX2D = NaN;
s.MaxSandObjectZ2D = NaN;
s.NumMaterialDomains = NaN;
s.NumClayDomains = NaN;
s.MeanPsmear = NaN;
s.MinPsmear = NaN;
s.MaxPsmear = NaN;
s.NumContinuousSmearDomains = NaN;
s.TopClayDomainCount = NaN;
s.MiddleClayDomainCount = NaN;
s.MiddleClayOwnsAnyDomain = false;
s.TopClayPsmear = NaN;
s.MiddleClayPsmear = NaN;
end


function variants = buildVariants(opt)
baseThick = [28.2932 33.1042 33.1699 33.1042];
baseZmax = [1440.6 1417.5 1392.5 1367.5];

variants = struct([]);
variants(1).Label = 'uniform_SCSC';
variants(1).FWPattern = 'SCSC';
variants(1).HWPattern = 'S';
variants(1).FWThick = baseThick;
variants(1).HWThick = 127.6715;
variants(1).FWZmax = baseZmax;
variants(1).HWZmax = 1400;
variants(1).FWVcl = [opt.SandVcl opt.ClayVcl opt.SandVcl opt.ClayVcl];
variants(1).HWVcl = opt.SandVcl;
variants(1).FaultDip = 46.0685;

variants(2).Label = 'nonuniform_SSSC';
variants(2).FWPattern = 'SSSC';
variants(2).HWPattern = 'S';
variants(2).FWThick = baseThick;
variants(2).HWThick = 127.6715;
variants(2).FWZmax = baseZmax;
variants(2).HWZmax = 1400;
variants(2).FWVcl = [opt.SandVcl opt.SandVcl opt.SandVcl opt.ClayVcl];
variants(2).HWVcl = opt.SandVcl;
variants(2).FaultDip = 46.0685;
end


function mySect = buildFaultedSectionForVariant(variant, opt)
footwall = Stratigraphy(variant.FWThick, variant.FWVcl, ...
                        'Dip', 0, ...
                        'DepthFaulting', opt.FaultingDepth, ...
                        'DepthBurial', variant.FWZmax);
hangingwall = Stratigraphy(variant.HWThick, variant.HWVcl, ...
                           'Dip', -5, ...
                           'IsHW', 1, ...
                           'NumLayersFW', footwall.NumLayers, ...
                           'DepthFaulting', opt.FaultingDepth, ...
                           'DepthBurial', variant.HWZmax);
mySect = FaultedSection(footwall, hangingwall, variant.FaultDip, 'maxPerm', 175);
mySect = mySect.getMatPropDistr();
end


function row = makeRow(variant, n, perm, stats)
row = {variant.Label, variant.FWPattern, variant.HWPattern, n, ...
       log10(perm(1)), log10(perm(2)), log10(perm(3)), ...
       stats.SandFrac3D, stats.AcrossFaultSandConnected3D, ...
       stats.VerticalSandConnected3D, stats.StrikeSandConnected3D, ...
       stats.MaxSandObjectX3D, stats.MaxSandObjectZ3D, stats.MaxSandObjectY3D, ...
       stats.NumSegments, stats.SegmentAcrossFaultConnFraction, ...
       stats.UpperHalfSandFracMean, stats.LowerHalfSandFracMean, ...
       stats.MeanPsmear, stats.MinPsmear};
end


function row = makeSectionRow(variant, n, sidx, stats)
row = {variant.Label, variant.FWPattern, variant.HWPattern, n, sidx, ...
       stats.SandFrac2D, stats.UpperHalfSandFrac, stats.LowerHalfSandFrac, ...
       stats.AcrossFaultSandConnected2D, stats.VerticalSandConnected2D, ...
       stats.MaxSandObjectX2D, stats.MaxSandObjectZ2D, ...
       stats.NumMaterialDomains, stats.NumClayDomains, ...
       stats.MeanPsmear, stats.MinPsmear, stats.MaxPsmear, ...
       stats.NumContinuousSmearDomains, stats.TopClayDomainCount, ...
       stats.MiddleClayDomainCount, stats.MiddleClayOwnsAnyDomain, ...
       stats.TopClayPsmear, stats.MiddleClayPsmear};
end


function names = rowNames()
names = {'VariantLabel', 'FWPattern', 'HWPattern', 'Realization', ...
         'Log10Kxx', 'Log10Kyy', 'Log10Kzz', ...
         'SandFrac3D', 'AcrossFaultSandConnected3D', ...
         'VerticalSandConnected3D', 'StrikeSandConnected3D', ...
         'MaxSandObjectX3D', 'MaxSandObjectZ3D', 'MaxSandObjectY3D', ...
         'NumSegments', 'SegmentAcrossFaultConnFraction', ...
         'UpperHalfSandFracMean', 'LowerHalfSandFracMean', ...
         'MeanPsmear', 'MinPsmear'};
end


function names = sectionRowNames()
names = {'VariantLabel', 'FWPattern', 'HWPattern', 'Realization', 'SegmentIndex', ...
         'SandFrac2D', 'UpperHalfSandFrac', 'LowerHalfSandFrac', ...
         'AcrossFaultSandConnected2D', 'VerticalSandConnected2D', ...
         'MaxSandObjectX2D', 'MaxSandObjectZ2D', ...
         'NumMaterialDomains', 'NumClayDomains', ...
         'MeanPsmear', 'MinPsmear', 'MaxPsmear', ...
         'NumContinuousSmearDomains', 'TopClayDomainCount', ...
         'MiddleClayDomainCount', 'MiddleClayOwnsAnyDomain', ...
         'TopClayPsmear', 'MiddleClayPsmear'};
end


function v = maxOrNaN(x)
if isempty(x)
    v = NaN;
else
    v = max(x);
end
end


function ensureFolder(folderPath)
if ~exist(folderPath, 'dir')
    mkdir(folderPath);
end
end


function setupPredictPaths()
thisFile = mfilename('fullpath');
examplesDir = fileparts(thisFile);
repoRoot = fileparts(examplesDir);
pathsToAdd = {repoRoot, fullfile(repoRoot, 'classes'), ...
              fullfile(repoRoot, 'functions'), fullfile(repoRoot, 'utils'), ...
              fullfile(repoRoot, 'utils', 'mrst-based')};
for i = 1:numel(pathsToAdd)
    if exist(pathsToAdd{i}, 'dir')
        addpath(pathsToAdd{i});
    end
end
end
