function [faults, faultSections, smears, mySect, Us, telapsed] = predict_inputs_gom(window, Nsim, varargin)
% Run the GOM paper windows with multiple 3D realizations.
%
% This example is the paper-specific analogue of example0_singleStrati_3D.
% It models one of the six throw windows described in the WRR paper and
% generates a distribution of upscaled permeability values by repeating the
% stochastic PREDICT workflow Nsim times.
%
% Usage:
%   faults = predict_inputs_gom()
%   faults = predict_inputs_gom('famp3', 100)
%   [faults, faultSections, smears, mySect, Us, telapsed] = ...
%       predict_inputs_gom('famp5', 250, 'MakePlots', false)
%
% Inputs:
%   window - one of 'famp1' to 'famp6'. Default: 'famp1'
%   Nsim   - number of realizations. Default: 100
%
% Name-value options:
%   'MakePlots'    - if true, plot permeability histograms. Default: true
%   'CorrCoef'     - copula correlation coefficient. Default: 0.6
%   'ShowProgress' - print progress in the command window. Default: true
%   'UpscaleOpts'  - struct overriding fields of the default U options
%
% Notes:
%   - Run MRST's startup.m before calling this example.
%   - This function models a single paper window per call. To cover all six
%     windows, call it once for each of 'famp1'...'famp6'.

if nargin < 1 || isempty(window)
    window = 'famp1';
end
if nargin < 2 || isempty(Nsim)
    Nsim = 100;
end

assert(ischar(window) || isstring(window), 'window must be a string.')
window = char(lower(string(window)));
assert(isscalar(Nsim) && isnumeric(Nsim) && Nsim >= 1 && mod(Nsim, 1) == 0, ...
       'Nsim must be a positive integer.')

opt.MakePlots = true;
opt.CorrCoef = 0.6;
opt.ShowProgress = true;
opt.UpscaleOpts = struct();
opt = merge_options_relaxed(opt, varargin{:});

assert(exist('mrstModule', 'file') == 2, ...
       ['MRST is not on the MATLAB path. Run startup.m in your MRST ' ...
        'folder before calling predict_inputs_gom.'])
mrstModule add mrst-gui coarsegrid upscaling incomp mpfa mimetic

windowOpt = getWindowOptions(window);

% Flow upscaling options
U.useAcceleration = 1;          % 1 requires MEX setup, 0 otherwise
U.method          = 'tpfa';
U.coarseDims      = [1 1 1];
U.flexible        = true;
U.exportJutulInputs = false;
if ~isempty(fieldnames(opt.UpscaleOpts))
    U = merge_options_relaxed(U, opt.UpscaleOpts);
end

% Define the faulted section once. Realizations are sampled below.
footwall = Stratigraphy(windowOpt.thick{1}, windowOpt.vcl{1}, ...
                        'Dip', windowOpt.dip(1), ...
                        'DepthFaulting', windowOpt.zf(1), ...
                        'DepthBurial', windowOpt.zmax{1});
hangingwall = Stratigraphy(windowOpt.thick{2}, windowOpt.vcl{2}, ...
                           'Dip', windowOpt.dip(2), ...
                           'IsHW', 1, ...
                           'NumLayersFW', footwall.NumLayers, ...
                           'DepthFaulting', windowOpt.zf(2), ...
                           'DepthBurial', windowOpt.zmax{2});

if isfield(windowOpt, 'totThick')
    mySect = FaultedSection(footwall, hangingwall, windowOpt.fDip, ...
                            'maxPerm', windowOpt.maxPerm, ...
                            'totThick', windowOpt.totThick);
else
    mySect = FaultedSection(footwall, hangingwall, windowOpt.fDip, ...
                            'maxPerm', windowOpt.maxPerm);
end
mySect = mySect.getMatPropDistr();

faults = cell(Nsim, 1);
faultSections = cell(Nsim, 1);
smears = cell(Nsim, 1);
Us = cell(Nsim, 1);

tstart = tic;
progressStep = max(1, min(50, floor(Nsim/10)));
for n = 1:Nsim
    [faults{n}, faultSections{n}, smears{n}, Us{n}] = ...
        runSingleWindowRealization(mySect, windowOpt, opt.CorrCoef, U);

    if opt.ShowProgress && (mod(n, progressStep) == 0 || n == Nsim)
        fprintf('Simulation %d / %d completed.\n', n, Nsim);
    end
end
telapsed = toc(tstart);

if opt.MakePlots
    plotUpscaledPerm(faults, 3, 'histOnly');
end
end


function [myFault, faultSections, smears, Urun] = runSingleWindowRealization(mySect, windowOpt, rho, U)
% Run one 3D realization for a single paper window.

nSeg = getNSeg(mySect.Vcl, mySect.IsClayVcl, mySect.DepthFaulting);
myFaultSection = Fault2D(mySect, windowOpt.fDip);
myFault = Fault3D(myFaultSection, mySect);

if U.flexible
    [myFault, Urun] = myFault.getSegmentationLength(U, nSeg.fcn);
else
    myFault = myFault.getSegmentationLength(U, nSeg.fcn);
    Urun = U;
end

G = [];
faultSections = cell(numel(myFault.SegLen), 1);
smears = cell(numel(myFault.SegLen), 1);
for k = 1:numel(myFault.SegLen)
    % Sample material properties for this along-strike section while
    % keeping the realization-wide 3D fault thickness fixed.
    myFaultSection = myFaultSection.getMaterialProperties(mySect, ...
                                                          'corrCoef', rho);
    myFaultSection.MatProps.thick = myFault.Thick;
    if isempty(G)
        G = makeFaultGrid(myFault.Thick, myFault.Disp, ...
                          myFault.Length, myFault.SegLen, Urun);
    end

    smear = Smear(mySect, myFaultSection, G, 1);
    myFaultSection = myFaultSection.placeMaterials(mySect, smear, G);
    myFault = myFault.assignExtrudedVals(G, myFaultSection, k);

    faultSections{k} = myFaultSection;
    smears{k} = smear;
end

[myFault, ~] = myFault.upscaleProps(G, Urun);
end


function opt = getWindowOptions(window)
% Window-specific paper inputs.

opt.window = window;
opt.maxPerm = 175; % [mD], max perm of Amp B interval (sand layers)

switch window
    case 'famp1' % bottom throw window
        opt.thick = {[115.6143 28.8949], [37.6113 37.6861 37.6113 31.6005]};
        opt.vcl = {[0.3 0.65], [0.2878 0.65 0.2878 0.65]};
        opt.dip = [0, -12.0136];
        opt.fDip = 41.6345;
        opt.zf = [200, 200];
        opt.zmax = {[1912 1861], [1934 1909 1884 1860]};

    case 'famp2'
        opt.thick = {[36.9255 35.8537 36.8537 36.3111], ...
                     [36.5042 36.5042 36.4314 36.5042]};
        opt.vcl = {[0.2878 0.65 0.2878 0.65], ...
                   [0.2878 0.65 0.2878 0.65]};
        opt.dip = [0, -13.8951];
        opt.fDip = 43.2508;
        opt.zf = [200, 200];
        opt.zmax = {[1837.5 1812.5 1787.5 1762.5], ...
                    [1837.5 1812.5 1787.5 1762.5]};

    case 'famp3'
        opt.thick = {[35.8537 35.8537 35.8537 35.8537], ...
                     [35.8537 35.8537 35.8537 35.8537]};
        opt.vcl = {[0.2878 0.65 0.2878 0.65], ...
                   [0.2878 0.65 0.2878 0.65]};
        opt.dip = [0, -9.7683];
        opt.fDip = 43.8;
        opt.zf = [200, 200];
        opt.zmax = {[1738.8 1713.8 1688.8 1663.8], ...
                    [1738.8 1713.8 1688.8 1663.8]};

    case 'famp4'
        opt.thick = {[35.8537 35.8537 35.8537 35.9255], ...
                     [35.8537 35.8537 35.8537 35.9255]};
        opt.vcl = {[0.2878 0.65 0.2878 0.65], ...
                   [0.2878 0.65 0.2878 0.65]};
        opt.dip = [0, -4.9456];
        opt.fDip = 44.1811;
        opt.zf = [200, 200];
        opt.zmax = {[1638.8 1613.8 1588.8 1563.8], ...
                    [1638.8 1613.8 1588.8 1563.8]};

    case 'famp5'
        opt.thick = {[35.8537 35.8537 35.8537 35.8537], ...
                     [37.4901 35.2847 35.3553 35.2847]};
        opt.vcl = {[0.2878 0.65 0.2878 0.65], ...
                   [0.2878 0.65 0.2878 0.65]};
        opt.dip = [0, -5.2221];
        opt.fDip = 45.0685;
        opt.zf = [200, 200];
        opt.zmax = {[1538.8 1513.82 1488.75 1463.99], ...
                    [1538.8 1513.82 1488.75 1463.99]};

    case 'famp6' % top throw window
        opt.thick = {[28.2932 33.1042 33.1699 33.1042], 127.6715};
        opt.vcl = {[0.2878 0.65 0.2878 0.65], 0.3266};
        opt.dip = [0, -5];
        opt.fDip = 46.0685;
        opt.zf = [200, 200];
        opt.zmax = {[1440.6 1417.5 1392.5 1367.5], 1400};

    otherwise
        error(['Unsupported window "' window '". Choose one of: ' ...
               'famp1, famp2, famp3, famp4, famp5, famp6.'])
end


end
