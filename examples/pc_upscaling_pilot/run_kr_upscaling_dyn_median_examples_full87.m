%RUN_KR_UPSCALING_DYN_MEDIAN_EXAMPLES_FULL87 Dynamic Kr upscaling from replay.
%
% This driver connects replayed PREDICT fine-scale maps to the dynamic
% relative-permeability workflow used in the WRR paper appendix:
%
%   1. build the fine 3D fault-core model from a replayed PREDICT map;
%   2. upscale Pc with invasion percolation to determine the accessible
%      saturation range;
%   3. run a high-rate 3D CO2-brine displacement simulation;
%   4. run pseudo-1D Corey-curve simulations over exponent pairs;
%   5. choose the exponent pair that best matches the 3D saturation history.
%
% Smoke test:
%   setenv('KR_DYN_MAX_ROWS','1')
%   setenv('KR_DYN_USE_PARALLEL','0')
%   setenv('KR_DYN_COREY_STEP','3')
%   setenv('KR_DYN_TIMESTEP_MODE','smoke')
%   setenv('KR_DYN_SMOKE_CARTDIMS','20,4,20')
%   run_kr_upscaling_dyn_median_examples_full87
%
% Production:
%   setenv('KR_DYN_MAX_ROWS','')
%   setenv('KR_DYN_USE_PARALLEL','1')
%   setenv('KR_DYN_NUM_WORKERS','6')
%   setenv('KR_DYN_COREY_STEP','0.2')
%   setenv('KR_DYN_TIMESTEP_MODE','paper')
%   run_kr_upscaling_dyn_median_examples_full87
%
% Swi-medoid representative reduction:
%   setenv('KR_DYN_SELECTION_MODE','swi_medoid')
%   % Runs the actual slice at the scalar effective-Swi medoid for each
%   % case/window and exports normalized Kr shapes plus slice-specific Pc
%   % endpoint mappings.

clear; clc;

scriptDir = fileparts(mfilename('fullpath'));
examplesDir = fileparts(scriptDir);
repoRoot = fileparts(examplesDir);
addpath(scriptDir);

cfg = struct();
cfg.geologyId = string(envOrDefault("KR_DYN_GEOLOGY_ID", "s05_c012"));
cfg.level3CaseIds = parseIdList(envOrDefault("KR_DYN_CASE_IDS", "1,3,4,7"));
cfg.caseToken = caseTokenFromIds(cfg.level3CaseIds);
cfg.windows = ["famp1", "famp2", "famp3", "famp4", "famp5", "famp6"];
% Common grid for exported Kr tables only. Effective Swi is inferred from
% the native precomputed Pc endpoint via max(Sg), matching the WRR workflow.
cfg.sgGrid = linspace(0.02, 0.68, 80);
cfg.sourceRoot = envOrDefault("KR_DYN_REPLAY_ROOT", ...
    envOrDefault("FULL87_REPLAY_OUTPUT_ROOT", defaultReplayRoot()));
cfg.replaySummaryCsv = envOrDefault("KR_DYN_REPLAY_SUMMARY_CSV", ...
    fullfile(cfg.sourceRoot, 'tables', sprintf( ...
    'replay_summary_with_full87_context_%s_%s.csv', ...
    cfg.geologyId, cfg.caseToken)));
cfg.pcPrestepMode = lower(strtrim(string(envOrDefault( ...
    "KR_DYN_PC_PRESTEP_MODE", "precomputed"))));
assert(any(cfg.pcPrestepMode == ["original", "precomputed"]), ...
    'KR_DYN_PC_PRESTEP_MODE must be original or precomputed.');
cfg.precomputedPcCurveCsv = envOrDefault("KR_DYN_PRECOMPUTED_PC_CURVE_CSV", ...
    fullfile(fileparts(cfg.sourceRoot), 'pc_ip', 'curves', sprintf( ...
    'pc_curve_points_%s_%s_ip_full87.csv', cfg.geologyId, cfg.caseToken)));
cfg.precomputedPcNativeCurveCsv = envOrDefault("KR_DYN_PRECOMPUTED_PC_NATIVE_CURVE_CSV", ...
    fullfile(fileparts(cfg.sourceRoot), 'pc_ip', 'curves', sprintf( ...
    'pc_native_curve_points_%s_%s_ip_full87.csv', cfg.geologyId, cfg.caseToken)));
cfg.precomputedPcSummaryCsv = envOrDefault("KR_DYN_PRECOMPUTED_PC_SUMMARY_CSV", ...
    fullfile(fileparts(cfg.sourceRoot), 'pc_ip', 'tables', sprintf( ...
    'pc_curve_summary_%s_%s_ip_full87.csv', cfg.geologyId, cfg.caseToken)));
cfg.permeabilityInput = envOrDefault("KR_DYN_PERMEABILITY_INPUT", ...
    defaultPermeabilityInput());
cfg.upscalingZip = envOrDefault("UPSCALING_ZIP", ...
    fullfile(repoRoot, 'upscaling.zip'));
cfg.upscalingRoot = envOrDefault("KR_DYN_UPSCALING_ROOT", ...
    envOrDefault("UPSCALING_ROOT", defaultUpscalingRoot()));
cfg.mrstRoot = envOrDefault("MRST_ROOT", defaultMrstRoot());
cfg.outputRoot = envOrDefault("KR_DYN_OUTPUT_ROOT", ...
    fullfile(defaultWorkflowRoot(), 'kr_upscaling_dyn_median_examples_full87'));
cfg.selectionMode = canonicalSelectionMode(envOrDefault( ...
    "KR_DYN_SELECTION_MODE", "all"));
assert(any(cfg.selectionMode == ["all", "swi_medoid"]), ...
    'KR_DYN_SELECTION_MODE must be all or swi_medoid.');
cfg.normalizedShapePoints = round(parseNumericEnv( ...
    "KR_DYN_NORMALIZED_SHAPE_POINTS", 101));
assert(cfg.normalizedShapePoints >= 2, ...
    'KR_DYN_NORMALIZED_SHAPE_POINTS must be at least 2.');
cfg.exportReservoirReady = parseLogicalEnv( ...
    "KR_DYN_EXPORT_RESERVOIR_READY", true);
cfg.reservoirPcRepresentation = lower(strtrim(string(envOrDefault( ...
    "KR_DYN_RESERVOIR_PC_REPRESENTATION", "full_slice"))));
assert(any(cfg.reservoirPcRepresentation == ...
    ["full_slice", "pe_branch_medoid", "both"]), ...
    ['KR_DYN_RESERVOIR_PC_REPRESENTATION must be full_slice, ', ...
     'pe_branch_medoid, or both.']);
cfg.peBranchMinLog10Gap = parseNumericEnv( ...
    "KR_DYN_PE_BRANCH_MIN_LOG10_GAP", 1.0);
cfg.peBranchMinCount = parseNumericEnv( ...
    "KR_DYN_PE_BRANCH_MIN_COUNT", 2);
cfg.peBranchMaxBranches = parseNumericEnv( ...
    "KR_DYN_PE_BRANCH_MAX_BRANCHES", 3);
cfg.onlyRows = parseIntegerListEnv("KR_DYN_ONLY_ROWS");
cfg.smokeCartDims = parseIntegerListEnv("KR_DYN_SMOKE_CARTDIMS");

maxRowsText = strtrim(string(getenv('KR_DYN_MAX_ROWS')));
if maxRowsText ~= ""
    cfg.maxRows = str2double(maxRowsText);
else
    cfg.maxRows = inf;
end
outputTagRaw = strtrim(string(getenv('KR_DYN_OUTPUT_TAG')));
if outputTagRaw ~= ""
    outputTag = matlab.lang.makeValidName(outputTagRaw);
    cfg.outputRoot = cfg.outputRoot + "_" + outputTag;
end
cfg.useParallel = parseLogicalEnv("KR_DYN_USE_PARALLEL", ~isfinite(cfg.maxRows));
cfg.numWorkers = parseNumericEnv("KR_DYN_NUM_WORKERS", 6);
cfg.coreyStep = parseNumericEnv("KR_DYN_COREY_STEP", 0.2);
cfg.pcNval = parseNumericEnv("KR_DYN_PC_NVAL", 32);
cfg.timestepMode = lower(strtrim(string(getenv('KR_DYN_TIMESTEP_MODE'))));
if cfg.timestepMode == ""
    cfg.timestepMode = "paper";
end
cfg.useMexBackend = parseLogicalEnv("KR_DYN_USE_MEX", false);
cfg.nlsMaxIterations = parseNumericEnv("KR_DYN_NLS_MAX_ITER", NaN);
cfg.nlsMaxTimestepCuts = parseNumericEnv("KR_DYN_NLS_MAX_CUTS", NaN);
cfg.disableRunMatCache = parseLogicalEnv("KR_DYN_DISABLE_RUN_MAT_CACHE", ...
    ~isempty(cfg.onlyRows));
cfg.oneDMatchMethod = lower(strtrim(string(getenv('KR_DYN_1D_METHOD'))));
if cfg.oneDMatchMethod == ""
    cfg.oneDMatchMethod = "ad";
end
assert(any(cfg.oneDMatchMethod == ["transport", "ad"]), ...
    'KR_DYN_1D_METHOD must be transport or ad.');
cfg.oneDAdSolver = lower(strtrim(string(getenv('KR_DYN_1D_AD_SOLVER'))));
if cfg.oneDAdSolver == ""
    cfg.oneDAdSolver = "legacy";
end
assert(any(cfg.oneDAdSolver == ["legacy", "robust"]), ...
    'KR_DYN_1D_AD_SOLVER must be legacy or robust.');
cfg.linearSolverPolicy = lower(strtrim(string(getenv('KR_DYN_LINEAR_SOLVER'))));
if cfg.linearSolverPolicy == ""
    cfg.linearSolverPolicy = "amgcl_auto";
end
assert(any(cfg.linearSolverPolicy == ["backslash", "amgcl_auto", "amgcl_require"]), ...
    'KR_DYN_LINEAR_SOLVER must be backslash, amgcl_auto, or amgcl_require.');
cfg.oneDLinearSolverPolicy = lower(strtrim(string(getenv('KR_DYN_1D_LINEAR_SOLVER'))));
if cfg.oneDLinearSolverPolicy == ""
    cfg.oneDLinearSolverPolicy = cfg.linearSolverPolicy;
end
assert(any(cfg.oneDLinearSolverPolicy == ["backslash", "amgcl_auto", "amgcl_require"]), ...
    'KR_DYN_1D_LINEAR_SOLVER must be backslash, amgcl_auto, or amgcl_require.');
cfg.threeDMaxIterations = parseNumericEnv("KR_DYN_3D_MAX_ITER", 10);
cfg.threeDMaxTimestepCuts = parseNumericEnv("KR_DYN_3D_MAX_CUTS", 14);
cfg.threeDUseLineSearch = parseLogicalEnv("KR_DYN_3D_USE_LINESEARCH", true);
cfg.threeDUseRelaxation = parseLogicalEnv("KR_DYN_3D_USE_RELAXATION", false);
cfg.threeDNumThreads = parseNumericEnv("KR_DYN_3D_NUM_THREADS", 2);
cfg.transportCfl = parseNumericEnv("KR_DYN_TRANSPORT_CFL", 0.35);
cfg.transportMaxSubstepsPerReport = parseNumericEnv( ...
    "KR_DYN_TRANSPORT_MAX_SUBSTEPS_PER_REPORT", 400);
cfg.transportRateScale = parseNumericEnv("KR_DYN_TRANSPORT_RATE_SCALE", NaN);
if isfinite(cfg.maxRows)
    if ~isempty(cfg.smokeCartDims)
        gridTag = sprintf('_smokeCart%d_%d_%d', cfg.smokeCartDims);
    else
        gridTag = "_fullGrid";
    end
    cfg.outputRoot = cfg.outputRoot + "_smoke" + string(cfg.maxRows) + gridTag;
end

cfg.curveDir = fullfile(cfg.outputRoot, 'curves');
cfg.tableDir = fullfile(cfg.outputRoot, 'tables');
cfg.checkpointDir = fullfile(cfg.curveDir, 'curve_checkpoints');
ensureFolder(cfg.curveDir);
ensureFolder(cfg.tableDir);
ensureFolder(cfg.checkpointDir);

initializeDynamicKrPaths(cfg);
preflightDynamicKrLinearSolvers(cfg);
cfg.originalDeckFile = fullfile(cfg.upscalingRoot, 'eclipse_data_files', ...
    'gom_forUps_theta30_PVDO_incompRock.DATA');
assert(exist(cfg.originalDeckFile, 'file') == 2, ...
    'Missing deck file: %s', cfg.originalDeckFile);

fprintf('\n=== Load replay summary for full-87 dynamic Kr upscaling ===\n')
assert(exist(cfg.replaySummaryCsv, 'file') == 2, ...
    'Missing replay summary: %s', cfg.replaySummaryCsv);
replaySummaryAll = readtable(cfg.replaySummaryCsv, 'TextType', 'string');
caseMask = replaySummaryAll.GeologyId == cfg.geologyId & ...
    ismember(replaySummaryAll.Level3CaseId, cfg.level3CaseIds);
replaySummary = replaySummaryAll(caseMask, :);
replaySummary.WindowOrder = windowOrder(replaySummary.Window);
replaySummary = sortrows(replaySummary, ...
    {'Level3CaseId', 'SliceIndex', 'WindowOrder'});
replaySummary.ProductionCurveId = (1:height(replaySummary))';

expectedRows = 87 * numel(cfg.windows) * numel(cfg.level3CaseIds);
if height(replaySummary) ~= expectedRows
    allowPartial = parseLogicalEnv("KR_DYN_ALLOW_PARTIAL_REPLAY", ...
        isfinite(cfg.maxRows) || ~isempty(cfg.onlyRows));
    assert(allowPartial, ...
        'Expected %d rows for cases %s, found %d.', ...
        expectedRows, cfg.caseToken, height(replaySummary));
    warning('Expected %d rows for cases %s, found %d; continuing with partial replay data.', ...
        expectedRows, cfg.caseToken, height(replaySummary));
end

selectionTable = table();
pcSummaryForMapping = table();
if cfg.selectionMode == "swi_medoid"
    assert(exist(cfg.precomputedPcSummaryCsv, 'file') == 2, ...
        'Swi-medoid selection requires the completed Pc summary CSV: %s', ...
        cfg.precomputedPcSummaryCsv);
    pcSummaryForMapping = readtable(cfg.precomputedPcSummaryCsv, ...
        'TextType', 'string');
    assert(all(ismember({'PoreVolume', 'BulkVolume', ...
        'UpscaledPorosity'}, pcSummaryForMapping.Properties.VariableNames)), ...
        ['Pc summary lacks reservoir porosity fields. Run the Pc stage ', ...
         'once with the current code; cached Pc curves will be retained ', ...
         'and porosity will be backfilled directly from replay maps.']);
    [replaySummary, selectionTable] = select_swi_medoid_replay_rows( ...
        replaySummary, pcSummaryForMapping, ...
        cfg.level3CaseIds, cfg.windows);
    selectionCsv = fullfile(cfg.tableDir, sprintf( ...
        'kr_representative_selection_%s_%s_swi_medoid.csv', ...
        cfg.geologyId, cfg.caseToken));
    writetable(selectionTable, selectionCsv);
    fprintf(['Selected %d representative replay rows using the ', ...
        'scalar effective-Swi medoid.\n'], height(replaySummary));
    fprintf('Saved representative selection table: %s\n', selectionCsv);
end

if ~isempty(cfg.onlyRows)
    assert(all(cfg.onlyRows >= 1 & cfg.onlyRows <= height(replaySummary)), ...
        'KR_DYN_ONLY_ROWS contains row ids outside 1:%d.', height(replaySummary));
    replaySummary = replaySummary(cfg.onlyRows, :);
elseif isfinite(cfg.maxRows)
    replaySummary = replaySummary(1:min(cfg.maxRows, height(replaySummary)), :);
end
fprintf('Using %d replayed rows from: %s\n', ...
    height(replaySummary), cfg.replaySummaryCsv);
fprintf('Kr row-selection mode: %s\n', cfg.selectionMode);

krOpt = krDynOptions(cfg);
fprintf('Reference deck: %s\n', cfg.originalDeckFile);
fprintf('Kr mode = %s, Pc mode = %s, Corey exponent step = %.3g, timestep mode = %s.\n', ...
    krOpt.krMode, krOpt.pcMode, krOpt.coreyStep, krOpt.timestepMode);
fprintf('Kr Pc pre-step mode = %s', krOpt.pcPrestepMode);
if ~isempty(krOpt.precomputedPcTable)
    fprintf(' (%s)', cfg.precomputedPcCurveCsv);
end
fprintf('\n');
fprintf('1D matching method = %s\n', krOpt.oneDMatchMethod);
if krOpt.oneDMatchMethod == "ad"
    fprintf('1D AD solver mode = %s\n', krOpt.oneDAdSolver);
end
fprintf('3D linear solver policy = %s\n', krOpt.linearSolverPolicy);
fprintf('1D linear solver policy = %s\n', krOpt.oneDLinearSolverPolicy);
fprintf('Use parallel execution: %d', cfg.useParallel);
if cfg.useParallel
    fprintf(' (%d workers)', cfg.numWorkers);
end
fprintf('\n');

if cfg.selectionMode == "all"
    outputSuffix = "dyn_full87";
else
    outputSuffix = "dyn_swi_medoid";
end
curveLongCsv = fullfile(cfg.curveDir, sprintf( ...
    'kr_curve_points_%s_%s_%s.csv', ...
    cfg.geologyId, cfg.caseToken, outputSuffix));
curveSummaryCsv = fullfile(cfg.tableDir, sprintf( ...
    'kr_curve_summary_%s_%s_%s.csv', ...
    cfg.geologyId, cfg.caseToken, outputSuffix));
curveMatFile = fullfile(cfg.curveDir, sprintf( ...
    'kr_curves_%s_%s_%s.mat', ...
    cfg.geologyId, cfg.caseToken, outputSuffix));

fprintf('\n=== Compute dynamic Kr curves ===\n')
useCachedCurveMat = ~cfg.disableRunMatCache && ...
    exist(curveMatFile, 'file') == 2;
if useCachedCurveMat
    fprintf('Loading cached dynamic Kr curve MAT: %s\n', curveMatFile);
    cached = load(curveMatFile, 'curveMat');
    curveMat = cached.curveMat;
    useCachedCurveMat = cachedCurveRowsMatchSelection( ...
        curveMat, replaySummary);
    if ~useCachedCurveMat
        fprintf(['Cached Kr MAT does not match the current replay-row ', ...
            'selection; rebuilding from row checkpoints.\n']);
    end
end
if ~useCachedCurveMat
    [curveLong, curveSummary, curveMat] = computeKrDynCurves( ...
        replaySummary, krOpt, cfg);
    writetable(curveLong, curveLongCsv);
    writetable(curveSummary, curveSummaryCsv);
    save(curveMatFile, 'curveMat', 'krOpt', 'cfg', '-v7.3');
    fprintf('Saved dynamic Kr curve points: %s\n', curveLongCsv);
    fprintf('Saved dynamic Kr curve summary: %s\n', curveSummaryCsv);
    fprintf('Saved dynamic Kr curve MAT: %s\n', curveMatFile);
end

if cfg.selectionMode == "swi_medoid"
    if ~exist('curveSummary', 'var')
        curveSummary = curveMat.summary;
    end
    exportPcGuidedRepresentativeKr( ...
        curveMat, curveSummary, pcSummaryForMapping, selectionTable, cfg);
end

fprintf('\nDynamic Kr upscaling run complete.\n')
fprintf('Output root: %s\n', cfg.outputRoot);


function tf = cachedCurveRowsMatchSelection(curveMat, replaySummary)
% Require cached curves to represent exactly the currently selected rows.

tf = isfield(curveMat, 'summary') && ...
    ismember('SourceRow', curveMat.summary.Properties.VariableNames) && ...
    height(curveMat.summary) == height(replaySummary);
if ~tf
    return
end
cachedRows = double(curveMat.summary.SourceRow(:));
selectedRows = double(replaySummary.SourceRow(:));
tf = isequal(cachedRows, selectedRows);
end


function exportPcGuidedRepresentativeKr( ...
        curveMat, curveSummary, pcSummary, selectionTable, cfg)
% Export normalized representative shapes and locally rescaled slice curves.

uGrid = linspace(0, 1, cfg.normalizedShapePoints)';
nRep = height(curveSummary);
shapeRows = cell(nRep * numel(uGrid), 15);
shapeIndex = containers.Map('KeyType', 'char', 'ValueType', 'double');
shapeKrg = cell(nRep, 1);
shapeKrw = cell(nRep, 1);
shapeIds = strings(nRep, 1);
shapeRowId = 0;

for i = 1:nRep
    [nativeSg, nativeKrg, nativeKrw] = curveMatNativeValues( ...
        curveMat, curveSummary, i);
    [krgShape, krwShape] = normalizedKrShape( ...
        nativeSg, nativeKrg, nativeKrw, uGrid);
    shapeKrg{i} = krgShape;
    shapeKrw{i} = krwShape;
    shapeId = sprintf('%s_case%02d_%s', char(curveSummary.GeologyId(i)), ...
        curveSummary.Level3CaseId(i), char(curveSummary.Window(i)));
    shapeIds(i) = string(shapeId);
    shapeIndex(shapeId) = i;

    for j = 1:numel(uGrid)
        shapeRowId = shapeRowId + 1;
        shapeRows(shapeRowId, :) = { ...
            curveSummary.GeologyId(i), curveSummary.Level3CaseId(i), ...
            curveSummary.Level3CaseName(i), curveSummary.Window(i), ...
            string(shapeId), curveSummary.ProductionCurveId(i), ...
            curveSummary.SourceRow(i), curveSummary.SliceIndex(i), ...
            curveSummary.PcMaxSg(i), ...
            curveSummary.IrreducibleWaterSaturation(i), ...
            curveSummary.BrineCoreyExponent(i), ...
            curveSummary.GasCoreyExponent(i), ...
            uGrid(j), krgShape(j), krwShape(j)};
    end
end

shapeTable = cell2table(shapeRows(1:shapeRowId, :), 'VariableNames', { ...
    'GeologyId', 'Level3CaseId', 'Level3CaseName', 'Window', ...
    'ShapeId', 'RepresentativeProductionCurveId', ...
    'RepresentativeReplaySourceRow', 'RepresentativeSliceIndex', ...
    'RepresentativeBulkSgMax', 'RepresentativeEffectiveSwi', ...
    'BrineCoreyExponent', 'GasCoreyExponent', 'NormalizedSg', ...
    'Krg', 'Krw'});

pcMask = pcSummary.GeologyId == cfg.geologyId & ...
    ismember(pcSummary.Level3CaseId, cfg.level3CaseIds);
pcRows = pcSummary(pcMask, :);
pcRows.WindowOrder = windowOrder(pcRows.Window);
pcRows = sortrows(pcRows, {'Level3CaseId', 'SliceIndex', 'WindowOrder'});
availableShape = false(height(pcRows), 1);
for i = 1:height(pcRows)
    key = sprintf('%s_case%02d_%s', char(pcRows.GeologyId(i)), ...
        pcRows.Level3CaseId(i), char(pcRows.Window(i)));
    availableShape(i) = isKey(shapeIndex, key);
end
pcRows = pcRows(availableShape, :);
if ismember('EffectiveSwi', pcRows.Properties.VariableNames)
    localSwi = double(pcRows.EffectiveSwi);
else
    localSwi = 1.0 - double(pcRows.BulkSgMax);
end
localSgMax = double(pcRows.BulkSgMax);

nMap = height(pcRows);
mappingRows = cell(nMap, 17);
sliceRows = cell(nMap * numel(uGrid), 17);
sliceRowId = 0;

for i = 1:nMap
    shapeId = sprintf('%s_case%02d_%s', char(pcRows.GeologyId(i)), ...
        pcRows.Level3CaseId(i), char(pcRows.Window(i)));
    repIdx = shapeIndex(shapeId);
    if ismember('CurveId', pcRows.Properties.VariableNames)
        pcCurveId = double(pcRows.CurveId(i));
    else
        pcCurveId = NaN;
    end

    mappingRows(i, :) = { ...
        pcCurveId, double(pcRows.ReplaySourceRow(i)), ...
        pcRows.GeologyId(i), pcRows.Level3CaseId(i), ...
        pcRows.Level3CaseName(i), pcRows.Window(i), ...
        double(pcRows.SliceIndex(i)), string(shapeId), ...
        double(curveSummary.ProductionCurveId(repIdx)), ...
        double(curveSummary.SourceRow(repIdx)), ...
        double(curveSummary.SliceIndex(repIdx)), ...
        localSgMax(i), localSwi(i), ...
        double(curveSummary.BrineCoreyExponent(repIdx)), ...
        double(curveSummary.GasCoreyExponent(repIdx)), ...
        double(pcRows.SelectedSampleIndex(i)), ...
        double(pcRows.ReplaySeed(i))};

    for j = 1:numel(uGrid)
        sliceRowId = sliceRowId + 1;
        localSg = uGrid(j) * localSgMax(i);
        sliceRows(sliceRowId, :) = { ...
            pcCurveId, double(pcRows.ReplaySourceRow(i)), ...
            pcRows.GeologyId(i), pcRows.Level3CaseId(i), ...
            pcRows.Level3CaseName(i), pcRows.Window(i), ...
            double(pcRows.SliceIndex(i)), string(shapeId), ...
            double(curveSummary.SourceRow(repIdx)), ...
            localSwi(i), localSgMax(i), uGrid(j), localSg, ...
            1.0 - localSg, shapeKrg{repIdx}(j), shapeKrw{repIdx}(j), ...
            j == numel(uGrid)};
    end
end

mappingTable = cell2table(mappingRows, 'VariableNames', { ...
    'PcCurveId', 'PcReplaySourceRow', 'GeologyId', 'Level3CaseId', ...
    'Level3CaseName', 'Window', 'SliceIndex', 'ShapeId', ...
    'RepresentativeProductionCurveId', 'RepresentativeKrReplaySourceRow', ...
    'RepresentativeKrSliceIndex', 'BulkSgMax', 'EffectiveSwi', ...
    'BrineCoreyExponent', 'GasCoreyExponent', ...
    'SelectedSampleIndex', 'ReplaySeed'});
sliceCurveTable = cell2table(sliceRows(1:sliceRowId, :), ...
    'VariableNames', {'PcCurveId', 'PcReplaySourceRow', 'GeologyId', ...
    'Level3CaseId', 'Level3CaseName', 'Window', 'SliceIndex', ...
    'ShapeId', 'RepresentativeKrReplaySourceRow', 'EffectiveSwi', ...
    'BulkSgMax', 'NormalizedSg', 'GasSaturation', 'WaterSaturation', ...
    'Krg', 'Krw', 'IsEndpoint'});

token = sprintf('%s_%s', cfg.geologyId, cfg.caseToken);
shapeCsv = fullfile(cfg.curveDir, sprintf( ...
    'kr_normalized_representative_shapes_%s.csv', token));
mappingCsv = fullfile(cfg.tableDir, sprintf( ...
    'kr_slice_endpoint_mapping_%s.csv', token));
sliceCsv = fullfile(cfg.curveDir, sprintf( ...
    'kr_slice_curves_swi_medoid_%s.csv', token));
matFile = fullfile(cfg.curveDir, sprintf( ...
    'kr_swi_medoid_representatives_%s.mat', token));

writetable(shapeTable, shapeCsv);
writetable(mappingTable, mappingCsv);
writetable(sliceCurveTable, sliceCsv);
representativeKr = struct();
representativeKr.selection = selectionTable;
representativeKr.normalizedShapes = shapeTable;
representativeKr.sliceEndpointMapping = mappingTable;
representativeKr.sliceCurves = sliceCurveTable;
representativeKr.normalizedSgGrid = uGrid;
save(matFile, 'representativeKr', 'cfg', '-v7.3');

fprintf('Saved normalized representative Kr shapes: %s\n', shapeCsv);
fprintf('Saved slice endpoint mapping: %s\n', mappingCsv);
fprintf('Saved Swi-medoid slice Kr curves: %s\n', sliceCsv);
fprintf('Saved Swi-medoid Kr MAT: %s\n', matFile);

isFullProduction = ~isfinite(cfg.maxRows) && isempty(cfg.onlyRows);
if cfg.exportReservoirReady && isFullProduction
    reservoirDir = fullfile(cfg.outputRoot, 'reservoir_ready');
    selectionCsvForExport = fullfile(cfg.tableDir, sprintf( ...
        'kr_representative_selection_%s_%s_swi_medoid.csv', ...
        cfg.geologyId, cfg.caseToken));
    fullSliceOutputs = export_reservoir_ready_pc_kr_cases( ...
        cfg.precomputedPcNativeCurveCsv, sliceCsv, ...
        selectionCsvForExport, reservoirDir, ...
        'PorosityInput', cfg.precomputedPcSummaryCsv, ...
        'PermeabilityInput', cfg.permeabilityInput);
    if any(cfg.reservoirPcRepresentation == ...
            ["pe_branch_medoid", "both"])
        reducedDir = fullfile(cfg.outputRoot, ...
            'reservoir_ready_pe_branch_medoid');
        for fileId = 1:numel(fullSliceOutputs.matFiles)
            build_pe_branch_medoid_reservoir_inputs( ...
                fullSliceOutputs.matFiles(fileId), reducedDir, ...
                'MinLog10PeGap', cfg.peBranchMinLog10Gap, ...
                'MinBranchCount', cfg.peBranchMinCount, ...
                'MaxBranches', cfg.peBranchMaxBranches);
        end
    end
elseif cfg.exportReservoirReady
    fprintf(['Skipping reservoir-ready export for a partial/smoke run; ', ...
        'full 87-slice coverage is required.\n']);
end
end


function [sg, krg, krw] = curveMatNativeValues(curveMat, curveSummary, rowId)
% Return native Kr arrays, reconstructing the fitted Corey form if needed.

if isfield(curveMat, 'nativeSg') && ...
        numel(curveMat.nativeSg) >= rowId && ...
        ~isempty(curveMat.nativeSg{rowId})
    sg = curveMat.nativeSg{rowId};
    krg = curveMat.nativeKrg{rowId};
    krw = curveMat.nativeKrw{rowId};
    return
end

u = linspace(0, 1, 101)';
sgMax = double(curveSummary.PcMaxSg(rowId));
sg = u * sgMax;
krg = u .^ double(curveSummary.GasCoreyExponent(rowId));
krw = (1.0 - u) .^ double(curveSummary.BrineCoreyExponent(rowId));
end


function [sg, krg, krw] = nativeKrCurveValues(curve)
% Return checkpoint-native Kr arrays or the equivalent fitted Corey curve.

if isfield(curve, 'nativeSg') && ~isempty(curve.nativeSg)
    sg = curve.nativeSg(:);
    krg = curve.nativeKrg(:);
    krw = curve.nativeKrw(:);
    return
end

u = linspace(0, 1, 101)';
sg = u * curve.pcMaxSg;
krg = u .^ curve.gasCoreyExponent;
krw = (1.0 - u) .^ curve.brineCoreyExponent;
end


function [krgShape, krwShape] = normalizedKrShape( ...
        sg, krg, krw, normalizedGrid)
% Interpolate one native Kr pair onto a unit saturation coordinate.

sg = double(sg(:));
krg = min(max(double(krg(:)), 0), 1);
krw = min(max(double(krw(:)), 0), 1);
valid = isfinite(sg) & isfinite(krg) & isfinite(krw);
sg = sg(valid);
krg = krg(valid);
krw = krw(valid);
assert(numel(sg) >= 2 && max(sg) > 0, ...
    'Representative Kr curve requires at least two valid saturation points.');

u = sg ./ max(sg);
[u, order] = sort(u);
krg = krg(order);
krw = krw(order);
[u, keep] = unique(u, 'stable');
krg = krg(keep);
krw = krw(keep);

if u(1) > 0
    u = [0; u];
    krg = [0; krg];
    krw = [1; krw];
end
if u(end) < 1
    u = [u; 1];
    krg = [krg; krg(end)];
    krw = [krw; krw(end)];
end

krgShape = interp1(u, krg, normalizedGrid, 'linear');
krwShape = interp1(u, krw, normalizedGrid, 'linear');
krgShape = min(max(krgShape(:), 0), 1);
krwShape = min(max(krwShape(:), 0), 1);
end


function initializeDynamicKrPaths(cfg)
% Extract and patch original dynamic-upscaling helpers, then add MRST paths.

assert(exist(cfg.upscalingZip, 'file') == 2, ...
    'Missing archived upscaling bundle: %s', cfg.upscalingZip);
materializePatchedUpscalingBundle(cfg.upscalingZip, cfg.upscalingRoot);

if exist(fullfile(cfg.mrstRoot, 'startup.m'), 'file') == 2
    run(fullfile(cfg.mrstRoot, 'startup.m'));
else
    warning('MRST startup not found. Continuing with current MATLAB path: %s', ...
        cfg.mrstRoot);
end
addDynamicKrMrstModules();
addpath(cfg.upscalingRoot);
addpath(fullfile(cfg.upscalingRoot, 'upscaling'));
end


function initializeDynamicKrWorkerRuntime(cfg)
% Initialize MRST, upscaling helpers, and AMGCL on one process-pool worker.

persistent initialized
if ~isempty(initialized) && initialized
    return
end

if exist(fullfile(cfg.mrstRoot, 'startup.m'), 'file') == 2
    run(fullfile(cfg.mrstRoot, 'startup.m'));
else
    warning('MRST startup not found on worker. Continuing with current path: %s', ...
        cfg.mrstRoot);
end
addDynamicKrMrstModules();
addpath(cfg.upscalingRoot);
addpath(fullfile(cfg.upscalingRoot, 'upscaling'));

needsWorkerAmgcl = cfg.linearSolverPolicy == "amgcl_require" || ...
    (cfg.oneDMatchMethod == "ad" && cfg.oneDLinearSolverPolicy == "amgcl_require");
if needsWorkerAmgcl
    [isReady, reportText] = checkAmgclRuntimeReady();
    if ~isReady
        error('AMGCL:WorkerRequiredUnavailable', ...
            ['AMGCL is required but not usable on a MATLAB worker.', newline, ...
             '%s'], reportText);
    end
end

initialized = true;
end


function addDynamicKrMrstModules()
% Add every MRST module needed by the dynamic Pc/Kr workflow.

mrstModule add mrst-gui mimetic upscaling incomp coarsegrid deckformat ...
    ad-props ad-core ad-blackoil linearsolvers sequential
end


function preflightDynamicKrLinearSolvers(cfg)
% Fail early when production settings require AMGCL but the MEX is unusable.

fprintf('\n=== Linear solver preflight ===\n');
fprintf('3D dynamic linear solver policy: %s\n', cfg.linearSolverPolicy);
fprintf('1D AD linear solver policy: %s\n', cfg.oneDLinearSolverPolicy);

needs3D = any(cfg.linearSolverPolicy == ["amgcl_auto", "amgcl_require"]);
needs1D = cfg.oneDMatchMethod == "ad" && ...
    any(cfg.oneDLinearSolverPolicy == ["amgcl_auto", "amgcl_require"]);

if ~(needs3D || needs1D)
    warning('AMGCL:Disabled', ...
        ['AMGCL is disabled by configuration. Dynamic Kr simulations may ', ...
         'be substantially slower.']);
    return
end

[isReady, reportText] = checkAmgclRuntimeReady();
fprintf('%s\n', reportText);

if ~isReady
    requiresAmgcl = cfg.linearSolverPolicy == "amgcl_require" || ...
        (needs1D && cfg.oneDLinearSolverPolicy == "amgcl_require");
    if requiresAmgcl
        error('AMGCL:RequiredUnavailable', ...
            ['AMGCL is required but not usable. Refusing to start dynamic ', ...
             'Kr simulation. Run examples/pc_upscaling_pilot/engaging/', ...
             'setup_mrst_amgcl.sh on Engaging.']);
    end
    warning('AMGCL:UnavailableFallback', ...
        ['AMGCL is not usable. The workflow may fall back to a slower ', ...
         'MRST solver. Use KR_DYN_LINEAR_SOLVER=amgcl_require for production.']);
else
    fprintf('AMGCL preflight passed. Dynamic Kr will use AMGCL wherever configured.\n');
end
end


function [ok, reportText] = checkAmgclRuntimeReady()
% Verify that compiled AMGCL MEX functions are visible and callable.

ok = false;
report = strings(0, 1);

amgclMex = which('amgcl_matlab');
amgclBlockMex = which('amgcl_matlab_block');
report(end+1) = "amgcl_matlab: " + stringOrMissing(amgclMex);
report(end+1) = "amgcl_matlab_block: " + stringOrMissing(amgclBlockMex);

if isempty(amgclMex)
    report(end+1) = "AMGCL check failed: amgcl_matlab is not on the MATLAB path.";
    reportText = strjoin(report, newline);
    return
end

try
    A = sparse(gallery('poisson', 8));
    b = ones(size(A, 1), 1);
    [x, reportedResidual, iterations] = callAMGCL(A, b, ...
        'preconditioner', 'amg', ...
        'coarsening', 'aggregation', ...
        'relaxation', 'spai0', ...
        'solver', 'bicgstab', ...
        'tolerance', 1e-8, ...
        'maxIterations', 1000, ...
        'verbose', false);
    relResidual = norm(A*x - b) ./ max(norm(b), realmin);
    report(end+1) = sprintf("AMGCL smoke solve relative residual: %.3e", relResidual);
    report(end+1) = sprintf("AMGCL reported residual: %.3e, iterations: %d", ...
        reportedResidual, iterations);
    ok = isfinite(relResidual) && relResidual < 1e-6;
    if ~ok
        report(end+1) = "AMGCL check failed: smoke-solve residual is too large.";
    end
catch err
    report(end+1) = "AMGCL check failed with error: " + string(err.message);
end

reportText = strjoin(report, newline);
end


function out = stringOrMissing(value)
if isempty(value)
    out = "<missing>";
else
    out = string(value);
end
end


function materializePatchedUpscalingBundle(zipFile, outputRoot)
% Extract the original upscaling bundle and patch old helper assumptions.

marker = fullfile(outputRoot, '.codex_dyn_patch_v19');
if exist(marker, 'file') == 2
    return
end
if exist(outputRoot, 'dir') == 7
    rmdir(outputRoot, 's');
end
ensureFolder(outputRoot);
unzip(zipFile, outputRoot);

patchTextFile(fullfile(outputRoot, 'upscaling', 'dynamic3Drun.m'), { ...
    "model.PVTPropertyFunctions = model.PVTPropertyFunctions.setStateFunction('PoreVolume', MyPvMult(model));", ...
    "model.PVTPropertyFunctions = model.PVTPropertyFunctions.setStateFunction('PoreVolume', BlackOilPoreVolume(model));"; ...
    "model.AutoDiffBackend = DiagonalAutoDiffBackend('useMex', true, ...", ...
    "useMexBackend = true; if isfield(opt, 'dyn_use_mex'), useMexBackend = opt.dyn_use_mex; end" + newline + ...
    "    model.AutoDiffBackend = DiagonalAutoDiffBackend('useMex', useMexBackend, ..." ...
    ; ...
    "nls.useRelaxation = false;" + newline + ...
    "    nls.useLinesearch = true;" + newline + ...
    "    nls.maxIterations = 10;" + newline + ...
    "    nls.maxTimestepCuts = 14;", ...
    "if isfield(opt, 'dyn_3d_use_relaxation'), nls.useRelaxation = opt.dyn_3d_use_relaxation; else, nls.useRelaxation = false; end" + newline + ...
    "    if isfield(opt, 'dyn_3d_use_linesearch'), nls.useLinesearch = opt.dyn_3d_use_linesearch; else, nls.useLinesearch = true; end" + newline + ...
    "    if isfield(opt, 'dyn_3d_max_iterations') && isfinite(opt.dyn_3d_max_iterations), nls.maxIterations = opt.dyn_3d_max_iterations; else, nls.maxIterations = 10; end" + newline + ...
    "    if isfield(opt, 'dyn_3d_max_timestep_cuts') && isfinite(opt.dyn_3d_max_timestep_cuts), nls.maxTimestepCuts = opt.dyn_3d_max_timestep_cuts; else, nls.maxTimestepCuts = 14; end" ...
    ; ...
    "assert(sum(timesteps)== tsim, 'sum of timesteps must equal simTime')", ...
    "assert(abs(sum(timesteps) - tsim) <= max(1, abs(tsim))*1e-10, 'sum of timesteps must equal simTime')" ...
    ; ...
    "nls = getNonLinearSolver(model, 'TimestepStrategy', 'iteration', 'useCPR', true);" + newline + ...
    "    nls.LinearSolver.maxIterations = 50;", ...
    "nls = getNonLinearSolver(model, 'TimestepStrategy', 'iteration', 'useCPR', true);" + newline + ...
    "    dynLinearSolverPolicy = 'amgcl_auto';" + newline + ...
    "    if isfield(opt, 'dyn_linear_solver') && ~isempty(opt.dyn_linear_solver)" + newline + ...
    "        dynLinearSolverPolicy = lower(char(opt.dyn_linear_solver));" + newline + ...
    "    end" + newline + ...
    "    if any(strcmpi(dynLinearSolverPolicy, {'amgcl_auto', 'amgcl_require'}))" + newline + ...
    "        try" + newline + ...
    "            nls.LinearSolver = AMGCL_CPRSolverAD('maxIterations', 50, 'tolerance', 1e-6);" + newline + ...
    "            nls.LinearSolver.setCoarsening('aggregation');" + newline + ...
    "            nls.LinearSolver.setRelaxation('spai0');" + newline + ...
    "            nls.LinearSolver.setSRelaxation('ilu0');" + newline + ...
    "            nls.LinearSolver.setSolver('bicgstab');" + newline + ...
    "            fprintf('    3D dynamic linear solver: AMGCL_CPRSolverAD\n');" + newline + ...
    "        catch amgclErr" + newline + ...
    "            if strcmpi(dynLinearSolverPolicy, 'amgcl_require')" + newline + ...
    "                error('AMGCL:RequiredUnavailable', 'AMGCL is required but unavailable: %s', amgclErr.message);" + newline + ...
    "            end" + newline + ...
    "            warning('AMGCL:UnavailableFallback', 'AMGCL unavailable; falling back to MRST default linear solver: %s', amgclErr.message);" + newline + ...
    "            nls.LinearSolver.maxIterations = 50;" + newline + ...
    "        end" + newline + ...
    "    else" + newline + ...
    "        nls.LinearSolver.maxIterations = 50;" + newline + ...
    "    end" ...
    ; ...
    "N = 2;" + newline + ...
    "    maxNumCompThreads(N);" + newline + ...
    "    nls.LinearSolver.amgcl_setup.nthreads = N;                                  % Specify threads manually", ...
    "if isfield(opt, 'dyn_3d_num_threads') && isfinite(opt.dyn_3d_num_threads)" + newline + ...
    "        N = max(1, round(opt.dyn_3d_num_threads));" + newline + ...
    "    else" + newline + ...
    "        N = 2;" + newline + ...
    "    end" + newline + ...
    "    maxNumCompThreads(N);" + newline + ...
    "    try" + newline + ...
    "        nls.LinearSolver.amgcl_setup.nthreads = N;                                  % Specify threads manually when supported" + newline + ...
    "    catch" + newline + ...
    "    end" ...
    });

patchTextFile(fullfile(outputRoot, 'upscaling', 'dynamicBLrun.m'), { ...
    "solver_type = 'ad-bo';   % 'ad-bo' or 'incompr'", ...
    "solver_type = 'ad-bo';   % 'ad-bo' or 'incompr'" + newline + ...
    "        if isfield(opt, 'dyn_1d_method') && strcmpi(opt.dyn_1d_method, 'transport')" + newline + ...
    "            [s1D, vpar, G1D] = dynamicBLrunTransport(G2, rock2, fluid, state0, rate, ts, opt, sg, states_plot);" + newline + ...
    "            return" + newline + ...
    "        end" ...
    ; ...
    "model.PVTPropertyFunctions = model.PVTPropertyFunctions.setStateFunction('PoreVolume', MyPvMult(model));", ...
    "model.PVTPropertyFunctions = model.PVTPropertyFunctions.setStateFunction('PoreVolume', BlackOilPoreVolume(model));"; ...
    "vpar = {3:.2:6, 1:.2:4, swc};", ...
    "if isfield(opt, 'dyn_corey_step') && ~isempty(opt.dyn_corey_step)" + newline + ...
    "            coreyStep = opt.dyn_corey_step;" + newline + ...
    "        else" + newline + ...
    "            coreyStep = 0.2;" + newline + ...
    "        end" + newline + ...
    "        vpar = {3:coreyStep:6, 1:coreyStep:4, swc};" ...
    ; ...
    "%idp = 1:11:ntot;" + newline + ...
    "            tic" + newline + ...
    "            parfor n=1:ntot", ...
    "%idp = 1:11:ntot;" + newline + ...
    "            dynCandidateProgressFile = '';" + newline + ...
    "            if isfield(opt, 'dyn_timing_file') && ~isempty(opt.dyn_timing_file)" + newline + ...
    "                try" + newline + ...
    "                    [dynCandidatePath, dynCandidateName] = fileparts(char(opt.dyn_timing_file));" + newline + ...
    "                    dynCandidateProgressFile = fullfile(dynCandidatePath, [dynCandidateName '_progress.txt']);" + newline + ...
    "                catch" + newline + ...
    "                    dynCandidateProgressFile = '';" + newline + ...
    "                end" + newline + ...
    "            end" + newline + ...
    "            tic" + newline + ...
    "            for n=1:ntot" + newline + ...
    "                candidateTimer = tic;" + newline + ...
    "                fprintf('                1D Corey candidate %d/%d kw=%.6g kg=%.6g swc=%.6g\n', n, ntot, kw_exp(n), kg_exp(n), swc(n));" + newline + ...
    "                if ~isempty(dynCandidateProgressFile)" + newline + ...
    "                    dynCandidateFid = fopen(dynCandidateProgressFile, 'a');" + newline + ...
    "                    if dynCandidateFid >= 0" + newline + ...
    "                        fprintf(dynCandidateFid, '%s start_1d_candidate %d/%d kw=%.6g kg=%.6g swc=%.6g\n', datestr(now, 31), n, ntot, kw_exp(n), kg_exp(n), swc(n));" + newline + ...
    "                        fclose(dynCandidateFid);" + newline + ...
    "                    end" + newline + ...
    "                end" ...
    ; ...
    "s1D{n} = s1D_it;", ...
    "s1D{n} = s1D_it;" + newline + ...
    "                candidateSeconds = toc(candidateTimer);" + newline + ...
    "                if ~isempty(dynCandidateProgressFile)" + newline + ...
    "                    dynCandidateFid = fopen(dynCandidateProgressFile, 'a');" + newline + ...
    "                    if dynCandidateFid >= 0" + newline + ...
    "                        fprintf(dynCandidateFid, '%s end_1d_candidate %d/%d seconds=%.6g\n', datestr(now, 31), n, ntot, candidateSeconds);" + newline + ...
    "                        fclose(dynCandidateFid);" + newline + ...
    "                    end" + newline + ...
    "                end" ...
    ; ...
    "[~, states] = simulateScheduleAD(state01d, model, schedule, 'NonLinearSolver', ..." + newline + ...
    "                                                 [], 'Verbose', false);", ...
    "nls1D = [];" + newline + ...
    "                if isfield(opt, 'dyn_1d_ad_solver') && strcmpi(opt.dyn_1d_ad_solver, 'robust')" + newline + ...
    "                    nls1D = getNonLinearSolver(model, 'TimestepStrategy', 'iteration');" + newline + ...
    "                    nls1D.useRelaxation = false;" + newline + ...
    "                    nls1D.useLinesearch = true;" + newline + ...
    "                    dyn1DLinearSolverPolicy = 'backslash';" + newline + ...
    "                    if isfield(opt, 'dyn_1d_linear_solver') && ~isempty(opt.dyn_1d_linear_solver)" + newline + ...
    "                        dyn1DLinearSolverPolicy = lower(char(opt.dyn_1d_linear_solver));" + newline + ...
    "                    end" + newline + ...
    "                    if any(strcmpi(dyn1DLinearSolverPolicy, {'amgcl_auto', 'amgcl_require'}))" + newline + ...
    "                        try" + newline + ...
    "                            nls1D.LinearSolver = AMGCLSolverAD('maxIterations', 100, 'tolerance', 1e-6);" + newline + ...
    "                            nls1D.LinearSolver.setCoarsening('aggregation');" + newline + ...
    "                            nls1D.LinearSolver.setRelaxation('spai0');" + newline + ...
    "                            nls1D.LinearSolver.setSolver('bicgstab');" + newline + ...
    "                            fprintf('                1D AD linear solver: AMGCLSolverAD\n');" + newline + ...
    "                        catch amgcl1DErr" + newline + ...
    "                            if strcmpi(dyn1DLinearSolverPolicy, 'amgcl_require')" + newline + ...
    "                                error('AMGCL:RequiredUnavailable1D', '1D AMGCL is required but unavailable: %s', amgcl1DErr.message);" + newline + ...
    "                            end" + newline + ...
    "                            warning('AMGCL:UnavailableFallback1D', '1D AMGCL unavailable; falling back to MRST default linear solver: %s', amgcl1DErr.message);" + newline + ...
    "                        end" + newline + ...
    "                    end" + newline + ...
    "                    if isfield(opt, 'dyn_nls_max_iterations') && isfinite(opt.dyn_nls_max_iterations)" + newline + ...
    "                        nls1D.maxIterations = opt.dyn_nls_max_iterations;" + newline + ...
    "                    else" + newline + ...
    "                        nls1D.maxIterations = 15;" + newline + ...
    "                    end" + newline + ...
    "                    if isfield(opt, 'dyn_nls_max_timestep_cuts') && isfinite(opt.dyn_nls_max_timestep_cuts)" + newline + ...
    "                        nls1D.maxTimestepCuts = opt.dyn_nls_max_timestep_cuts;" + newline + ...
    "                    else" + newline + ...
    "                        nls1D.maxTimestepCuts = 20;" + newline + ...
    "                    end" + newline + ...
    "                end" + newline + ...
    "                [~, states] = simulateScheduleAD(state01d, model, schedule, 'NonLinearSolver', ..." + newline + ...
    "                                                 nls1D, 'Verbose', false);" ...
    });

writeTextFile(fullfile(outputRoot, 'upscaling', 'dynamicBLrunTransport.m'), strjoin([ ...
"function [s1D, vpar, G1D] = dynamicBLrunTransport(G2, rock2, fluid, state0, rate, ts, opt, sg, ~)", ...
"%DYNAMICBLRUNTRANSPORT Robust 1D Corey-candidate transport matcher.", ...
"% This helper mirrors the candidate-grid logic in dynamicBLrun, but avoids", ...
"% running a nonlinear AD simulation for every 1D candidate. It solves the", ...
"% gas-fraction transport equation with an upwind finite-volume update.", ...
"", ...
"        physDim = max(G2.nodes.coords) - min(G2.nodes.coords);", ...
"        G1D = cartGrid([G2.cartDims(end), 1], [physDim(end), physDim(1)]);", ...
"        G1D = computeGeometry(G1D);", ...
"        rock1D = makeRock(G1D, mean(rock2.perm(rock2.regions.saturation==1,:)), ...", ...
"                               mean(rock2.poro(rock2.regions.saturation==1)));", ...
"        gcNum = prod(G2.cartDims) - prod(G2.cartDims(1:2));", ...
"        rock1D.perm(1) = mean(rock2.perm(gcNum+1:end, end));", ...
"        rock1D.poro(1) = mean(rock2.poro(gcNum+1:end));", ...
"", ...
"        swc = 1 - max(sg);", ...
"        assert(swc >= fluid.krPts.og(1,2) && swc <= fluid.krPts.og(2,2))", ...
"        if isfield(opt, 'dyn_corey_step') && ~isempty(opt.dyn_corey_step)", ...
"            coreyStep = opt.dyn_corey_step;", ...
"        else", ...
"            coreyStep = 0.2;", ...
"        end", ...
"        vparGrid = {3:coreyStep:6, 1:coreyStep:4, swc};", ...
"        ntot = numel(vparGrid{1}) * numel(vparGrid{2}) * numel(vparGrid{3});", ...
"        kw_exp = repmat(vparGrid{1}', ntot/numel(vparGrid{1}), 1);", ...
"        kg_exp = repmat(repelem(vparGrid{2}, numel(vparGrid{1}))', ...", ...
"                        ntot/(numel(vparGrid{1})*numel(vparGrid{2})), 1);", ...
"        swc_exp = repelem(vparGrid{3}, numel(vparGrid{1})*numel(vparGrid{2}))';", ...
"        vpar = [kw_exp, kg_exp, swc_exp];", ...
"", ...
"        muW = dynamicTransportViscosity(fluid, state0, 'muO');", ...
"        muG = dynamicTransportViscosity(fluid, state0, 'muG');", ...
"        pv = max(G1D.cells.volumes(:) .* rock1D.poro(:), realmin);", ...
"        rateScale = dynamicTransportOption(opt, 'dyn_transport_rate_scale', NaN);", ...
"        if isfinite(rateScale)", ...
"            q = abs(rate) * rateScale;", ...
"        else", ...
"            q = abs(rate);", ...
"        end", ...
"        if ~(isfinite(q) && q > 0)", ...
"            q = sum(pv) / day();", ...
"        end", ...
"        ts = ts(:)';", ...
"        s1D = cell(ntot, 1);", ...
"        progressFile = dynamicTransportProgressFile(opt);", ...
"        cfl = dynamicTransportOption(opt, 'dyn_transport_cfl', 0.35);", ...
"        cfl = min(max(cfl, 0.02), 0.95);", ...
"        maxSubstepsPerReport = round(dynamicTransportOption(opt, ...", ...
"            'dyn_transport_max_substeps_per_report', 400));", ...
"        maxSubstepsPerReport = max(1, maxSubstepsPerReport);", ...
"", ...
"        for n = 1:ntot", ...
"            candidateTimer = tic;", ...
"            appendDynamicTransportProgress(progressFile, sprintf(...", ...
"                'start_1d_candidate %d/%d kw=%.6g kg=%.6g swc=%.6g', ...", ...
"                n, ntot, kw_exp(n), kg_exp(n), swc_exp(n)));", ...
"            sMax = max(1 - swc_exp(n), eps);", ...
"            s = zeros(G1D.cells.num, 1);", ...
"            sHistory = zeros(G1D.cells.num, numel(ts));", ...
"            tPrevious = 0;", ...
"            for k = 1:numel(ts)", ...
"                dtTarget = max(ts(k) - tPrevious, 0);", ...
"                dtCfl = cfl * min(pv) / max(q, realmin);", ...
"                nSub = max(1, ceil(dtTarget / max(dtCfl, realmin)));", ...
"                if nSub > maxSubstepsPerReport", ...
"                    nSub = maxSubstepsPerReport;", ...
"                end", ...
"                dt = dtTarget / nSub;", ...
"                for isub = 1:nSub", ...
"                    fg = dynamicTransportFractionalGas(s, sMax, kw_exp(n), kg_exp(n), muW, muG);", ...
"                    inflow = [q; q .* fg(1:end-1)];", ...
"                    outflow = q .* fg;", ...
"                    s = s + dt .* (inflow - outflow) ./ pv;", ...
"                    s = min(max(s, 0), sMax);", ...
"                end", ...
"                sHistory(:, k) = flipud(s);", ...
"                tPrevious = ts(k);", ...
"            end", ...
"            s1D{n} = sHistory;", ...
"            appendDynamicTransportProgress(progressFile, sprintf(...", ...
"                'end_1d_candidate %d/%d seconds=%.6g', n, ntot, toc(candidateTimer)));", ...
"        end", ...
"end", ...
"", ...
"function fg = dynamicTransportFractionalGas(s, sMax, kwExp, kgExp, muW, muG)", ...
"        sn = min(max(s ./ max(sMax, eps), 0), 1);", ...
"        krg = sn .^ kgExp;", ...
"        krw = (1 - sn) .^ kwExp;", ...
"        lambdaG = krg ./ max(muG, realmin);", ...
"        lambdaW = krw ./ max(muW, realmin);", ...
"        fg = lambdaG ./ max(lambdaG + lambdaW, realmin);", ...
"end", ...
"", ...
"function mu = dynamicTransportViscosity(fluid, state0, fieldName)", ...
"        mu = 1;", ...
"        try", ...
"            f = fluid.(fieldName);", ...
"            p = mean(state0.pressure(:), 'omitnan');", ...
"            muValue = f(p);", ...
"            mu = mean(muValue(:), 'omitnan');", ...
"        catch", ...
"            mu = 1;", ...
"        end", ...
"        if ~(isfinite(mu) && mu > 0)", ...
"            mu = 1;", ...
"        end", ...
"end", ...
"", ...
"function value = dynamicTransportOption(opt, fieldName, defaultValue)", ...
"        if isfield(opt, fieldName) && ~isempty(opt.(fieldName)) && isfinite(opt.(fieldName))", ...
"            value = opt.(fieldName);", ...
"        else", ...
"            value = defaultValue;", ...
"        end", ...
"end", ...
"", ...
"function progressFile = dynamicTransportProgressFile(opt)", ...
"        progressFile = '';", ...
"        if isfield(opt, 'dyn_timing_file') && ~isempty(opt.dyn_timing_file)", ...
"            try", ...
"                [progressPath, progressName] = fileparts(char(opt.dyn_timing_file));", ...
"                progressFile = fullfile(progressPath, [progressName '_progress.txt']);", ...
"            catch", ...
"                progressFile = '';", ...
"            end", ...
"        end", ...
"end", ...
"", ...
"function appendDynamicTransportProgress(progressFile, message)", ...
"        if isempty(progressFile)", ...
"            return", ...
"        end", ...
"        fid = fopen(progressFile, 'a');", ...
"        if fid >= 0", ...
"            fprintf(fid, '%s %s\n', datestr(now, 31), message);", ...
"            fclose(fid);", ...
"        end", ...
"end" ...
], newline));

patchTextFile(fullfile(outputRoot, 'upscaling', 'upscaleKrReg.m'), { ...
    "opt.dyn_ispc = 0;           % 0: ignore pc (VL conditions); 1: consider pc", ...
    "if ~isfield(opt, 'dyn_ispc') || isempty(opt.dyn_ispc), opt.dyn_ispc = 0; end           % 0: ignore pc (VL conditions); 1: consider pc"; ...
    "opt.dyn_incomp_run = true;  % true: set very low compressibility for fluids and rock", ...
    "if ~isfield(opt, 'dyn_incomp_run') || isempty(opt.dyn_incomp_run), opt.dyn_incomp_run = true; end  % true: set very low compressibility for fluids and rock"; ...
    "opt.dyn_perm_case = 'sand'; % none (sand + clay), geomean or sand", ...
    "if ~isfield(opt, 'dyn_perm_case') || isempty(opt.dyn_perm_case), opt.dyn_perm_case = 'sand'; end % none (sand + clay), geomean or sand"; ...
    "opt.dyn_mrate = 1;          % PV/day", ...
    "if ~isfield(opt, 'dyn_mrate') || isempty(opt.dyn_mrate), opt.dyn_mrate = 1; end          % PV/day"; ...
    "opt.dyn_tsim_year = 0.33;", ...
    "if ~isfield(opt, 'dyn_tsim_year') || isempty(opt.dyn_tsim_year), opt.dyn_tsim_year = 0.33; end"; ...
    "ts = [[1,2,3,6,9,12,18,24:24:60*24 62*24:48:120*24]*hour opt.dyn_tsim_year*year];    % used in paper 1", ...
    "if isfield(opt, 'dyn_trep') && ~isempty(opt.dyn_trep)" + newline + ...
    "        ts = opt.dyn_trep;" + newline + ...
    "    else" + newline + ...
    "        ts = [[1,2,3,6,9,12,18,24:24:60*24 62*24:48:120*24]*hour opt.dyn_tsim_year*year];    % used in paper 1" + newline + ...
    "    end"; ...
    "[states, G2, rock2, fluid, state0, rate] = dynamic3Drun(G, rock , fluid, ...", ...
    "dynProgressFile = '';" + newline + ...
    "    if isfield(opt, 'dyn_timing_file') && ~isempty(opt.dyn_timing_file)" + newline + ...
    "        try" + newline + ...
    "        dynTimingFile = char(opt.dyn_timing_file);" + newline + ...
    "        [dynProgressPath, dynProgressName] = fileparts(dynTimingFile);" + newline + ...
    "        dynProgressFile = fullfile(dynProgressPath, [dynProgressName '_progress.txt']);" + newline + ...
    "        dynProgressFid = fopen(dynProgressFile, 'a');" + newline + ...
    "        fprintf(dynProgressFid, '%s start_dynamic3D\n', datestr(now, 31));" + newline + ...
    "        fclose(dynProgressFid);" + newline + ...
    "        catch" + newline + ...
    "            dynProgressFile = '';" + newline + ...
    "        end" + newline + ...
    "    end" + newline + ...
    "    dyn3dTimer = tic;" + newline + ...
    "    [states, G2, rock2, fluid, state0, rate] = dynamic3Drun(G, rock , fluid, ..."; ...
    "            ts, fault3D, opt, states_plots);", ...
    "            ts, fault3D, opt, states_plots);" + newline + ...
    "    dyn3dSeconds = toc(dyn3dTimer);" + newline + ...
    "    if isfield(opt, 'dyn_timing_file') && ~isempty(opt.dyn_timing_file)" + newline + ...
    "        save(opt.dyn_timing_file, 'dyn3dSeconds');" + newline + ...
    "    end" + newline + ...
    "    if ~isempty(dynProgressFile)" + newline + ...
    "        dynProgressFid = fopen(dynProgressFile, 'a');" + newline + ...
    "        fprintf(dynProgressFid, '%s end_dynamic3D %.6g\n', datestr(now, 31), dyn3dSeconds);" + newline + ...
    "        fprintf(dynProgressFid, '%s start_dynamic1D_matching\n', datestr(now, 31));" + newline + ...
    "        fclose(dynProgressFid);" + newline + ...
    "    end"; ...
    "[s1D, vpar] = dynamicBLrun(G2, rock2, fluid, state0, rate, ts, opt, sg, states_plots);", ...
    "dyn1dTimer = tic;" + newline + ...
    "    [s1D, vpar] = dynamicBLrun(G2, rock2, fluid, state0, rate, ts, opt, sg, states_plots);" + newline + ...
    "    dyn1dSeconds = toc(dyn1dTimer);" + newline + ...
    "    if ~isempty(dynProgressFile)" + newline + ...
    "        dynProgressFid = fopen(dynProgressFile, 'a');" + newline + ...
    "        fprintf(dynProgressFid, '%s end_dynamic1D_matching %.6g\n', datestr(now, 31), dyn1dSeconds);" + newline + ...
    "        fclose(dynProgressFid);" + newline + ...
    "    end"; ...
    "vpar_fit = vpar(id_min, :);", ...
    "vpar_fit = vpar(id_min, :);" + newline + ...
    "    if isfield(opt, 'dyn_timing_file') && ~isempty(opt.dyn_timing_file)" + newline + ...
    "        save(opt.dyn_timing_file, 'dyn3dSeconds', 'dyn1dSeconds', 'diff_min', 'vpar_fit');" + newline + ...
    "    end"; ...
    "vpar_sort = vpar(id_sort(1:10), :);", ...
    "nSort = min(10, numel(id_sort));" + newline + ...
    "    vpar_sort = vpar(id_sort(1:nSort), :);" ...
    });

fid = fopen(marker, 'w');
fprintf(fid, 'patched dynamic upscaling helpers\n');
fclose(fid);
end


function patchTextFile(filePath, replacements)
% Apply exact string replacements to a MATLAB helper file.

txt = fileread(filePath);
for i = 1:size(replacements, 1)
    old = char(replacements{i, 1});
    new = char(replacements{i, 2});
    assert(contains(txt, old), ...
        'Patch target was not found in %s: %s', filePath, old);
    txt = strrep(txt, old, new);
end
fid = fopen(filePath, 'w');
fprintf(fid, '%s', txt);
fclose(fid);
end


function writeTextFile(filePath, txt)
% Write a text file atomically enough for local runtime helper generation.

fid = fopen(filePath, 'w');
assert(fid >= 0, 'Could not open file for writing: %s', filePath);
cleanupObj = onCleanup(@() fclose(fid));
fprintf(fid, '%s', txt);
delete(cleanupObj);
end


function krOpt = krDynOptions(cfg)
% Return options for paper-style dynamic relative-perm upscaling.

krOpt = struct();
krOpt.sgGrid = cfg.sgGrid(:);
krOpt.mDInM2 = 9.869233e-16;
krOpt.minPoro = 1.0e-4;
krOpt.minPermMD = 1.0e-9;
krOpt.refPermSandSI = 7.60393535652603e-13;
krOpt.refPoroSand = 0.289875;
krOpt.clayPceRmse = 0.2953;
krOpt.clayPceUncertaintyQuantile = 0.5;
krOpt.contactAngleDeg = 30;
krOpt.scalingPermComponent = "kzz";
krOpt.direction = 'z';
krOpt.krMode = 'dyn';
krOpt.pcMode = 'inv-per';
krOpt.sg = 'sandClay';
krOpt.deckFile = cfg.originalDeckFile;
krOpt.smokeCartDims = cfg.smokeCartDims;
krOpt.checkpointDir = cfg.checkpointDir;
krOpt.coreyStep = cfg.coreyStep;
krOpt.pcNval = cfg.pcNval;
krOpt.timestepMode = cfg.timestepMode;
krOpt.useMexBackend = cfg.useMexBackend;
krOpt.nlsMaxIterations = cfg.nlsMaxIterations;
krOpt.nlsMaxTimestepCuts = cfg.nlsMaxTimestepCuts;
krOpt.oneDMatchMethod = cfg.oneDMatchMethod;
krOpt.oneDAdSolver = cfg.oneDAdSolver;
krOpt.linearSolverPolicy = cfg.linearSolverPolicy;
krOpt.oneDLinearSolverPolicy = cfg.oneDLinearSolverPolicy;
krOpt.threeDMaxIterations = cfg.threeDMaxIterations;
krOpt.threeDMaxTimestepCuts = cfg.threeDMaxTimestepCuts;
krOpt.threeDUseLineSearch = cfg.threeDUseLineSearch;
krOpt.threeDUseRelaxation = cfg.threeDUseRelaxation;
krOpt.threeDNumThreads = cfg.threeDNumThreads;
krOpt.transportCfl = cfg.transportCfl;
krOpt.transportMaxSubstepsPerReport = cfg.transportMaxSubstepsPerReport;
krOpt.transportRateScale = cfg.transportRateScale;
krOpt.referenceCurves = readSgofReferenceCurves(cfg.originalDeckFile);
krOpt.pcPrestepMode = cfg.pcPrestepMode;
krOpt.precomputedPcCurveCsv = cfg.precomputedPcCurveCsv;
krOpt.precomputedPcNativeCurveCsv = cfg.precomputedPcNativeCurveCsv;
krOpt.precomputedPcSummaryCsv = cfg.precomputedPcSummaryCsv;
krOpt.precomputedPcTable = table();
krOpt.precomputedPcNativeTable = table();
krOpt.precomputedPcSummaryTable = table();
if krOpt.pcPrestepMode == "precomputed"
    assert(exist(cfg.precomputedPcCurveCsv, 'file') == 2, ...
        'Precomputed Pc curve CSV not found: %s', cfg.precomputedPcCurveCsv);
    assert(exist(cfg.precomputedPcSummaryCsv, 'file') == 2, ...
        'Precomputed Pc summary CSV not found: %s', cfg.precomputedPcSummaryCsv);
    krOpt.precomputedPcTable = readtable(cfg.precomputedPcCurveCsv, ...
        'TextType', 'string');
    if exist(cfg.precomputedPcNativeCurveCsv, 'file') == 2
        krOpt.precomputedPcNativeTable = readtable( ...
            cfg.precomputedPcNativeCurveCsv, 'TextType', 'string');
    else
        warning(['Native precomputed Pc curve CSV not found. ', ...
            'Falling back to fixed-grid Pc curve plus endpoint: %s'], ...
            cfg.precomputedPcNativeCurveCsv);
    end
    krOpt.precomputedPcSummaryTable = readtable(cfg.precomputedPcSummaryCsv, ...
        'TextType', 'string');
end
krOpt.runLogFile = fullfile(cfg.tableDir, ...
    sprintf('dynamic_kr_events_%s.log', cfg.caseToken));
end


function [curveLong, curveSummary, curveMat] = computeKrDynCurves( ...
        replaySummary, krOpt, cfg)
% Compute dynamic Kr curves for every requested replay row.

n = height(replaySummary);
sgGrid = krOpt.sgGrid(:)';
krg = nan(n, numel(sgGrid));
krw = nan(n, numel(sgGrid));
nativeSg = cell(n, 1);
nativeKrg = cell(n, 1);
nativeKrw = cell(n, 1);
summaryRows = cell(n, 54);
longRows = cell(n * numel(sgGrid), 19);
longIdx = 0;
curveCells = cell(n, 1);
outputFiles = cellstr(replaySummary.OutputFile);
windowNames = replaySummary.Window;
caseIds = replaySummary.Level3CaseId;
sliceIds = replaySummary.SliceIndex;
curveIds = replaySummary.ProductionCurveId;
sourceRows = replaySummary.SourceRow;

if cfg.useParallel && n > 1
    ensureParallelPool(cfg.numWorkers);
    parfor i = 1:n
        initializeDynamicKrWorkerRuntime(cfg);
        fprintf('Dynamic Kr row %4d/%4d: case %02d slice %02d %s\n', ...
            i, n, caseIds(i), sliceIds(i), char(windowNames(i)));
        curveCells{i} = loadOrComputeKrDynCurveCheckpoint( ...
            curveIds(i), outputFiles{i}, krOpt, windowNames(i), sourceRows(i));
    end
else
    initializeDynamicKrWorkerRuntime(cfg);
    for i = 1:n
        fprintf('Dynamic Kr row %4d/%4d: case %02d slice %02d %s\n', ...
            i, n, caseIds(i), sliceIds(i), char(windowNames(i)));
        curveCells{i} = loadOrComputeKrDynCurveCheckpoint( ...
            curveIds(i), outputFiles{i}, krOpt, windowNames(i), sourceRows(i));
    end
end

for i = 1:n
    curve = curveCells{i};
    krg(i, :) = curve.krg;
    krw(i, :) = curve.krw;
    [nativeSg{i}, nativeKrg{i}, nativeKrw{i}] = ...
        nativeKrCurveValues(curve);

    summaryRows(i, :) = { ...
        curveIds(i), replaySummary.SourceRow(i), replaySummary.GeologyId(i), ...
        replaySummary.ScenarioName(i), replaySummary.CaseLabel(i), ...
        replaySummary.Level3CaseId(i), replaySummary.Level3CaseName(i), ...
        replaySummary.Window(i), replaySummary.SliceIndex(i), ...
        replaySummary.AssignedState(i), replaySummary.SamplingPool(i), ...
        replaySummary.SelectedSampleIndex(i), replaySummary.ReplaySeed(i), ...
        curve.poreVolume, curve.numCells, curveValue(curve, 'originalCartDimY'), ...
        curveValue(curve, 'runtimeCartDimY'), ...
        char(curveTextValue(curve, 'gridMode', "unknown")), ...
        curve.numRegions, curve.numClayRegions, curve.clayPoreVolumeFraction, ...
        curve.meanLog10KzzMD, curve.medianLog10KzzMD, ...
        curve.p05Log10KzzMD, curve.p95Log10KzzMD, ...
        char(curveTextValue(curve, 'pcPrestepMode', "unknown")), ...
        curve.pcNumPoints, curve.pcMaxSg, ...
        curve.brineCoreyExponent, curve.gasCoreyExponent, ...
        curve.irreducibleWaterSaturation, curve.historyMatchError, ...
        curve.krgAtSg20, curve.krgAtSg50, curve.krgAtSg65, ...
        curve.krwAtSg20, curve.krwAtSg50, curve.krwAtSg65, ...
        curve.krgArea, curve.krwArea, curve.mobileGasThresholdSg, ...
        krOpt.krMode, krOpt.coreyStep, char(krOpt.oneDMatchMethod), ...
        char(krOpt.oneDAdSolver), ...
        curveValue(curve, 'replayLoadSeconds'), ...
        curveValue(curve, 'setupSeconds'), ...
        curveValue(curve, 'pcUpscalingSeconds'), ...
        curveValue(curve, 'dynamicKrTotalSeconds'), ...
        curveValue(curve, 'dynamic3DSeconds'), ...
        curveValue(curve, 'dynamic1DMatchSeconds'), ...
        curveValue(curve, 'postprocessSeconds'), ...
        curveValue(curve, 'checkpointSaveSeconds'), ...
        curveValue(curve, 'totalCurveSeconds')};

    for j = 1:numel(sgGrid)
        longIdx = longIdx + 1;
        longRows(longIdx, :) = { ...
            curveIds(i), replaySummary.SourceRow(i), replaySummary.GeologyId(i), ...
            replaySummary.Level3CaseId(i), replaySummary.Level3CaseName(i), ...
            replaySummary.Window(i), replaySummary.SliceIndex(i), ...
            replaySummary.AssignedState(i), replaySummary.SamplingPool(i), ...
            replaySummary.SelectedSampleIndex(i), replaySummary.ReplaySeed(i), ...
            sgGrid(j), krg(i, j), krw(i, j), 1.0 - sgGrid(j), ...
            curve.brineCoreyExponent, curve.gasCoreyExponent, ...
            curve.historyMatchError, char(krOpt.oneDMatchMethod)};
    end
end

curveSummary = cell2table(summaryRows, 'VariableNames', { ...
    'ProductionCurveId', 'SourceRow', 'GeologyId', 'ScenarioName', ...
    'CaseLabel', 'Level3CaseId', 'Level3CaseName', 'Window', ...
    'SliceIndex', 'AssignedState', 'SamplingPool', 'SelectedSampleIndex', ...
    'ReplaySeed', 'PoreVolume', 'NumCells', 'OriginalCartDimY', ...
    'RuntimeCartDimY', 'GridMode', 'NumRegions', 'NumClayRegions', ...
    'ClayPoreVolumeFraction', 'MeanLog10KzzMD', ...
    'MedianLog10KzzMD', 'P05Log10KzzMD', 'P95Log10KzzMD', ...
    'PcPrestepMode', 'PcNumPoints', 'PcMaxSg', 'BrineCoreyExponent', ...
    'GasCoreyExponent', 'IrreducibleWaterSaturation', ...
    'HistoryMatchError', 'KrgAtSg20', 'KrgAtSg50', 'KrgAtSg65', ...
    'KrwAtSg20', 'KrwAtSg50', 'KrwAtSg65', 'KrgArea', 'KrwArea', ...
    'MobileGasThresholdSg', 'KrMode', 'CoreyExponentStep', ...
    'OneDMatchMethod', 'OneDAdSolver', ...
    'ReplayLoadSeconds', 'SetupSeconds', 'PcUpscalingSeconds', ...
    'DynamicKrTotalSeconds', 'Dynamic3DSeconds', ...
    'Dynamic1DMatchSeconds', 'PostprocessSeconds', ...
    'CheckpointSaveSeconds', 'TotalCurveSeconds'});

curveLong = cell2table(longRows, 'VariableNames', { ...
    'ProductionCurveId', 'SourceRow', 'GeologyId', 'Level3CaseId', ...
    'Level3CaseName', 'Window', 'SliceIndex', 'AssignedState', ...
    'SamplingPool', 'SelectedSampleIndex', 'ReplaySeed', 'Sg', ...
    'Krg', 'Krw', 'Sw', 'BrineCoreyExponent', 'GasCoreyExponent', ...
    'HistoryMatchError', 'OneDMatchMethod'});

curveMat = struct();
curveMat.sgGrid = sgGrid;
curveMat.krg = krg;
curveMat.krw = krw;
curveMat.nativeSg = nativeSg;
curveMat.nativeKrg = nativeKrg;
curveMat.nativeKrw = nativeKrw;
curveMat.summary = curveSummary;
end


function curve = loadOrComputeKrDynCurveCheckpoint( ...
        curveId, outputFile, krOpt, windowName, sourceRow)
% Load a completed curve checkpoint or compute and save it.

checkpointFile = krDynCurveCheckpointFile(krOpt.checkpointDir, curveId);
failureFile = krDynCurveFailureFile(krOpt.checkpointDir, curveId);
if exist(checkpointFile, 'file') == 2
    S = load(checkpointFile, 'curve');
    curve = S.curve;
    if exist(failureFile, 'file') == 2
        delete(failureFile);
    end
    recordCurveEvent(krOpt, curveId, 'checkpoint_hit', checkpointFile);
    return
end

recordCurveEvent(krOpt, curveId, 'start_curve', ...
    sprintf('window=%s replay=%s', char(windowName), char(outputFile)));
loadTimer = tic;
S = load(outputFile, 'replay');
replayLoadSeconds = toc(loadTimer);
recordCurveEvent(krOpt, curveId, 'replay_loaded', ...
    sprintf('seconds=%.6g', replayLoadSeconds));
try
    curve = krDynCurveFromReplay(S.replay, krOpt, windowName, curveId, sourceRow);
catch ME
    failure = struct();
    failure.curveId = curveId;
    failure.sourceRow = sourceRow;
    failure.outputFile = char(outputFile);
    failure.windowName = char(windowName);
    failure.errorIdentifier = ME.identifier;
    failure.errorMessage = ME.message;
    failure.errorReport = getReport(ME, 'extended', 'hyperlinks', 'off');
    failure.failureTime = datetime('now');
    save(failureFile, 'failure');
    recordCurveEvent(krOpt, curveId, 'failed', ...
        sprintf('%s | %s', ME.identifier, ME.message));
    rethrow(ME)
end
curve.replayLoadSeconds = replayLoadSeconds;
tmpFile = checkpointFile + ".tmp_" + string(feature('getpid')) + ".mat";
saveTimer = tic;
save(tmpFile, 'curve', '-v7.3');
curve.checkpointSaveSeconds = toc(saveTimer);
if exist(checkpointFile, 'file') ~= 2
    movefile(tmpFile, checkpointFile);
elseif exist(tmpFile, 'file') == 2
    delete(tmpFile);
end
if exist(failureFile, 'file') == 2
    delete(failureFile);
end
recordCurveEvent(krOpt, curveId, 'checkpoint_saved', checkpointFile);
end


function curve = krDynCurveFromReplay(replay, krOpt, windowName, curveId, sourceRow)
% Compute one dynamic Appendix-C-style Kr curve from replay data.

totalTimer = tic;
setupTimer = tic;
[G, CG, rock, fluid, fault3D, opt, diagnostics] = ...
    buildDynInputsFromReplay(replay, krOpt, windowName);
setupSeconds = toc(setupTimer);
recordCurveEvent(krOpt, curveId, 'setup_complete', ...
    sprintf('seconds=%.6g cells=%d', setupSeconds, diagnostics.numCells));
timingFile = fullfile(krOpt.checkpointDir, ...
    sprintf('dyn_timing_%05d.mat', curveId));
if exist(timingFile, 'file') == 2
    delete(timingFile);
end
opt.dyn_timing_file = timingFile;

pcTimer = tic;
if krOpt.pcPrestepMode == "precomputed"
    fprintf('  Pc pre-step from completed IP Pc table...\n');
    recordCurveEvent(krOpt, curveId, 'pc_start', ...
        sprintf('source=precomputed sourceRow=%d', sourceRow));
    [fluid, sgmax] = initializePcInversesForDynamicKr(fluid, rock, opt);
    [sg, pc] = lookupPrecomputedPcCurve(krOpt, sourceRow);
else
    fprintf('  Pc invasion-percolation pre-step...\n');
    recordCurveEvent(krOpt, curveId, 'pc_start', ...
        sprintf('source=original nval=%d', krOpt.pcNval));
    [pc, sg, fluid, sgmax] = upscalePcReg(G, fluid, rock, opt, false);
end
pcUpscalingSeconds = toc(pcTimer);
recordCurveEvent(krOpt, curveId, 'pc_complete', ...
    sprintf('seconds=%.6g points=%d maxSg=%.6g', ...
    pcUpscalingSeconds, numel(pc), max(sg)));
assert(numel(sg) >= 2 && max(sg) > 0, ...
    'Dynamic Kr requires a non-empty Pc/Sg curve from invasion percolation.');

fprintf('  Dynamic 3D + pseudo-1D Corey matching...\n');
recordCurveEvent(krOpt, curveId, 'dynamic_start', ...
    sprintf('timestepMode=%s coreyStep=%.6g', ...
    char(krOpt.timestepMode), krOpt.coreyStep));
warnStates = [ ...
    warning('off', 'MATLAB:nearlySingularMatrix'), ...
    warning('off', 'MATLAB:singularMatrix')];
cleanupWarnings = onCleanup(@() restoreWarningStates(warnStates));
dynamicTimer = tic;
[krwDyn, krgDyn, sgDyn, vparFit, diffMin] = upscaleKrReg( ...
    G, CG, rock, fluid, fault3D, sg, pc, sgmax, [], [], replay.U, opt, false);
dynamicKrTotalSeconds = toc(dynamicTimer);
clear cleanupWarnings
recordCurveEvent(krOpt, curveId, 'dynamic_complete', ...
    sprintf('seconds=%.6g historyMatchError=%.6g', ...
    dynamicKrTotalSeconds, diffMin));

postTimer = tic;
sgDyn = sgDyn(:);
krgDyn = min(max(krgDyn(:), 0), 1);
krwDyn = min(max(krwDyn(:), 0), 1);
[sgDyn, uniqueIdx] = unique(sgDyn, 'stable');
krgDyn = krgDyn(uniqueIdx);
krwDyn = krwDyn(uniqueIdx);

krg = interp1(sgDyn, krgDyn, krOpt.sgGrid(:), 'linear', 'extrap');
krw = interp1(sgDyn, krwDyn, krOpt.sgGrid(:), 'linear', 'extrap');
krg(krOpt.sgGrid(:) > max(sgDyn)) = krgDyn(end);
krw(krOpt.sgGrid(:) > max(sgDyn)) = krwDyn(end);
krg = min(max(krg(:)', 0), 1);
krw = min(max(krw(:)', 0), 1);

curve = diagnostics;
curve.krg = krg;
curve.krw = krw;
curve.nativeSg = sgDyn(:)';
curve.nativeKrg = krgDyn(:)';
curve.nativeKrw = krwDyn(:)';
curve.pcNumPoints = numel(pc);
curve.pcMaxSg = max(sg);
if krOpt.pcPrestepMode == "precomputed"
    curve.pcPrestepMode = "precomputed";
else
    curve.pcPrestepMode = "original";
end
curve.brineCoreyExponent = vparFit(1);
curve.gasCoreyExponent = vparFit(2);
curve.irreducibleWaterSaturation = vparFit(3);
curve.historyMatchError = diffMin;
curve.krgAtSg20 = interp1(krOpt.sgGrid, krg, 0.20, 'linear', 'extrap');
curve.krgAtSg50 = interp1(krOpt.sgGrid, krg, 0.50, 'linear', 'extrap');
curve.krgAtSg65 = interp1(krOpt.sgGrid, krg, 0.65, 'linear', 'extrap');
curve.krwAtSg20 = interp1(krOpt.sgGrid, krw, 0.20, 'linear', 'extrap');
curve.krwAtSg50 = interp1(krOpt.sgGrid, krw, 0.50, 'linear', 'extrap');
curve.krwAtSg65 = interp1(krOpt.sgGrid, krw, 0.65, 'linear', 'extrap');
curve.krgArea = trapz(krOpt.sgGrid, krg);
curve.krwArea = trapz(krOpt.sgGrid, krw);
idx = find(krg >= 0.01, 1, 'first');
if isempty(idx)
    curve.mobileGasThresholdSg = NaN;
else
    curve.mobileGasThresholdSg = krOpt.sgGrid(idx);
end
curve.setupSeconds = setupSeconds;
curve.pcUpscalingSeconds = pcUpscalingSeconds;
curve.dynamicKrTotalSeconds = dynamicKrTotalSeconds;
curve.dynamic3DSeconds = NaN;
curve.dynamic1DMatchSeconds = NaN;
if exist(timingFile, 'file') == 2
    T = load(timingFile);
    if isfield(T, 'dyn3dSeconds')
        curve.dynamic3DSeconds = T.dyn3dSeconds;
    end
    if isfield(T, 'dyn1dSeconds')
        curve.dynamic1DMatchSeconds = T.dyn1dSeconds;
    end
end
curve.postprocessSeconds = toc(postTimer);
curve.totalCurveSeconds = toc(totalTimer);
recordCurveEvent(krOpt, curveId, 'curve_complete', ...
    sprintf('seconds=%.6g', curve.totalCurveSeconds));
end


function [G, CG, rock, fluid, fault3D, opt, diagnostics] = ...
        buildDynInputsFromReplay(replay, krOpt, windowName)
% Build MRST inputs required by original dynamic upscaling helpers.

grid = replay.Grid;
originalCartDims = double(replay.G.cartDims(:)');
gridMode = "full3d";
if ~isempty(krOpt.smokeCartDims)
    [G, CG, grid] = cropReplayGridForSmoke(replay, grid, krOpt.smokeCartDims);
    gridMode = "smoke_crop";
else
    G = replay.G;
    CG = replay.CG;
end
G = ensureGridCellDim(G);

poroAll = max(grid.poro(:), krOpt.minPoro);
perm = grid.perm;
units = grid.units(:);
isSmear = logical(grid.isSmear(:));
volume = G.cells.volumes(:);
permComponentSI = selectPermComponent(perm, krOpt);
permComponentSI = max(permComponentSI(:), krOpt.minPermMD * krOpt.mDInM2);
log10KzzMD = log10(permComponentSI ./ krOpt.mDInM2);

valid = isfinite(poroAll) & isfinite(permComponentSI) & ...
    isfinite(volume) & volume > 0 & units > 0;
assert(all(valid), 'Dynamic Kr runner expects valid replay cells for all fault-core cells.');

rock = struct();
rock.poro = poroAll;
rock.perm = perm;
rock.regions = struct();
rock.regions.saturation = units;
rock.regions.rocknum = ones(G.cells.num, 1);
rock.regions.rocknum(isSmear) = 2;

fault3D = struct();
fault3D.Grid = struct();
fault3D.Grid.isSmear = isSmear;
fault3D.Grid.perm = perm;
fault3D.Grid.units = units;
fault3D.Perm = replay.PermMD(:)' .* krOpt.mDInM2;

fluid = buildDeckFluidForReplay(G, rock, fault3D, krOpt);

opt = struct();
opt.kr_mode = krOpt.krMode;
opt.pc_mode = krOpt.pcMode;
opt.t = 1;
opt.nval = krOpt.pcNval;
opt.sg = krOpt.sg;
opt.window = char(windowName);
opt.fault = 'predict_3D';
opt.dir = krOpt.direction;
opt.theta = [krOpt.contactAngleDeg, krOpt.contactAngleDeg];
opt.zmax = inferWindowZmax(replay, windowName);
opt.thick = {1, 1};
opt.dyn_corey_step = krOpt.coreyStep;
opt.dyn_use_mex = krOpt.useMexBackend;
opt.dyn_nls_max_iterations = krOpt.nlsMaxIterations;
opt.dyn_nls_max_timestep_cuts = krOpt.nlsMaxTimestepCuts;
opt.dyn_1d_method = char(krOpt.oneDMatchMethod);
opt.dyn_1d_ad_solver = char(krOpt.oneDAdSolver);
opt.dyn_linear_solver = char(krOpt.linearSolverPolicy);
opt.dyn_1d_linear_solver = char(krOpt.oneDLinearSolverPolicy);
opt.dyn_3d_max_iterations = krOpt.threeDMaxIterations;
opt.dyn_3d_max_timestep_cuts = krOpt.threeDMaxTimestepCuts;
opt.dyn_3d_use_linesearch = krOpt.threeDUseLineSearch;
opt.dyn_3d_use_relaxation = krOpt.threeDUseRelaxation;
opt.dyn_3d_num_threads = krOpt.threeDNumThreads;
opt.dyn_transport_cfl = krOpt.transportCfl;
opt.dyn_transport_max_substeps_per_report = ...
    krOpt.transportMaxSubstepsPerReport;
opt.dyn_transport_rate_scale = krOpt.transportRateScale;
opt.dyn_ispc = 0;
opt.dyn_incomp_run = true;
opt.dyn_perm_case = 'sand';
opt.dyn_mrate = 1;
[opt.dyn_tsim_year, opt.dyn_trep] = dynamicTimesteps(krOpt.timestepMode);

poreWeights = poroAll .* volume;
diagnostics = struct();
diagnostics.poreVolume = sum(poreWeights, 'omitnan');
diagnostics.numCells = G.cells.num;
diagnostics.originalCartDimY = originalCartDims(2);
runtimeCartDims = double(G.cartDims(:)');
diagnostics.runtimeCartDimY = runtimeCartDims(min(2, numel(runtimeCartDims)));
diagnostics.gridMode = gridMode;
diagnostics.numRegions = numel(unique(units));
diagnostics.numClayRegions = countClayRegions(units, isSmear);
diagnostics.clayPoreVolumeFraction = ...
    sum(poreWeights(isSmear), 'omitnan') ./ max(sum(poreWeights, 'omitnan'), realmin);
diagnostics.meanLog10KzzMD = mean(log10KzzMD, 'omitnan');
diagnostics.medianLog10KzzMD = median(log10KzzMD, 'omitnan');
diagnostics.p05Log10KzzMD = prctile(log10KzzMD, 5);
diagnostics.p95Log10KzzMD = prctile(log10KzzMD, 95);
end


function fluid = buildDeckFluidForReplay(G, rock, fault3D, krOpt)
% Initialize AD-blackoil fluid and scale Pc curves to replayed material units.

deck = convertDeckUnits(readEclipseDeck(krOpt.deckFile));
deck.REGIONS.ROCKNUM = rock.regions.rocknum;
fluid = initDeckADIFluid(deck);
fluid = assignPvMultCompat(fluid, deck);

idReg = unique(rock.regions.saturation(:))';
pcOGups = cell(1, max(idReg));
fluid.isclay = false(1, numel(idReg));
for n = 1:numel(idReg)
    regId = idReg(n);
    id = rock.regions.saturation == regId;
    isClay = mean(double(fault3D.Grid.isSmear(id)), 'omitnan') >= 0.5;
    fluid.isclay(n) = isClay;
    region = struct();
    region.isClay = isClay;
    region.permSI = mean(selectPermComponent(rock.perm(id, :), krOpt), 'omitnan');
    region.poro = mean(rock.poro(id), 'omitnan');
    [sgRef, pcRefPa] = scaledRegionPc(region, krOpt, fluid);
    pcOGups{regId} = @(sg) interp1(sgRef, pcRefPa, sg, 'linear', 'extrap');
end
fluid.pcOG = pcOGups;
end


function fluid = assignPvMultCompat(fluid, deck)
% Minimal replacement for the archived helper assignPvMult.

rock = deck.PROPS.ROCK;
pRef = rock(1, 1);
cR = rock(1, 2);
fluid.pvMultR = cell(1, 1);
fluid.pvMultR{1} = @(p) max(1.0 + cR .* (p - pRef), 1.0e-8);
end


function [sgRef, pcPa] = scaledRegionPc(region, krOpt, fluid)
% Return scaled Pc curve for one material region using deck rock curves.

if region.isClay
    refSg = krOpt.referenceCurves.clay.sg;
    refPc = fluid.pcOG{2};
    log10KMD = log10(max(region.permSI / krOpt.mDInM2, krOpt.minPermMD));
    pceHgBar = 10.^(-0.1992 * log10KMD + 1.407 - krOpt.clayPceRmse + ...
        krOpt.clayPceUncertaintyQuantile * 2 * krOpt.clayPceRmse);
    pceCo2WaterPa = 1.0e5 * pceHgBar * ...
        abs(cosd(krOpt.contactAngleDeg) * 25 / (cosd(140) * 485));
    entryRefPa = refPc(0.10);
    pcPa = refPc(refSg) .* (pceCo2WaterPa ./ entryRefPa);
else
    refSg = krOpt.referenceCurves.sand.sg;
    refPc = fluid.pcOG{1};
    scale = sqrt((krOpt.refPermSandSI * region.poro) ./ ...
        (krOpt.refPoroSand * region.permSI));
    pcPa = refPc(refSg) .* scale;
end
sgRef = refSg(:);
pcPa = makeStrictlyIncreasing(pcPa(:));
end


function tf = hasPrecomputedPcCurve(krOpt, sourceRow)
% Return true when the completed Pc table contains this replay source row.

tf = false;
if ~isempty(krOpt.precomputedPcNativeTable)
    T = krOpt.precomputedPcNativeTable;
    if ismember('ReplaySourceRow', T.Properties.VariableNames)
        rows = str2double(string(T.ReplaySourceRow));
        tf = any(rows == double(sourceRow));
        if tf
            return
        end
    end
end
if isempty(krOpt.precomputedPcTable)
    return
end
T = krOpt.precomputedPcTable;
if ~ismember('ReplaySourceRow', T.Properties.VariableNames)
    return
end
rows = str2double(string(T.ReplaySourceRow));
tf = any(rows == double(sourceRow));
end


function [sg, pc] = lookupPrecomputedPcCurve(krOpt, sourceRow)
% Extract one completed IP Pc curve by replay source row.

assert(hasPrecomputedPcCurve(krOpt, sourceRow), ...
    'No precomputed Pc curve found for replay SourceRow %d.', sourceRow);

if ~isempty(krOpt.precomputedPcNativeTable)
    Tnative = krOpt.precomputedPcNativeTable;
    if ismember('ReplaySourceRow', Tnative.Properties.VariableNames)
        nativeRows = str2double(string(Tnative.ReplaySourceRow));
        nativeMask = nativeRows == double(sourceRow);
        if any(nativeMask)
            [sg, pc] = extractPrecomputedPcRows(Tnative, nativeMask, sourceRow);
            return
        end
    end
end

T = krOpt.precomputedPcTable;
rows = str2double(string(T.ReplaySourceRow));
mask = rows == double(sourceRow);

[sg, pc] = extractPrecomputedPcRows(T, mask, sourceRow);

endpoint = lookupPrecomputedPcEndpoint(krOpt, sourceRow);
assert(~isempty(endpoint), ...
    'No precomputed Pc endpoint found for replay SourceRow %d. The Pc summary table is required.', ...
    sourceRow);
endpointSg = endpoint(1);
endpointPc = endpoint(2);
keep = sg <= endpointSg + 1.0e-12;
sg = sg(keep);
pc = pc(keep);
if isempty(sg)
    sg = endpointSg;
    pc = endpointPc;
elseif abs(sg(end) - endpointSg) > 1.0e-10
    sg(end + 1, 1) = endpointSg;
    pc(end + 1, 1) = endpointPc;
else
    pc(end) = max(pc(end), endpointPc);
end

[sg, keep] = unique(sg, 'stable');
pc = pc(keep);
pc = makeStrictlyIncreasing(pc(:));
end


function [sg, pc] = extractPrecomputedPcRows(T, mask, sourceRow)
% Extract and clean one precomputed Pc curve from a table mask.

sg = str2double(string(T.GasSaturation(mask)));
pc = str2double(string(T.PcPa(mask)));
valid = isfinite(sg) & isfinite(pc);
sg = sg(valid);
pc = pc(valid);
assert(numel(sg) >= 2, ...
    'Precomputed Pc curve for SourceRow %d has fewer than two valid points.', ...
    sourceRow);

[sg, order] = sort(sg(:));
pc = pc(order);
[sg, keep] = unique(sg, 'stable');
pc = pc(keep);
pc = makeStrictlyIncreasing(pc(:));
end


function endpoint = lookupPrecomputedPcEndpoint(krOpt, sourceRow)
% Return [BulkSgMax, PcMaxPa] for a completed Pc curve when available.

endpoint = [];
S = krOpt.precomputedPcSummaryTable;
requiredColumns = {'ReplaySourceRow', 'BulkSgMax', 'PcMaxPa'};
if isempty(S) || ~all(ismember(requiredColumns, S.Properties.VariableNames))
    return
end
rows = str2double(string(S.ReplaySourceRow));
mask = rows == double(sourceRow);
if ~any(mask)
    return
end
bulkSgMax = str2double(string(S.BulkSgMax(find(mask, 1, 'first'))));
pcMaxPa = str2double(string(S.PcMaxPa(find(mask, 1, 'first'))));
if isfinite(bulkSgMax) && isfinite(pcMaxPa) && bulkSgMax > 0
    endpoint = [bulkSgMax, pcMaxPa];
end
end


function [fluid, sgmax] = initializePcInversesForDynamicKr(fluid, rock, opt)
% Build fluid.pcInv and region sgmax without rerunning Pc upscaling.
%
% The dynamic Kr helper requires inverse Pc functions for each saturation
% region. This mirrors the setup used by the optimized IP Pc workflow, but
% leaves the bulk Pc/Sg curve to the completed precomputed Pc table.

reg = rock.regions.saturation;
idReg = unique(reg);
nreg = numel(idReg);
fluid.pcInv = cell(1, max(reg));
sgmax = zeros(1, nreg);

for n = 1:nreg
    regId = idReg(n);
    if strcmp(opt.sg, 'sandClay')
        if fluid.isclay(n)
            sgmin = fluid.krPts.g(2, 1);
            sgmax(n) = fluid.krPts.g(2, 3);
        else
            sgmin = fluid.krPts.g(1, 1);
            sgmax(n) = fluid.krPts.g(1, 3) - 0.01;
        end
    else
        sgmin = fluid.krPts.g(n, 1);
        sgmax(n) = fluid.krPts.g(n, 3);
    end

    sgvals = linspace(sgmin, sgmax(n), pow2(6)-1)';
    sgvals = [sgvals(1); sgvals(1)+1e-3; sgvals(2:end)];
    pcvals = makeStrictlyIncreasing(fluid.pcOG{regId}(sgvals));
    fluid.pcInv{regId} = @(pcOG) interp1(pcvals, sgvals, pcOG, ...
        'linear', 'extrap');
end
end


function [tsimYear, trep] = dynamicTimesteps(mode)
% Return paper or smoke-test report times for dynamic matching.

switch lower(string(mode))
    case "smoke"
        trep = [1, 3, 7] * day;
        tsimYear = trep(end) / year;
    case "paper"
        tsimYear = 0.33;
        trep = [];
    case "paper_fine"
        tsimYear = 0.33;
        trep = unique([[1,2,3,6,9,12,18,24:12:60*24, ...
            61*24:24:120*24] * hour, tsimYear * year]);
    otherwise
        error('Unknown KR_DYN_TIMESTEP_MODE: %s', mode);
end
end


function zmax = inferWindowZmax(replay, windowName)
% Provide depth metadata needed for initial hydrostatic pressure.

if isfield(replay, 'WindowOpt') && isfield(replay.WindowOpt, 'zmax')
    zmax = replay.WindowOpt.zmax;
    return
end
z = replay.G.cells.centroids(:, 3);
topDepth = max(abs(z)) + 1500;
zmax = {[topDepth, topDepth - 1], [topDepth, topDepth - 1]};
if startsWith(lower(string(windowName)), "famp")
    w = str2double(extractAfter(lower(string(windowName)), "famp"));
    baseTop = 1950 - 100 * (w - 1);
    zmax = {[baseTop, baseTop - 50], [baseTop, baseTop - 50]};
end
end


function n = countClayRegions(units, isSmear)
% Count material units dominated by clay-smear cells.

idReg = unique(units(:))';
n = 0;
for r = idReg
    n = n + (mean(double(isSmear(units == r)), 'omitnan') >= 0.5);
end
end


function [Gout, CGout, gridOut] = cropReplayGridForSmoke(replay, gridIn, cropDims)
% Build a small replay-derived Cartesian subgrid for dynamic smoke testing.
%
% This intentionally changes the geometry and is only for plumbing tests.
% Production runs should leave KR_DYN_SMOKE_CARTDIMS unset.

origDims = double(replay.G.cartDims(:)');
cropDims = double(cropDims(:)');
assert(numel(cropDims) == 3, ...
    'KR_DYN_SMOKE_CARTDIMS must contain exactly three integers.');
assert(all(cropDims >= 2) && all(cropDims <= origDims), ...
    'KR_DYN_SMOKE_CARTDIMS must be >=2 and no larger than replay.G.cartDims.');

idx = cell(1, 3);
for d = 1:3
    startIdx = floor((origDims(d) - cropDims(d)) / 2) + 1;
    idx{d} = startIdx:(startIdx + cropDims(d) - 1);
end

extent = max(replay.G.nodes.coords, [], 1) - min(replay.G.nodes.coords, [], 1);
cellSize = extent ./ origDims;
physDims = cellSize .* cropDims;
Gout = cartGrid(cropDims, physDims);
Gout = computeGeometry(Gout);
Gout = ensureGridCellDim(Gout);
CGout = generateCoarseGrid(Gout, ones(Gout.cells.num, 1));
CGout = coarsenGeometry(CGout);

gridOut = struct();
names = fieldnames(gridIn);
for i = 1:numel(names)
    values = gridIn.(names{i});
    gridOut.(names{i}) = cropReplayValues(values, origDims, idx);
end
fprintf('  Smoke crop: replay grid [%d %d %d] -> [%d %d %d] cells.\n', ...
    origDims, cropDims);
end


function G = ensureGridCellDim(G)
% Add the cellDim field expected by the archived dynamic-upscaling helper.

if isfield(G, 'cellDim')
    return
end
dims = double(G.cartDims(:)');
extent = max(G.nodes.coords, [], 1) - min(G.nodes.coords, [], 1);
G.cellDim = extent ./ dims;
end


function out = cropReplayValues(values, origDims, idx)
% Crop a cell-wise replay vector/matrix using Cartesian cell indices.

if size(values, 1) ~= prod(origDims)
    out = values;
    return
end
mask = false(origDims);
mask(idx{1}, idx{2}, idx{3}) = true;
out = values(mask(:), :);
if size(values, 2) == 1
    out = out(:);
end
end


function curves = readSgofReferenceCurves(deckFile)
% Read the first two SGOF tables from the deck as sand and clay references.

txt = fileread(deckFile);
lines = regexp(txt, '\r\n|\n|\r', 'split')';
start = find(strcmp(strtrim(lines), 'SGOF'), 1, 'first') + 1;
tables = {};
current = [];
for i = start:numel(lines)
    line = stripComments(lines{i});
    if strlength(strtrim(line)) == 0
        continue
    end
    if contains(line, '/')
        line = erase(line, '/');
        vals = sscanf(line, '%f')';
        if ~isempty(vals)
            current = [current; vals]; %#ok<AGROW>
        end
        tables{end + 1} = current; %#ok<AGROW>
        current = [];
        if numel(tables) >= 2
            break
        end
    else
        vals = sscanf(line, '%f')';
        if ~isempty(vals)
            current = [current; vals]; %#ok<AGROW>
        end
    end
end
assert(numel(tables) >= 2, 'Could not read two SGOF tables from %s.', deckFile);
curves = struct();
curves.sand = sgofTableToCurve(tables{1});
curves.clay = sgofTableToCurve(tables{2});
end


function line = stripComments(line)
% Remove Eclipse-style inline comments.

idx = strfind(line, '--');
if ~isempty(idx)
    line = extractBefore(string(line), idx(1));
else
    line = string(line);
end
end


function curve = sgofTableToCurve(T)
% Convert one SGOF numeric table to a curve struct.

curve = struct();
curve.sg = T(:, 1);
curve.krg = T(:, 2);
curve.krog = T(:, 3);
curve.pcPa = T(:, 4) * 1.0e5;
end


function pcPa = makeStrictlyIncreasing(pcPa)
% Add tiny monotonic increments so inverse interpolation remains stable.

pcPa = pcPa(:);
for i = 2:numel(pcPa)
    if pcPa(i) <= pcPa(i-1)
        pcPa(i) = pcPa(i-1) + max(abs(pcPa(i-1)) * 1e-9, 1e-6);
    end
end
end


function kSI = selectPermComponent(perm, krOpt)
% Select the permeability component used for Pc scaling.

switch lower(string(krOpt.scalingPermComponent))
    case "kxx"
        col = 1;
    case "kyy"
        if size(perm, 2) >= 4
            col = 4;
        elseif size(perm, 2) >= 2
            col = 2;
        else
            col = 1;
        end
    case "kzz"
        if size(perm, 2) >= 6
            col = 6;
        elseif size(perm, 2) >= 3
            col = 3;
        else
            col = 1;
        end
    otherwise
        error('Unknown permeability component: %s', krOpt.scalingPermComponent);
end
kSI = perm(:, col);
end


function checkpointFile = krDynCurveCheckpointFile(checkpointDir, curveId)
% Return checkpoint file path for one production curve id.

checkpointFile = fullfile(checkpointDir, sprintf('kr_dyn_curve_%05d.mat', curveId));
end


function failureFile = krDynCurveFailureFile(checkpointDir, curveId)
% Return failure checkpoint path for one production curve id.

failureFile = fullfile(checkpointDir, sprintf('kr_dyn_curve_%05d_failure.mat', curveId));
end


function ensureParallelPool(numWorkers)
% Start a process pool for independent Kr curve jobs if needed.

pool = gcp('nocreate');
if isempty(pool)
    parpool('Processes', numWorkers);
elseif pool.NumWorkers ~= numWorkers
    warning('Existing parallel pool has %d workers; requested %d.', ...
        pool.NumWorkers, numWorkers);
end
end


function order = windowOrder(windowNames)
% Convert fampN labels to numeric sort order.

windowNames = string(windowNames);
order = nan(numel(windowNames), 1);
for i = 1:numel(windowNames)
    tok = regexp(windowNames(i), 'famp(\d+)', 'tokens', 'once');
    if ~isempty(tok)
        order(i) = str2double(tok{1});
    end
end
end


function value = curveValue(curve, fieldName)
% Return a scalar curve field or NaN for older checkpoints.

if isfield(curve, fieldName)
    value = curve.(fieldName);
else
    value = NaN;
end
end


function value = curveTextValue(curve, fieldName, defaultValue)
% Return a text curve field or a default for older checkpoints.

if isfield(curve, fieldName)
    value = string(curve.(fieldName));
else
    value = string(defaultValue);
end
end


function mode = canonicalSelectionMode(rawMode)
% Return the canonical Kr row-selection mode.

mode = lower(strtrim(string(rawMode)));
if mode == "median_swi"
    warning('KR_DYN_SELECTION_MODE=median_swi is deprecated; using swi_medoid.');
    mode = "swi_medoid";
end
end


function value = envOrDefault(name, defaultValue)
% Read an environment variable or return a default value.

raw = getenv(char(name));
if isempty(raw)
    value = defaultValue;
else
    value = raw;
end
end


function ids = parseIdList(textValue)
% Parse comma-separated integer ids from a string.

parts = regexp(char(textValue), '\s*,\s*', 'split');
ids = str2double(parts);
ids = ids(isfinite(ids));
assert(~isempty(ids), 'No valid integer ids were provided.');
end


function token = caseTokenFromIds(caseIds)
% Build a stable case-token string such as cases_01_03_04_07.

parts = strings(1, numel(caseIds));
for i = 1:numel(caseIds)
    parts(i) = sprintf('%02d', caseIds(i));
end
token = "cases_" + strjoin(parts, "_");
end


function rootPath = defaultWorkflowRoot()
% Return the default workflow root for the current platform.

if ispc
    rootPath = fullfile('D:', 'codex_gom', 'UQ_workflow');
else
    rootPath = fullfile('/home', 'shaowen', 'orcd', 'scratch', ...
        'predict_shaowen', 'runs', 'manual');
end
end


function rootPath = defaultReplayRoot()
% Return the default replay root for the current platform.

rootPath = fullfile(defaultWorkflowRoot(), 'full87_replay_median_examples');
end


function inputPath = defaultPermeabilityInput()
% Return the canonical Level 3 permeability array for reservoir export.

if ispc
    inputPath = fullfile('D:', 'codex_gom', 'UQ_workflow', ...
        'texas_offshore_field_sampling', ...
        'texas_field_sampling_compact.mat');
else
    projectRoot = strtrim(string(getenv('PROJECT_ROOT')));
    if projectRoot == ""
        projectRoot = fullfile('/home', 'shaowen', 'orcd', 'pool', ...
            'predict_shaowen');
    end
    inputPath = fullfile(projectRoot, 'inputs', ...
        'texas_offshore_field_sampling', ...
        'texas_field_sampling_compact.mat');
end
end


function rootPath = defaultUpscalingRoot()
% Return a platform-appropriate temporary upscaling-code root.

if ispc
    rootPath = fullfile('D:', 'codex_gom', 'tmp_upscaling_dyn_runtime');
else
    rootPath = fullfile(tempdir, 'tmp_upscaling_dyn_runtime');
end
end


function rootPath = defaultMrstRoot()
% Return the default MRST root for the current platform.

if ispc
    rootPath = fullfile('C:', 'Users', 'Shaow', 'OneDrive', 'MIT', ...
        'mrst-2025a', 'SINTEF-AppliedCompSci-MRST-75749fa');
else
    rootPath = fullfile('/home', 'shaowen', 'orcd', 'pool', ...
        'predict_shaowen', 'software', 'mrst-current');
end
end


function value = parseLogicalEnv(name, defaultValue)
% Parse a boolean environment variable.

txt = lower(strtrim(string(getenv(char(name)))));
if txt == ""
    value = defaultValue;
elseif any(txt == ["1", "true", "yes", "on"])
    value = true;
elseif any(txt == ["0", "false", "no", "off"])
    value = false;
else
    error('Environment variable %s must be boolean-like, got: %s', name, txt);
end
end


function value = parseNumericEnv(name, defaultValue)
% Parse a scalar numeric environment variable.

txt = strtrim(string(getenv(char(name))));
if txt == ""
    value = defaultValue;
else
    value = str2double(txt);
    assert(isfinite(value), 'Environment variable %s must be numeric.', name);
end
end


function values = parseIntegerListEnv(name)
% Parse a comma/space separated integer-list environment variable.

txt = strtrim(string(getenv(char(name))));
if txt == ""
    values = [];
    return
end
parts = regexp(txt, '[,\s]+', 'split');
values = str2double(parts);
assert(all(isfinite(values)) && all(values == round(values)), ...
    'Environment variable %s must contain integer row ids.', name);
values = values(:);
end


function ensureFolder(folderPath)
% Create a folder if needed.

if exist(folderPath, 'dir') ~= 7
    mkdir(folderPath);
end
end


function restoreWarningStates(states)
% Restore warning states captured before a noisy numerical solve.

for i = 1:numel(states)
    warning(states(i).state, states(i).identifier);
end
end


function recordCurveEvent(krOpt, curveId, stage, detail)
% Append a lightweight event-log line without affecting numerical results.

if nargin < 4
    detail = "";
end
try
    if ~isfield(krOpt, 'runLogFile') || isempty(krOpt.runLogFile)
        return
    end
    logFile = char(krOpt.runLogFile);
    logDir = fileparts(logFile);
    if exist(logDir, 'dir') ~= 7
        mkdir(logDir);
    end
    fid = fopen(logFile, 'a');
    if fid < 0
        return
    end
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid, '%s curve=%05d stage=%s %s\n', ...
        datestr(now, 31), curveId, char(stage), char(detail));
catch
    % Logging must never change or stop an otherwise valid numerical run.
end
end
