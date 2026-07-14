function [selectedReplay, selection] = select_swi_medoid_replay_rows( ...
        replaySummary, pcSummary, caseIds, windows)
%SELECT_SWI_MEDOID_REPLAY_ROWS Select one actual Swi medoid per window.
%
%   [SELECTEDREPLAY, SELECTION] = SELECT_SWI_MEDOID_REPLAY_ROWS(
%   REPLAYSUMMARY, PCSUMMARY, CASEIDS, WINDOWS) joins replay rows to their
%   upscaled Pc endpoints and selects the observed effective-Swi value
%   minimizing total absolute distance within every case/window.
%
%   In one dimension this medoid is an observation nearest the sample
%   median. With the production count of 87 slices, the median is itself an
%   observed value. Ties are resolved by replay SourceRow and SliceIndex so
%   reruns select exactly the same PREDICT realization.

requiredReplay = {'SourceRow', 'ProductionCurveId', 'GeologyId', ...
    'Level3CaseId', 'Level3CaseName', 'Window', 'WindowOrder', ...
    'SliceIndex', 'SelectedSampleIndex', 'ReplaySeed'};
requiredPc = {'ReplaySourceRow', 'GeologyId', 'Level3CaseId', ...
    'Level3CaseName', 'Window', 'SliceIndex', 'BulkSgMax'};
requireColumns(replaySummary, requiredReplay, 'replay summary');
requireColumns(pcSummary, requiredPc, 'Pc summary');

caseIds = double(caseIds(:)');
windows = string(windows(:)');
replaySummary.GeologyId = string(replaySummary.GeologyId);
replaySummary.Window = string(replaySummary.Window);
replaySummary.Level3CaseId = numericColumn(replaySummary.Level3CaseId);
pcSummary.GeologyId = string(pcSummary.GeologyId);
pcSummary.Window = string(pcSummary.Window);
pcSummary.Level3CaseId = numericColumn(pcSummary.Level3CaseId);

if ismember('EffectiveSwi', pcSummary.Properties.VariableNames)
    pcSwi = numericColumn(pcSummary.EffectiveSwi);
else
    pcSwi = 1.0 - numericColumn(pcSummary.BulkSgMax);
end
pcSourceRows = numericColumn(pcSummary.ReplaySourceRow);

selectedMask = false(height(replaySummary), 1);
selectionRows = cell(numel(caseIds) * numel(windows), 14);
rowId = 0;

for c = 1:numel(caseIds)
    caseId = caseIds(c);
    for w = 1:numel(windows)
        windowName = windows(w);
        candidateIdx = find(replaySummary.Level3CaseId == caseId & ...
            replaySummary.Window == windowName);
        assert(~isempty(candidateIdx), 'SwiMedoid:MissingCandidates', ...
            'No replay candidates found for case %d, window %s.', ...
            caseId, windowName);

        candidateSourceRows = numericColumn( ...
            replaySummary.SourceRow(candidateIdx));
        [found, pcIdx] = ismember(candidateSourceRows, pcSourceRows);
        assert(all(found), 'SwiMedoid:MissingPcRows', ...
            'Pc summary is missing replay rows for case %d, window %s.', ...
            caseId, windowName);
        candidateSwi = pcSwi(pcIdx);
        assert(all(isfinite(candidateSwi)), 'SwiMedoid:InvalidSwi', ...
            'Effective Swi contains non-finite values for case %d, window %s.', ...
            caseId, windowName);

        targetSwi = median(candidateSwi);
        distanceToMedoid = abs(candidateSwi - targetSwi);
        tieTable = table(distanceToMedoid, candidateSourceRows, ...
            numericColumn(replaySummary.SliceIndex(candidateIdx)), ...
            (1:numel(candidateIdx))', 'VariableNames', ...
            {'Distance', 'SourceRow', 'SliceIndex', 'LocalIndex'});
        tieTable = sortrows(tieTable, ...
            {'Distance', 'SourceRow', 'SliceIndex'});
        localIdx = tieTable.LocalIndex(1);
        replayIdx = candidateIdx(localIdx);
        selectedMask(replayIdx) = true;
        selectedPcIdx = pcIdx(localIdx);

        if ismember('CurveId', pcSummary.Properties.VariableNames)
            selectedPcCurveId = numericColumn( ...
                pcSummary.CurveId(selectedPcIdx));
        else
            selectedPcCurveId = NaN;
        end

        rowId = rowId + 1;
        selectionRows(rowId, :) = { ...
            replaySummary.GeologyId(replayIdx), caseId, ...
            replaySummary.Level3CaseName(replayIdx), windowName, ...
            numel(candidateIdx), targetSwi, candidateSwi(localIdx), ...
            distanceToMedoid(localIdx), selectedPcCurveId, ...
            numericColumn(replaySummary.ProductionCurveId(replayIdx)), ...
            numericColumn(replaySummary.SourceRow(replayIdx)), ...
            numericColumn(replaySummary.SliceIndex(replayIdx)), ...
            numericColumn(replaySummary.SelectedSampleIndex(replayIdx)), ...
            numericColumn(replaySummary.ReplaySeed(replayIdx))};
    end
end

selectedReplay = replaySummary(selectedMask, :);
selectedReplay = sortrows(selectedReplay, ...
    {'Level3CaseId', 'WindowOrder', 'SliceIndex'});
selection = cell2table(selectionRows(1:rowId, :), 'VariableNames', { ...
    'GeologyId', 'Level3CaseId', 'Level3CaseName', 'Window', ...
    'CandidateCount', 'SwiMedoidTarget', 'SelectedEffectiveSwi', ...
    'AbsoluteSwiDistance', 'SelectedPcCurveId', ...
    'SelectedProductionCurveId', 'SelectedReplaySourceRow', ...
    'SelectedSliceIndex', 'SelectedSampleIndex', 'SelectedReplaySeed'});
end


function requireColumns(T, required, label)
% Require all named columns in a table.

missing = setdiff(required, T.Properties.VariableNames);
assert(isempty(missing), 'SwiMedoid:MissingColumns', ...
    '%s is missing columns: %s', label, strjoin(missing, ', '));
end


function values = numericColumn(values)
% Convert a table variable to a finite double column.

if ~isnumeric(values)
    values = str2double(string(values));
end
values = double(values(:));
assert(all(isfinite(values)), 'SwiMedoid:NonfiniteColumn', ...
    'Expected a finite numeric column.');
end
