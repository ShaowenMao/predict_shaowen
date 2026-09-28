function check_selection(root,caseId)
% Verify the independent Python selection with the production selector.
T=readtable(fullfile(root,'inputs','replay_template.csv'),'TextType','string');
S=readtable(fullfile(root,'inputs','pc_summary.csv'),'TextType','string');
T.WindowOrder=str2double(extractAfter(T.Window,'famp'));
T=sortrows(T,{'Level3CaseId','SliceIndex','WindowOrder'});
T.ProductionCurveId=(1:height(T))';
[~,M]=select_swi_medoid_replay_rows(T,S,caseId,"famp"+(1:6));
E=readtable(fullfile(root,'inputs','selected.csv'),'TextType','string');
assert(isequal(M.SelectedReplaySourceRow,double(E.case_source_row)));
assert(isequal(M.SelectedSampleIndex,double(E.selected_sample_index)));
assert(isequal(M.SelectedReplaySeed,double(E.exact_replay_seed)));
writetable(M,fullfile(root,'inputs','matlab_selection.csv'));
end
