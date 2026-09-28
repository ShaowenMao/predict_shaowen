function baseline_export(matfile,assignments,out)
% Preserve native production curves; no resampling or reservoir time stepping.
assert(~isfolder(out));mkdir(out);D=load(matfile,'reservoirReady');R=D.reservoirReady;
A=readtable(assignments,'TextType','string');assert(height(A)==522);
assert(string(R.geologyId)=="s03_c022"&&R.level3CaseId==12);
assert(isequal(size(R.pcCurves),[6,87])&&isequal(size(R.krCurves),[6,87]));
maps=zeros(522,10);
for slot=1:522
    w=floor((slot-1)/87)+1;s=mod(slot-1,87)+1;
    assert(A.window(slot)=="famp"+w&&A.slice_index(slot)==s);
    assert(R.predictRealizationIdMap(w,s)==A.selected_sample_index(slot));
    P=R.pcCurves{w,s};K=R.krCurves{w,s};
    assert(R.effectivePermeability.exactReplaySeed(w,s)==A.exact_replay_seed(slot));
    folder=fullfile(out,sprintf('slot%03d',slot));mkdir(folder);
    writematrix([P.pcBar(:)*1e5,P.gasSaturation(:),zeros(numel(P.pcBar),2)],fullfile(folder,'pc.csv'));
    writematrix([zeros(numel(K.gasSaturation),1),K.gasSaturation(:),K.krw(:),K.krg(:)],fullfile(folder,'kr.csv'));
    maps(slot,:)=[slot,w,s,R.Kxx_SI(w,s),R.Kyy_SI(w,s),R.Kzz_SI(w,s),R.upscaledPorosity(w,s), ...
        P.bulkSgMax,K.krw(end),K.krg(end)];
end
writematrix(maps,fullfile(out,'maps.csv'));
fprintf('BASELINE: 522 identities and original Pc/Kr curves exported without resampling\n');
end
