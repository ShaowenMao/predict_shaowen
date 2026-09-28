function test_raw_export(baseline,out)
assert(~isfolder(out));mkdir(out);D=load(baseline,'reservoirReady');B=D.reservoirReady;
curve=fullfile(out,'synthetic_curve.csv');C=[(0:99)',(0:99)'/110,1-(0:99)'/110,(0:99)'/110,zeros(100,4)];
T=array2table(C,'VariableNames',{'PcPa','Sg','Krw','Krg','Trapped','Immobile','Breakthrough','Invaded'});writetable(T,curve);
audit=fullfile(out,'synthetic_audit.json');fid=fopen(audit,'w');fprintf(fid,'{"synthetic_test_only":true}');fclose(fid);
records=repmat(struct('slot',0,'window',0,'slice',0,'sample',0,'seed',0,'curve','','audit',''),522,1);
for j=1:522
    w=floor((j-1)/87)+1;s=mod(j-1,87)+1;
    records(j)=struct('slot',j,'window',w,'slice',s,'sample',B.predictRealizationIdMap(w,s), ...
        'seed',B.effectivePermeability.exactReplaySeed(w,s),'curve',curve,'audit',audit);
end
M=struct('source','synthetic_test_only','baseline_mat',baseline,'records',records);
manifest=fullfile(out,'manifest.json');fid=fopen(manifest,'w');fprintf(fid,'%s',jsonencode(M));fclose(fid);
assemble_raw(manifest,fullfile(out,'export'));
R=load(fullfile(out,'export','candidate_raw_s03_c022_case12.mat'),'candidateRaw');
assert(isequal(size(R.candidateRaw.rawCurves),[6,87]));
assert(isequaln(R.candidateRaw.Kxx_SI,B.Kxx_SI)&&isequaln(R.candidateRaw.Kyy_SI,B.Kyy_SI)&&isequaln(R.candidateRaw.Kzz_SI,B.Kzz_SI));
assert(~R.candidateRaw.reservoirExportApproved);
assert(~isfield(R,'reservoirReady'));
fprintf('PASS synthetic 522-slot raw MAT export, exact readback, identity and protected schema\n');
end
