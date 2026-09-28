function make_specific60(baseline, tables, audit, scenarioFile, output)
% The existing exporter owns numerical conversion; this wrapper adds provenance.
temp=[output '.native.mat'];make_specific(baseline,tables,audit,temp);
D=load(temp);S=jsondecode(fileread(scenarioFile));
assert(S.clay_theta_deg==60&&S.sand_endpoint==0.95&&S.clay_endpoint==0.8);
assert(strcmp(S.wetting_rule,'krw_strictly_positive')&&S.cell_based_pc&&S.flow_axis==3);
D.wettability_scenario=S;D.metadata.wettability_scenario=S;
D.name='isolated_reference_wrr60_all522_qualified_engine';
D.candidate_export.wettability_scenario=S;
assert(~isfile(output));save(output,'-struct','D','-v7.3');R=load(output);
T=load(temp,'fault');assert(isequaln(D,R));assert(isequaln(R.fault,T.fault));
end
