function export_core(planfile,accepted,out,reuse_ref,reuse_model)
plan=jsondecode(fileread(planfile));spec=plan.record.spec;
addpath(spec.numeric_package);[~,cfg]=setup_candidate(out);
F=load(spec.frozen_file,'m','net');R=load(spec.replay_file,'replay');G=R.replay.G;grid=R.replay.Grid;
deck=convertDeckUnits(readEclipseDeck(fullfile(spec.numeric_package,'vendor','reference.DATA')));
assert(isequal(F.m.kr,deck.PROPS.SGOF)&&isequal(F.m.material,1+double(grid.isSmear(:))));
t=tic;[m,audit]=cell_pc_model(F.m,grid.perm,grid.poro,G.cells.volumes,cfg);preparation_seconds=toc(t);
assert(isequal(m.e,F.net.e));
assert(isequal(sort(m.inlet(:)),sort(F.net.ci(:)))&&isequal(sort(m.outlet(:)),sort(F.net.co(:))));
V=load(fullfile(accepted,'fixed100.mat'));pp=V.result(:,1);assert(isequal(pp,plan.tables.fixed100(:,1)));
write_raw(fullfile(out,'raw.bin'),spec.core,m,grid.perm,grid.poro,G.cells.volumes,pp,plan.pressure_dictionary(:),cfg);
write_mip_packet(fullfile(out,'model.bin'),'model',m,pp);
save(fullfile(out,'reference_model.mat'),'m','-v7');
save_json(fullfile(out,'input_audit.json'),struct('audit',audit,'matlab_preparation_seconds',preparation_seconds, ...
    'core',spec.core,'no_flow_solves',true,'raw_gridblocks_exported',true));
clear F R G grid
if ~isempty(reuse_ref)
    assert(spec.core==25&&~isempty(reuse_model));B=load(reuse_model,'m','expected');
    for name={'e','inlet','outlet','material','scale','entry','endpoint_pressure','pv','kr'}
        assert(isequal(m.(name{1}),B.m.(name{1})),'JuliaPc:Reuse','Existing reference model differs');
    end
    assert(isequal(pp,B.expected(:,1)));clear B
end
stats=export_reference(m,pp,out,reuse_ref);
assert(max(abs(stats(:,2:6)-V.result(:,[2,5:8])),[],'all')<=2e-12,'JuliaPc:Reference','Accepted physical summaries changed');
end
