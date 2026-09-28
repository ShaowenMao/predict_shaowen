function make_specific(productionSpecific, tableFile, auditFile, output, caseKey, angle, pcConvention)
% Export into a new specific-input file, preserving geology and cell assignment.
assert(~isfile(output));B=load(productionSpecific);D=B;E=load(tableFile);
A=jsondecode(fileread(auditFile));assert(A.cores==522&&A.all_structural_checks_passed);
sgof=D.fault.fluid_tables.SGOF;indices=double(sgof.source_curve_linear_indices(:));
assert(numel(indices)==522&&isequal(sort(indices),(1:522)'));
assert(isequal(size(E.exportTables),[6,87]));n=numel(indices);counts=zeros(n,1);ends=zeros(n,1);
for j=1:n
    idx=indices(j);[w,s]=ind2sub([6,87],idx);slot=(w-1)*87+s;
    assert(E.tableSlotMap(w,s)==slot);
    assert(D.fault.region_matrix(w,s)==sgof.regions(j));
    t=E.exportTables{idx};assert(size(t,2)==4&&all(isfinite(t),'all'));
    assert(all(diff(t(:,1))>0)&&all(diff(t(:,2))>=0)&&all(diff(t(:,3))<=0)&&all(diff(t(:,4))>=0));
    assert(t(1,1)==0&&t(1,2)==0&&t(1,3)==1&&t(1,4)>0&&t(end,3)==0);
    sgof.tables{j}=t;counts(j)=size(t,1);ends(j)=t(end,1);
end
sgof.point_counts=reshape(counts,size(sgof.point_counts));
sgof.bulk_sg_max=reshape(ends,size(sgof.bulk_sg_max));
sgof.effective_swi=reshape(1-ends,size(sgof.effective_swi));
D.fault.fluid_tables.SGOF=sgof;
endmap=cellfun(@(t)t(end,1),E.exportTables);
D.curve_selection=struct('schema','candidate_cell_pc_mip_kr_full_slice_v1', ...
    'pc_representation','full_slice','kr_representation','mip_dip_parallel_full_slice', ...
    'window_names',{B.curve_selection.window_names},'slice_indices',B.curve_selection.slice_indices, ...
    'bulk_sg_max',endmap,'effective_swi',1-endmap, ...
    'endpoint_definition','complement of sampled bulk drainage Sg endpoint; not a new local Swi', ...
    'source_raw_sha256',E.rawSourceSha256,'entry_metadata_sha256',E.entryMetadataSha256, ...
    'representatives_used',0,'independent_upscaled_assignments',522);
D.name=['candidate_' caseKey '_wrr' num2str(angle) '_' pcConvention];
D.created=char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd HH:mm:ss Z'));
D.metadata.pc_representation='full_slice';D.metadata.kr_representation='mip_dip_parallel_full_slice';
D.metadata.representation_detection='explicit candidate export contract';
[~,tableName,tableExt]=fileparts(tableFile);
D.metadata.reservoir_ready_file=[tableName tableExt];
D.metadata.reservoir_ready_schema='candidate_sgof_native_nodes_v1';
D.metadata.sgof_point_count_min=min(counts);D.metadata.sgof_point_count_max=max(counts);
D.metadata.pc_curve_treatment='exact inlet entry at Sg=0; all positive-Sg native Pc nodes retained';
assert(ismember(angle,[30,60]) && E.wrrContactAngleDeg==angle);
assert(strcmp(E.pcConvention,pcConvention));
D.metadata.wrr_contact_angle_deg=angle;
D.curve_selection.pc_convention=pcConvention;
if strcmp(pcConvention,'zero_before_breakthrough')
    D.metadata.pc_curve_treatment='Sg=0 before exact core breakthrough; linear onset to first sampled post-breakthrough point; Kr(Sg) unchanged';
else
    assert(strcmp(pcConvention,'physical_prebreakthrough'));
end
D.metadata.note='Candidate only. No six-core Kr approximation. Explicit endpoint and bounded roundoff policies.';
D.metadata.provenance=struct('original_specific',productionSpecific,'table_file',tableFile, ...
    'source_raw_sha256',E.rawSourceSha256,'entry_metadata_sha256',E.entryMetadataSha256);
if isfield(D.downstream_contract,'validation')
    D.downstream_contract.source_geometry_validation=D.downstream_contract.validation;
    D.downstream_contract=rmfield(D.downstream_contract,'validation');
end
D.candidate_export=struct('schema','candidate_table_export_validation_v1', ...
    'production_promotion',false,'requires_external_validation_receipt',true, ...
    'raw_preserved',true,'table_audit_file',auditFile, ...
    'terminal_extension','constant Pc and Kr values; not a hard Sg cap');
% Field-level protection, including the full rotated permeability tensors.
assert(isequaln(rmfield(D.fault,'fluid_tables'),rmfield(B.fault,'fluid_tables')));
for name={'stratigraphy','window_slice','predict_sample_ids','saturation_regions','geology_hash','geology_link','pairing_key'}
    assert(isequaln(D.(name{1}),B.(name{1})));
end
save(output,'-struct','D','-v7.3');R=load(output);assert(isequaln(D,R));
fid=fopen([output '.roundtrip.json'],'w');assert(fid>0);
fprintf(fid,'%s\n',jsonencode(struct('passed',true,'tables',522,'all_table_values_preserved',true, ...
    'fault_K_and_porosity_unchanged',true,'all_cell_and_region_mappings_unchanged',true, ...
    'stratigraphy_unchanged',true,'production_input_unchanged',isequaln(B,load(productionSpecific)))));fclose(fid);
end
