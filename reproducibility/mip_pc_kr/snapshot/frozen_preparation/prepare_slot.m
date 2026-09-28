function prepare_slot(selection,summaryfile,out,physics,data,mrst)
% Replay the frozen stochastic identity; export geometry, not upscaled Pc/Kr.
assert(~isempty(getenv('SLURM_JOB_ID')),'FullFault:Compute','Compute node required');
root=fileparts(mfilename('fullpath'));addpath(fullfile(root,'vendor'),'-begin');
addpath(fullfile(root,'vendor','pipeline','vendor','pc','validation'),'-begin');
cfg=jsondecode(fileread(fullfile(root,'vendor','config.json')));
assert(~isfolder(out));mkdir(out);maxNumCompThreads(2);
context=prepare_production_replay_batch(string(selection),string(fullfile(out,'replay')), ...
    string(data),string(physics),string(mrst),0.005);
C=readtable(context,'TextType','string');assert(height(C)==1);
S=readtable(selection,'TextType','string');B=jsondecode(fileread(summaryfile));
assert(C.GeologyId==string(S.geology_id)&&C.Window==string(S.window));
assert(C.SelectedSampleIndex==double(S.selected_sample_index)&&C.ReplaySeed==double(S.exact_replay_seed));
R=load(char(C.OutputFile),'replay');G=R.replay.G;grid=R.replay.Grid;
assert(numel(grid.poro)==G.cells.num&&G.cells.num==B.NumCells);
rock=struct('perm',grid.perm,'poro',grid.poro);
net=phase_network_axis(G,rock,3);
deck=convertDeckUnits(readEclipseDeck(fullfile(root,'vendor','reference.DATA')));
m=struct('e',net.e,'inlet',net.ci,'outlet',net.co,'material',1+double(grid.isSmear(:)), ...
    'kr',{deck.PROPS.SGOF});
assert(m.kr{1}(end,1)==.95&&m.kr{2}(end,1)==.70);
pv=sum(G.cells.volumes(:).*grid.poro(:));vol=sum(G.cells.volumes);
assert(abs(pv/B.PoreVolume-1)<1e-10&&abs(vol/B.BulkVolume-1)<1e-10, ...
    'FullFault:PV','Physical grid pore volume changed relative to production');
raw=fullfile(out,'raw_unplanned.bin');
write_raw(raw,B.slot,m,grid.perm,grid.poro,G.cells.volumes,[0;1],B.production_pressure_pa(:),cfg);
fid=fopen(fullfile(out,'geometry_unplanned.bin'),'w','ieee-le');assert(fid>0);
cleanup=onCleanup(@()fclose(fid));
fwrite(fid,int64([2026090906,B.slot,3,net.n,size(net.e,1),numel(net.ci),numel(net.co)]),'int64');
fwrite(fid,[net.length,net.area],'double');fwrite(fid,zeros(32,1,'uint8'),'uint8');
for field={'e','h1','h2','ci','co','hi','ho'}
    v=net.(field{1});if ismember(field{1},{'e','ci','co'}),fwrite(fid,int64(v),'int64');else,fwrite(fid,v,'double');end
end
clear cleanup
report=struct('slot',B.slot,'geology',char(C.GeologyId),'window',char(C.Window), ...
    'sample',C.SelectedSampleIndex,'seed',C.ReplaySeed,'num_cells',G.cells.num, ...
    'physical_pore_volume',pv,'bulk_volume',vol,'porosity',pv/vol, ...
    'architecture_sha256',char(C.DiscreteArchitectureSha256),'max_log10_replay_difference',C.MaxAbsLog10Diff, ...
    'replay_file',char(C.OutputFile),'geometry_only',true,'pc_kr_upscaling_performed',false,'matlab_version',version);
fid=fopen(fullfile(out,'preparation.json'),'w');fprintf(fid,'%s\n',jsonencode(report));fclose(fid);
% Record exactly the external MATLAB functions loaded by this replay.
loaded=inmem('-completenames');fid=fopen(fullfile(out,'loaded_matlab_files.txt'),'w');
for i=1:numel(loaded),if isfile(loaded{i}),fprintf(fid,'%s\n',loaded{i});end,end
fclose(fid);
fprintf('PREPARED slot %d: %d cells, exact sample %d seed %d\n',B.slot,G.cells.num,C.SelectedSampleIndex,C.ReplaySeed);
end
