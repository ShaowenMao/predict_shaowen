function prepare_packet(requestfile)
% Keep exact-draw replay unchanged; adapt only the new manifest and Pc tables.
c=jsondecode(fileread(requestfile));maxNumCompThreads(2);
addpath(fullfile(c.code,'matlab'),'-begin');
addpath(fullfile(c.science,'vendor','pipeline','vendor','pc','validation'),'-begin');
cfg=jsondecode(fileread(fullfile(c.science,'CONFIG.json')));cfg=cfg.explicit_local_cell_settings;
sand=readmatrix(fullfile(c.science,'sand_unchanged.csv'));
kr=readmatrix(fullfile(c.science,'clay_kr_unchanged.csv'));
for i=1:numel(c.cores)
    r=c.cores(i);started=tic;
    try
        context=prepare_production_replay_batch(string(r.selection),string(fullfile(r.local,'replay')), ...
            string(c.data),string(c.physics),string(c.mrst),0.005);
        C=readtable(context,'TextType','string');S=readtable(r.selection,'TextType','string');
        assert(height(C)==1&&C.SelectedSampleIndex==S.selected_sample_index&&C.ReplaySeed==S.exact_replay_seed);
        assert(C.GeologyId==S.geology_id&&C.Window==S.window);
        a=load(char(C.OutputFile),'replay');G=a.replay.G;grid=a.replay.Grid;
        % Tensor cross-terms may be zero/negative; unchanged computeTrans checks physical conductances.
        assert(numel(grid.poro)==G.cells.num&&all(isfinite(grid.perm(:))));
        assert(all(grid.poro(:)>0)&&all(grid.poro(:)<1)&&all(G.cells.volumes>0));
        net=phase_network_axis(G,struct('perm',grid.perm,'poro',grid.poro),3);
        master=readmatrix(fullfile(c.science,'data',sprintf('master_sample%d.csv',r.sample)));
        sg=unique([kr(:,1);master(:,1)]);
        clay=[sg,interp1(kr(:,1),kr(:,2:3),sg,'linear'),interp1(master(:,1),master(:,2),sg,'linear')];
        m=struct('e',net.e,'inlet',net.ci,'outlet',net.co,'material',1+double(grid.isSmear(:)),'kr',{{sand,clay}});
        mkdir(r.rawdir);
        write_raw(fullfile(r.rawdir,'raw_unplanned.bin'),r.slot,m,grid.perm,grid.poro,G.cells.volumes,[0;1],0,cfg);
        fid=fopen(fullfile(r.rawdir,'geometry_unplanned.bin'),'w','ieee-le');assert(fid>0);
        cleaner=onCleanup(@()fclose(fid));
        fwrite(fid,int64([2026090906,r.slot,3,net.n,size(net.e,1),numel(net.ci),numel(net.co)]),'int64');
        fwrite(fid,[net.length,net.area],'double');fwrite(fid,zeros(32,1,'uint8'),'uint8');
        for field={'e','h1','h2','ci','co','hi','ho'}
            v=net.(field{1});if ismember(field{1},{'e','ci','co'}),fwrite(fid,int64(v),'int64');else,fwrite(fid,v,'double');end
        end
        clear cleaner
        status=struct('passed',true,'slot',r.slot,'sample',r.sample,'source_row',C.SelectedSampleIndex, ...
            'seed',C.ReplaySeed,'attempt_index',C.SourceAcceptedAttemptIndex,'architecture_sha256',char(C.DiscreteArchitectureSha256), ...
            'max_log10_difference',C.MaxAbsLog10Diff,'cells',G.cells.num,'clay_fraction',mean(grid.isSmear(:)), ...
            'pore_volume',sum(G.cells.volumes(:).*grid.poro(:)),'bulk_volume',sum(G.cells.volumes(:)), ...
            'seconds',toc(started));
    catch err
        status=struct('passed',false,'slot',r.slot,'error',getReport(err,'extended'),'seconds',toc(started));
    end
    fid=fopen(r.status,'w');assert(fid>0);fprintf(fid,'%s\n',jsonencode(status));fclose(fid);
    fprintf('PREPARE_CORE %d passed=%d seconds=%.3f\n',r.slot,status.passed,status.seconds);
end
loaded=inmem('-completenames');fid=fopen(c.loaded,'w');assert(fid>0);
for i=1:numel(loaded),if isfile(loaded{i}),fprintf(fid,'%s\n',loaded{i});end,end
fclose(fid);
end
