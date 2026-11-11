function stats=export_reference(m,pp,out,reuse)
% No Kr flow solves. Only reference MIP state fields and physical PV summaries.
stats=zeros(numel(pp),8);events=unique([m.entry;m.endpoint_pressure]);
m.separated_maps=separated_event_maps(m);
if isempty(reuse)
    ref=fullfile(out,'reference.bin');assert(~isfile(ref));
    fid=fopen(ref,'w','ieee-le');assert(fid>0);close_ref=onCleanup(@()fclose(fid));
    fwrite(fid,int64(numel(pp)),'int64');s=mip_initial(numel(m.pv));s.pressure=0;
else
    ref=reuse;fid=fopen(ref,'r','ieee-le');assert(fid>0);close_ref=onCleanup(@()fclose(fid));
    assert(fread(fid,1,'int64')==numel(pp));
end
previous=0;
for j=1:numel(pp)
    timer=tic;
    if isempty(reuse)
        [s,ne]=cell_mip_advance(m,s,pp(j),events);seconds=toc(timer);
        [kw,kg]=phase_values(m,s.sg);write_mip_state(fid,s);fwrite(fid,kw,'double');fwrite(fid,kg,'double');
    else
        n=numel(m.pv);p=fread(fid,2,'double');sg=fread(fid,n,'double');
        invaded=logical(fread(fid,n,'uint8'));trapped=logical(fread(fid,n,'uint8'));
        kw=fread(fid,n,'double');kg=fread(fid,n,'double');
        s=struct('pressure',p(1),'breakthrough',p(2),'sg',sg,'invaded',invaded,'trapped',trapped);
        assert(s.pressure==pp(j));[a,b]=phase_values(m,s.sg);assert(isequal(a,kw)&&isequal(b,kg));
        ne=sum(events>previous&events<=pp(j));seconds=NaN;
    end
    pv=sum(m.pv);
    stats(j,:)=[pp(j),sum(m.pv.*s.sg)/pv,sum(m.pv(s.trapped))/pv,sum(m.pv(kw==0))/pv, ...
        double(isfinite(s.breakthrough)),sum(m.pv(s.invaded))/pv,ne,seconds];
    previous=pp(j);
    if mod(j,10)==0,fprintf('REFERENCE %d/%d Pc %.9g Sg %.9g\n',j,numel(pp),pp(j),stats(j,2));end
end
if ~isempty(reuse),assert(isempty(fread(fid,1,'uint8')));end
clear close_ref
fid=fopen(fullfile(out,'reference_stats.bin'),'w','ieee-le');assert(fid>0);clean=onCleanup(@()fclose(fid));
fwrite(fid,int64(size(stats)),'int64');fwrite(fid,stats,'double');clear clean
fid=fopen(fullfile(out,'reference_path.txt'),'w');assert(fid>0);fprintf(fid,'%s\n',ref);fclose(fid);
save_json(fullfile(out,'reference_meta.json'),struct('fast_path',m.separated_maps.enabled, ...
    'reused_completed_states',~isempty(reuse),'points',numel(pp),'events',sum(stats(:,7)), ...
    'no_flow_solves',true,'reservoir_export_approved',false));
end
