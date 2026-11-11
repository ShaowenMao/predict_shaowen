function export_packet(numeric,packet,out,profile_sample,profile_phase)
addpath(numeric);setup=fullfile(out,'export_setup');assert(~isfolder(setup));mkdir(setup);
test_parts_fingerprint;
setup_candidate(setup);cd(out);P=load(packet);
assert(numel(P.pc)==100&&all(P.owner>=1)&&max(P.owner)==numel(P.unique_sample));
path=fullfile(out,'phase_packet.bin');assert(~isfile(path));fid=fopen(path,'w','ieee-le');assert(fid>0);
clean=onCleanup(@()fclose(fid));net=P.net;
fwrite(fid,int64([2026090902,P.core,net.n,size(net.e,1),numel(net.ci),numel(net.co),numel(P.unique_sample),100]),'int64');
fwrite(fid,[net.length,net.area,P.qa_ref],'double');
fwrite(fid,int64(net.e),'int64');fwrite(fid,net.h1,'double');fwrite(fid,net.h2,'double');
fwrite(fid,int64(net.ci),'int64');fwrite(fid,int64(net.co),'int64');fwrite(fid,net.hi,'double');fwrite(fid,net.ho,'double');
fwrite(fid,P.pc,'double');fwrite(fid,P.expected(:,2:4),'double');fwrite(fid,int64(P.owner),'int64');
fwrite(fid,int64(P.unique_sample),'int64');fwrite(fid,P.kw,'double');fwrite(fid,P.kg,'double');clear clean
rows=struct([]);
checks=fopen(fullfile(out,'parts_checks.bin'),'w','ieee-le');assert(checks>0);
close_checks=onCleanup(@()fclose(checks));fwrite(checks,int64(2*numel(P.unique_sample)+1),'int64');
add_one(ones(net.n,1),0,'absolute');
for k=1:numel(P.unique_sample)
    add_one(P.kw(:,k),P.unique_sample(k),'brine');add_one(P.kg(:,k),P.unique_sample(k),'gas');
end
clear close_checks
save_json(fullfile(out,'parts_checks.json'),struct('records',rows,'all_inputs_exported',true));
save_json(fullfile(out,'mapping.json'),struct('core',P.core,'qa_ref',P.qa_ref,'owner',P.owner, ...
    'unique_sample',P.unique_sample,'pc',P.pc,'sg',P.expected(:,2)));
export_precision_controls(fullfile(out,'precision_controls.bin'));
if strcmp(profile_phase,'absolute'),kr=ones(net.n,1);else
    k=find(P.unique_sample==profile_sample);assert(isscalar(k));
    if strcmp(profile_phase,'brine'),kr=P.kw(:,k);else,assert(strcmp(profile_phase,'gas'));kr=P.kg(:,k);end
end
profile clear;profile on;timer=tic;[q,info]=candidate_phase_conductance(net,kr,'direct');wall=toc(timer);
profile off;details=profile('info');save(fullfile(out,'matlab_profile.mat'),'details','q','info','wall','profile_sample','profile_phase');
save_json(fullfile(out,'matlab_profile.json'),struct('sample',profile_sample,'phase',profile_phase, ...
    'q',q,'info',info,'wall_seconds',wall,'scope','diagnostic only; profiling overhead excluded from paired timings', ...
    'functions',{details.FunctionTable}));
fprintf('EXPORTED core%d %d unique states, %d conservative operator checks\n',P.core,numel(P.unique_sample),numel(rows));
    function add_one(kr,sample,phase)
        [p,~,~,~,keep]=phase_parts(net,kr);n=sum(keep);
        [h,bytes]=parts_fingerprint(p,keep);
        phase_id=find(strcmp(phase,{'absolute','brine','gas'}))-1;
        fwrite(checks,int64([sample,phase_id,n]),'int64');fwrite(checks,bytes,'uint8');
        row=struct('sample',sample,'phase',phase,'parts_sha256',h,'spanning_cells',n);
        if isempty(rows),rows=row;else,rows(end+1)=row;end
    end
end
