function export_controls(numeric,out)
assert(~isfolder(out));mkdir(out);addpath(numeric);addpath(fullfile(numeric,'vendor'));
cfg=jsondecode(fileread(fullfile(numeric,'config.json')));rng(1909,'twister');
sand=[0 0 1 1e3;.001 .001 .99 1.1e3;.1 .1 .8 2e3;.95 1 0 1e7];
clay=[0 0 1 2e5;.001 .001 .99 2.1e5;.1 .1 .8 4e5;.70 1 0 1e8];
for trial=1:24
    folder=fullfile(out,sprintf('control%02d',trial));mkdir(folder);n=4+mod(trial,9);
    material=1+double(rand(n,1)>.5);if trial==1,material(:)=1;elseif trial==2,material(:)=2;end
    e=[(1:n-1)',(2:n)'];if mod(trial,6)==0,e=e(1:end-1,:);end
    m=struct('material',material,'kr',{{sand,clay}},'e',e,'inlet',1,'outlet',n);
    phi=.1+.3*rand(n,1);vol=.3+rand(n,1);k=10.^(-22+12*rand(n,1));
    if mod(trial,4)==0,phi(2)=0;k(3)=1e-30;end
    nk=[1,3,6];nk=nk(1+mod(trial,3));perm=repmat(k,1,nk);
    if nk==6,perm(:,[2,3,5])=0;end
    [m,~]=cell_pc_model(m,perm,phi,vol,cfg);
    pool=unique([0;logspace(log10(.98*min(m.entry)),log10(1.01*max(m.endpoint_pressure)),401)']);
    pp=pool([1,round(linspace(2,numel(pool),99))]);
    write_raw(fullfile(folder,'raw.bin'),trial,m,perm,phi,vol,pp,[],cfg);
    write_mip_packet(fullfile(folder,'model.bin'),'model',m,pp);
    export_reference(m,pp,folder,'');
end
fprintf('EXPORTED 24 independently prepared MATLAB reference controls\n');
end
