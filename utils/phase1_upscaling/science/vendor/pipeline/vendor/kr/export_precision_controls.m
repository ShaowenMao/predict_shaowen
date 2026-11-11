function export_precision_controls(path)
rng(6140926,'twister');assert(~isfile(path));fid=fopen(path,'w','ieee-le');assert(fid>0);
cleanup=onCleanup(@()fclose(fid));fwrite(fid,int64([2026090903,32]),'int64');
for k=1:32
    n=5+mod(k,27);i=[(1:n-1)';randi(n,20,1)];j=[(2:n)';randi(n,20,1)];
    p=struct('i',i,'j',j,'tt',10.^(-18+24*rand(numel(i),1)), ...
        'ii',[1;1],'oo',[n;n],'tin',10.^(-8+10*rand(2,1)), ...
        'tout',10.^(-8+10*rand(2,1)),'scale',10.^(-3+6*rand(n,1)));
    x=[rand(n,1),randn(n,1)*1e-17];dx=randn(n,1)*1e-16;v=randn(n,1);
    [r,m]=phase_precision('audit',p,x);added=phase_precision('add',x,dx);
    applied=phase_precision_repair('apply',p,v);
    fwrite(fid,int64([n,numel(i),numel(p.ii),numel(p.oo)]),'int64');
    for a={p.i,p.j,p.tt,p.ii,p.oo,p.tin,p.tout,p.scale,x,dx,v,r,m,added,applied},fwrite(fid,double(a{1}),'double');end
end
fprintf('EXPORTED 32 binary128 ABI equivalence controls\n');
end
