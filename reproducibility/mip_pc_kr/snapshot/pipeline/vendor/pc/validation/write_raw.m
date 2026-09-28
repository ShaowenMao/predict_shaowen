function write_raw(path,core,m,perm,phi,vol,pp,pool,cfg)
assert(~isfile(path));fid=fopen(path,'w','ieee-le');assert(fid>0);clean=onCleanup(@()fclose(fid));
names={'entry_sg','sand_endpoint','clay_endpoint','min_scaling_phi','min_scaling_k_md', ...
    'md_to_m2','reference_sand_k_m2','reference_sand_phi','clay_fit_slope','clay_fit_intercept', ...
    'co2_brine_sigma_mn_m','co2_brine_theta_deg','hg_air_sigma_mn_m','hg_air_theta_deg','clay_anchor_sg'};
fwrite(fid,int64([2026090904,core,numel(phi),size(perm,2),size(m.e,1),numel(m.inlet),numel(m.outlet), ...
    size(m.kr{1},1),size(m.kr{2},1),numel(pp),numel(pool),numel(names)]),'int64');
for name=names,fwrite(fid,double(cfg.(name{1})),'double');end
fwrite(fid,perm,'double');fwrite(fid,phi,'double');fwrite(fid,vol,'double');
fwrite(fid,int64(m.e),'int64');fwrite(fid,int64(m.inlet),'int64');fwrite(fid,int64(m.outlet),'int64');
fwrite(fid,uint8(m.material),'uint8');
for x={m.kr{1},m.kr{2},pp,pool},fwrite(fid,x{1},'double');end
end
