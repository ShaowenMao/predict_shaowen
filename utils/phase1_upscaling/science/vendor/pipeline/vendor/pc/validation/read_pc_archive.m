function metadata=read_pc_archive(path,callback,expected_hash)
fid=fopen(path,'r','ieee-le');assert(fid>0);clean=onCleanup(@()fclose(fid));
header=fread(fid,3,'int64');assert(numel(header)==3&&header(1)==2026090905);
n=header(2);np=header(3);assert(n>0&&np>0);
hash=uint8(fread(fid,32,'uint8'));assert(numel(hash)==32);
hex=lower(reshape(dec2hex(hash,2)',1,[]));assert(strcmp(hex,expected_hash));
info=dir(path);assert(info.bytes==56+np*(24+26*n),'JuliaPc:Truncated','Wrong archive length');
for j=1:np
    assert(fread(fid,1,'int64')==j);p=fread(fid,2,'double');
    s=struct('pressure',p(1),'breakthrough',p(2),'sg',fread(fid,n,'double'), ...
        'invaded',logical(fread(fid,n,'uint8')),'trapped',logical(fread(fid,n,'uint8')));
    kw=fread(fid,n,'double');kg=fread(fid,n,'double');callback(j,s,kw,kg);
end
assert(isempty(fread(fid,1,'uint8')));
metadata=struct('points',np,'cells',n,'input_sha256',hex,'complete',true);
end
