function test_export_geometry(out)
assert(~isfolder(out));mkdir(out);
net=struct('n',3,'e',[1 2;2 3],'h1',[2;4],'h2',[6;8], ...
    'ci',1,'co',3,'hi',10,'ho',12,'length',3,'area',1);
core=7;pc=(0:99)';expected=[pc,repmat([.1,.2,.3,.4,.5,1,.6],100,1)];qa_ref=.125;
packet=fullfile(out,'fixture.mat');save(packet,'net','core','pc','expected','qa_ref','-v7');
hash=repmat('a5',1,32);export_geometry(packet,hash,fullfile(out,'inputs'));
fid=fopen(fullfile(out,'inputs','geometry.bin'),'r','ieee-le');assert(fid>0);closefile=onCleanup(@()fclose(fid));
assert(isequal(fread(fid,7,'int64')',[2026090906 7 3 3 2 1 1]));
assert(isequal(fread(fid,2,'double')',[3 1]));assert(all(fread(fid,32,'uint8')==165));
assert(isequal(reshape(fread(fid,4,'int64'),2,2),net.e));
assert(isequal(fread(fid,2,'double'),net.h1));assert(isequal(fread(fid,2,'double'),net.h2));
assert(fread(fid,1,'int64')==1&&fread(fid,1,'int64')==3);
assert(fread(fid,1,'double')==10&&fread(fid,1,'double')==12);
assert(isempty(fread(fid,1,'uint8')));clear closefile
assert(isequal(readmatrix(fullfile(out,'inputs','expected.csv')),expected));
assert(str2double(fileread(fullfile(out,'inputs','qa_reference.txt')))==qa_ref);
disp('PASS geometry-only binary packaging, raw binding and reference export');
end
