function m=read_reference_model(path)
fid=fopen(path,'r','ieee-le');assert(fid>0);clean=onCleanup(@()fclose(fid));
h=fread(fid,8,'int64');assert(numel(h)==8&&h(1)==2026090901);n=h(2);
m.e=fread(fid,[h(3),2],'int64');m.inlet=fread(fid,h(4),'int64');m.outlet=fread(fid,h(5),'int64');
m.material=fread(fid,n,'uint8');
for field={'scale','entry','endpoint_pressure','pv'},m.(field{1})=fread(fid,n,'double');end
m.kr={fread(fid,[h(6),4],'double'),fread(fid,[h(7),4],'double')};
m.pressure=fread(fid,h(8),'double');assert(isempty(fread(fid,1,'uint8')));
end
