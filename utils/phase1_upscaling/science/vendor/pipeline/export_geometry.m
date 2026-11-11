function export_geometry(packet,rawhash,out)
% Validation input packaging only; no MIP, flow solve, or property upscaling.
assert(~isfolder(out));mkdir(out);
P=load(packet,'net','core','qa_ref','expected','pc');net=P.net;
hash=uint8(sscanf(rawhash,'%2x').');assert(numel(hash)==32);
f=fopen(fullfile(out,'geometry.bin'),'w','ieee-le');assert(f>0);closefile=onCleanup(@()fclose(f));
fwrite(f,int64([2026090906,P.core,3,net.n,size(net.e,1),numel(net.ci),numel(net.co)]),'int64');
fwrite(f,[net.length,net.area],'double');fwrite(f,hash,'uint8');
fwrite(f,int64(net.e),'int64');fwrite(f,net.h1,'double');fwrite(f,net.h2,'double');
fwrite(f,int64(net.ci),'int64');fwrite(f,int64(net.co),'int64');
fwrite(f,net.hi,'double');fwrite(f,net.ho,'double');clear closefile
assert(size(P.expected,1)==100&&size(P.expected,2)>=8&&isequal(P.pc(:),P.expected(:,1)));
f=fopen(fullfile(out,'expected.csv'),'w');assert(f>0);
fprintf(f,[repmat('%.17g,',1,7),'%.17g\n'],P.expected(:,1:8)');fclose(f);
f=fopen(fullfile(out,'qa_reference.txt'),'w');assert(f>0);fprintf(f,'%.17g\n',P.qa_ref);fclose(f);
fprintf('EXPORTED geometry only for core%d; no mobility fields in numerical input\n',P.core);
end
