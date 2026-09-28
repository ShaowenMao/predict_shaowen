function write_mip_packet(path,kind,varargin)
fid=fopen(path,'w','ieee-le');assert(fid>=0);cleanup=onCleanup(@() fclose(fid));
switch kind
    case 'model'
        m=varargin{1};pp=varargin{2};
        fwrite(fid,int64([2026090901,numel(m.pv),size(m.e,1),numel(m.inlet),numel(m.outlet), ...
            size(m.kr{1},1),size(m.kr{2},1),numel(pp)]),'int64');
        fwrite(fid,int64(m.e),'int64');fwrite(fid,int64(m.inlet),'int64');fwrite(fid,int64(m.outlet),'int64');
        fwrite(fid,uint8(m.material),'uint8');
        for v={m.scale,m.entry,m.endpoint_pressure,m.pv,m.kr{1},m.kr{2},pp}
            fwrite(fid,double(v{1}),'double');
        end
    case 'window'
        s=varargin{1};pp=varargin{2};fwrite(fid,int64(numel(pp)),'int64');write_mip_state(fid,s);
        fwrite(fid,pp,'double');
    otherwise,error('JuliaMIP:Packet','Unknown packet kind');
end
end
