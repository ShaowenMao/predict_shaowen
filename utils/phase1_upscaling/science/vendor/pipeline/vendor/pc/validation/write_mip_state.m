function write_mip_state(fid,s)
fwrite(fid,[s.pressure;s.breakthrough],'double');fwrite(fid,s.sg,'double');
fwrite(fid,uint8(s.invaded),'uint8');fwrite(fid,uint8(s.trapped),'uint8');
end
