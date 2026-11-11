function [h,bytes]=parts_fingerprint(p,keep)
digest=java.security.MessageDigest.getInstance('SHA-256');
digest.update(typecast(int64([sum(keep),numel(p.tt),numel(p.tin),numel(p.tout)]),'int8'));
digest.update(typecast(uint8(keep),'int8'));
for a={p.i,p.j,p.tt,p.ii,p.oo,p.tin,p.tout}
    % Empty byte streams leave SHA256 unchanged; MATLAB's Java overload rejects [].
    if ~isempty(a{1}),digest.update(typecast(double(a{1}(:)),'int8'));end
end
bytes=typecast(digest.digest(),'uint8');h=lower(reshape(dec2hex(bytes,2)',1,[]));
end
