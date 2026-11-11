function test_parts_fingerprint
p=struct('i',[],'j',[],'tt',[],'ii',[],'oo',[],'tin',[],'tout',[]);
[h,bytes]=parts_fingerprint(p,false(4,1));
assert(strcmp(h,'6db65fd59fd356f6729140571b5bcd6bb3b83492a16e1bf0a3884442fc3c8a0e'));
assert(numel(bytes)==32);
p.ii=1;p.oo=1;p.tin=2;p.tout=3;
assert(strcmp(parts_fingerprint(p,true),'8f9baeaebba10eb47ee6fb0c3b049758c56fcce3c43692b4ef056527ee3c9052'));
p.i=[1;2];p.j=[2;3];p.tt=[2;4];p.ii=1;p.oo=3;keep=true(3,1);
digest=java.security.MessageDigest.getInstance('SHA-256');
digest.update(typecast(int64([3,2,1,1]),'int8'));
digest.update(typecast(uint8(keep),'int8'));
for a={p.i,p.j,p.tt,p.ii,p.oo,p.tin,p.tout},digest.update(typecast(double(a{1}(:)),'int8'));end
original=typecast(digest.digest(),'uint8');[~,recovered]=parts_fingerprint(p,keep);
assert(isequal(original,recovered));
fprintf('PASS four fingerprint regression checks: empty, single-node, and unchanged nonempty encoding\n');
end
