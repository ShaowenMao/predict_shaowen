function [diagonal, span] = tensor_descriptors(k)
% MRST symmetric 3-D ordering is [xx, xy, xz, yy, yz, zz].
assert(isnumeric(k) && ismatrix(k) && ~isempty(k) && all(isfinite(k(:))));
if size(k,2)==6
    diagonal=k(:,[1,4,6]);
    scale=max(diagonal,[],2); assert(all(scale>0));
    a=k./scale;
    minor2=a(:,1).*a(:,4)-a(:,2).^2;
    determinant=a(:,1).*a(:,4).*a(:,6)+2*a(:,2).*a(:,3).*a(:,5) ...
        -a(:,1).*a(:,5).^2-a(:,4).*a(:,3).^2-a(:,6).*a(:,2).^2;
    assert(all(a(:,1)>0) && all(minor2>0) && all(determinant>0));
elseif size(k,2)==3
    diagonal=k;
elseif size(k,2)==1
    diagonal=repmat(k,1,3);
else
    error('GeometryReport:TensorLayout','Unsupported permeability tensor layout');
end
assert(all(diagonal(:)>0));
span=max(log10(diagonal),[],1)-min(log10(diagonal),[],1);
end
