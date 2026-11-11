function net=phase_network_axis(G,rock,axis)
assert(ismember(axis,1:3));n=G.cells.num;nf=G.faces.num;
c=rldecode((1:n)',diff(G.cells.facePos));f=G.cells.faces(:,1);
h=computeTrans(G,rock);assert(all(isfinite(h)&h>0));
side=1+(G.faces.neighbors(f,1)~=c);
H=accumarray([f,side],h,[nf,2],@sum,0);
inside=all(G.faces.neighbors>0,2);exterior=~inside;
extent=max(G.nodes.coords)-min(G.nodes.coords);
lo=min(G.nodes.coords(:,axis));hi=max(G.nodes.coords(:,axis));
tol=1e-10*max(1,hi-lo);coordinate=G.faces.centroids(:,axis);
fi=find(exterior & abs(coordinate-lo)<tol);
fo=find(exterior & abs(coordinate-hi)<tol);
ci=max(G.faces.neighbors(fi,:),[],2);co=max(G.faces.neighbors(fo,:),[],2);
expected=prod(G.cartDims)/G.cartDims(axis);
assert(numel(ci)==expected && numel(co)==expected);
ids=reshape(1:n,G.cartDims);a={':',':',':'};b=a;a{axis}=1;b{axis}=G.cartDims(axis);
want=ids(a{:});assert(isequal(sort(ci),sort(want(:))));
want=ids(b{:});assert(isequal(sort(co),sort(want(:))));
area=sum(G.faces.areas(fo));area_in=sum(G.faces.areas(fi));
assert(abs(area/area_in-1)<1e-10);
assert(abs(area/prod(extent(setdiff(1:3,axis)))-1)<1e-10);
net=struct('n',n,'e',double(G.faces.neighbors(inside,:)), ...
    'h1',H(inside,1),'h2',H(inside,2),'ci',ci,'co',co, ...
    'hi',sum(H(fi,:),2),'ho',sum(H(fo,:),2),'length',extent(axis),'area',area);
end
