function check_wettability(base,changed,reference,mrst)
% Verify the native MRST importer sees exactly the intended clay-only change.
run(fullfile(mrst,'startup.m'));
mrstModule add ad-core ad-props ad-blackoil deckformat;
A=convertDeckUnits(readEclipseDeck(base));B=convertDeckUnits(readEclipseDeck(changed));
ref=readmatrix(reference);
assert(numel(A.PROPS.SGOF)==2 && numel(B.PROPS.SGOF)==2);
assert(isequaln(A.PROPS.SGOF{1},B.PROPS.SGOF{1}));
assert(all(abs(B.PROPS.SGOF{2}-ref)<=1e-12+1e-12*abs(ref),'all'));
T=B.PROPS.SGOF{2};
assert(T(end,1)==.8 && T(end,2)==1 && T(end,3)==0);
B.PROPS.SGOF{2}=A.PROPS.SGOF{2};
assert(isequaln(A,B),'Non-clay deck properties changed');
B=convertDeckUnits(readEclipseDeck(changed));F=initDeckADIFluid(B);
assert(abs(F.krPts.g(2,3)-.8)<1e-12 && abs(F.krPts.og(2,2)-.2)<1e-12);
fprintf('WRR60_REFERENCE_PASS clay Sgmax=.8 Swi=.2; sand and all other deck fields unchanged\n');
end
