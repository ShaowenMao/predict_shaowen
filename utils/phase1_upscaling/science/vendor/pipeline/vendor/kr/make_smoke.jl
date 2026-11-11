include("PhaseFlow.jl")
using .PhaseFlow
out=ARGS[1];@assert !isdir(out);mkpath(out)
n=5;e=hcat(collect(1:4),collect(2:5));net=Net(n,e,fill(2.,4),fill(2.,4),[1],[5],[2.],[2.])
kw=hcat(ones(n),[1.,.1,.2,.3,1.],[1.,1.,0.,1.,1.]);kg=hcat(zeros(n),[.1,.2,.3,.4,.5],ones(n))
sample=Int64[1,34,67];owner=vcat(fill(1,33),fill(2,33),fill(3,34));qa=1/n
q(k)=any(k.==0) ? 0.0 : 1/sum(1.0./k)
expected=zeros(100,3)
for i=1:100;expected[i,:]=[0.2*(owner[i]-1),q(kw[:,owner[i]])/qa,q(kg[:,owner[i]])/qa];end
open(joinpath(out,"phase_packet.bin"),"w") do io
    write(io,Int64[2026090902,7,n,4,1,1,3,100],Float64[n,1,qa],Int64.(e),net.h1,net.h2,
          Int64.(net.ci),Int64.(net.co),net.hi,net.ho,Float64.(0:99),expected,Int64.(owner),sample,kw,kg)
end
open(joinpath(out,"parts_checks.bin"),"w") do io
    write(io,Int64(7))
    for (s,phase,kr) in vcat([(0,0,ones(n))],[(sample[k],phase,phase==1 ? kw[:,k] : kg[:,k]) for k=1:3 for phase=1:2])
        p,_,_,keep,_=prepare(net,kr);write(io,Int64[s,phase,p.n],hex2bytes(parts_hash(p,keep)))
    end
end
println("Synthetic smoke packet only: ",out)
