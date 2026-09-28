include("FaultCoreUpscaling.jl")
using .FaultCoreUpscaling
const Pc=FaultCoreUpscaling.Pc
reference_dir,out=ARGS
@assert !ispath(out);mkpath(out)
input=Pc.read_input(joinpath(reference_dir,"raw.bin"));n=length(input.phi);k=input.perm[:,end]
qa=1/sum(1 ./ k)
stats=open(joinpath(reference_dir,"reference_stats.bin")) do io
    nr,nc=Pc.StandaloneMIP.readvec(io,Int64,2);reshape(Pc.StandaloneMIP.readvec(io,Float64,nr*nc),nr,nc)
end
open(joinpath(out,"expected.csv"),"w") do dest
    open(strip(read(joinpath(reference_dir,"reference_path.txt"),String))) do io
        @assert read(io,Int64)==100
        for j=1:100
            state=Pc.StandaloneMIP.read_state(io,n)
            kw=Pc.StandaloneMIP.readvec(io,Float64,n);kg=Pc.StandaloneMIP.readvec(io,Float64,n)
            q(kr)=any(==(0),kr) ? 0. : 1/sum(1 ./ (k.*kr))
            println(dest,join([state.pressure,stats[j,2],q(kw)/qa,q(kg)/qa,stats[j,3:6]...],','))
        end
    end
end
open(joinpath(out,"qa_reference.txt"),"w") do io;println(io,qa);end
