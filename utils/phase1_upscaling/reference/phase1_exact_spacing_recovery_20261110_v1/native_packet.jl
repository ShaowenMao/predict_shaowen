using JutulDarcy,Jutul,ForwardDiff,DelimitedFiles,TOML,SHA,Test,LinearAlgebra
BLAS.set_num_threads(1)
cfg=TOML.parsefile(ARGS[1]);include(joinpath(cfg["science"],"ClayPc40.jl"))
using .ClayPc40
function check_native(r)
    out=joinpath(r["work"],"NATIVE_PASS.toml")
    isfile(out)&&return
    path=joinpath(r["work"],"export/sgof.csv")
    t=Matrix{Float64}(readdlm(path,',';skipstart=1));original=copy(t)
    T=[t];plateau=JutulDarcy.apply_mrst_all_active_pc_entry_plateau!(T,[1])
    @assert T[1]==original && plateau["adjusted_tables"]==0
    props=Dict("SGOF"=>T)
    pc=JutulDarcy.deck_pc(props;oil=true,water=false,gas=true,satnum=[1])
    kr=JutulDarcy.deck_relperm(Dict(),props;oil=true,water=false,gas=true,satnum=[1])
    funcs=(kr.krg[1],s->kr.krog[1](1-s),pc.pc[1][1]);err=zeros(3)
    points=sort!(unique(vcat(collect(range(0,1;length=1001)),t[:,1],(t[1:end-1,1]+t[2:end,1])/2)))
    for sg in points,(j,f) in enumerate(funcs)
        v=f(sg);d=ForwardDiff.derivative(f,sg)
        @assert isfinite(v)&&isfinite(d)
        expected=interpolate_bounded(t[:,1],t[:,j+1],clamp(sg,t[1,1],t[end,1]))
        err[j]=max(err[j],abs(v-expected))
        @assert j==2 ? d<=1e-10 : d>=-1e-6
    end
    @assert maximum(err[1:2])<1e-9 && err[3]<1e-6+1e-12*maximum(t[:,4]) && t==original
    open(out,"w") do io
        TOML.print(io,Dict("passed"=>true,"record_sha256"=>r["record_sha256"],"method"=>cfg["method"],
                          "sgof_sha256"=>bytes2hex(open(sha256,path)),"max_errors"=>err));flush(io)
    end
    println("NATIVE_PASS ",r["key"]);flush(stdout)
end
for r in cfg["cases"]
    try
        check_native(r)
    catch err
        out=joinpath(r["work"],"NATIVE_ERROR.toml")
        if !ispath(out)
            open(out,"w") do io;TOML.print(io,Dict("passed"=>false,"error"=>sprint(showerror,err,catch_backtrace())));end
        end
        println("NATIVE_HOLD ",r["key"]);flush(stdout)
    end
end
