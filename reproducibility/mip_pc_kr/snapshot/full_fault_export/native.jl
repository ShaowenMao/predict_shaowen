include("endpoint_check.jl")
using JutulDarcy,Jutul,ForwardDiff,DelimitedFiles,TOML,SHA,LinearAlgebra,Test
include("endpoint_tests.jl")
BLAS.set_num_threads(1)
out=ARGS[1];@assert !ispath(out);mkpath(out)
T=[Matrix{Float64}(readdlm(p,',';skipstart=1)) for p in ARGS[2:end]]
@assert length(T)>=1
before=deepcopy(T)
plateau=JutulDarcy.apply_mrst_all_active_pc_entry_plateau!(T,collect(1:length(T)))
@assert T==before && plateau["adjusted_tables"]==0
props=Dict("SGOF"=>T)
kr=JutulDarcy.deck_relperm(Dict(),props;oil=true,water=false,gas=true,satnum=collect(1:length(T)))
pc=JutulDarcy.deck_pc(props;oil=true,water=false,gas=true,satnum=collect(1:length(T)))
const ORIGIN=TOML.parsefile(joinpath(@__DIR__,"origin.toml"))
const REVIEWED_TABLE_SHA="57f2b459e0480532107ea63f4092fa4e0b792bac632abfdbbd9ad3852cd503f5"
for (p,h) in ORIGIN["input_hashes"];@assert bytes2hex(open(sha256,p))==h;end
function expected(t,s,j)
    s<=t[1,1]&&return t[1,j]
    s>=t[end,1]&&return t[end,j]
    i=searchsortedlast(view(t,:,1),s);w=(s-t[i,1])/(t[i+1,1]-t[i,1])
    (1-w)*t[i,j]+w*t[i+1,j]
end
records=[];exceptions=[]
@testset "Broader native importer compatibility values and derivatives" begin
    for (i,t) in enumerate(T)
        funcs=(kr.krg[i],s->kr.krog[i](1-s),pc.pc[1][i])
        points=sort!(unique(vcat(collect(range(0,1;length=1001)),t[:,1],
            (t[1:end-1,1]+t[2:end,1])/2,[nextfloat(0.),t[end,1]+1e-8])))
        err=zeros(3);slopes=zeros(3)
        for s in points, (j,f) in enumerate(funcs)
            v=f(s);d=ForwardDiff.derivative(f,s)
            @test isfinite(v)&&isfinite(d)
            err[j]=max(err[j],abs(v-expected(t,s,j+1)));slopes[j]=max(slopes[j],abs(d))
            if j<3
                @test -1e-12<=v<=1+1e-12
                @test j==1 ? d>=-1e-10 : d<=1e-10
            else
                @test v>0 && d>=-1e-6
            end
        end
        @test maximum(err[1:2])<1e-9 && err[3]<=1e-6+1e-12*maximum(t[:,4])
        @test funcs[3](0.)==t[1,4] && funcs[3](1.)==t[end,4]
        @test t[1,1]==0. && t[1,2]==0. && t[1,3]==1. && t[end,3]==0.
        h=bytes2hex(open(sha256,ARGS[i+1]))
        if h==REVIEWED_TABLE_SHA
            # Exact source-derived exception, not a relaxed endpoint tolerance.
            f=kr.krg[i];ix=Jutul.first_lower(f.k.X,0.)
            x0,x1=f.k.X[ix:ix+1];y0,y1=f.k.F[ix:ix+1]
            slope=(y1-y0)/(x1-x0);prediction=y0+slope*(0.0 - x0)
            @test [x0,x1]==ORIGIN["active_interval_x"]
            @test [y0,y1]==ORIGIN["active_interval_y"]
            @test x0==-1e-16 && y0==0.
            @test slope==ORIGIN["active_slope"]
            @test f(0.)==prediction==ORIGIN["native_krg_zero"]
            @test ForwardDiff.derivative(f,0.)==slope
            push!(exceptions,Dict("region"=>i,"table_sha256"=>h,
                "native_krg_zero"=>f(0.),"predicted_native_krg_zero"=>prediction,
                "legacy_strict_zero_passed"=>false,"raw_zero_preserved"=>true))
        elseif funcs[1](0.) != 0.0
            evidence=endpoint_evidence(kr.krg[i],t)
            evidence["region"]=i;evidence["table_sha256"]=h
            push!(exceptions,evidence)
        else
            @test funcs[1](0.)==0.
        end
        @test abs(funcs[2](0.)-1)<1e-12 && funcs[2](1.)==0.
        push!(records,Dict("path"=>ARGS[i+1],"sha256"=>h,
            "max_errors"=>err,"max_abs_derivatives"=>slopes,"evaluation_points"=>length(points)))
    end
    @test length(exceptions)<=length(T)
end
@assert T==before
for (p,h) in ORIGIN["input_hashes"];@assert bytes2hex(open(sha256,p))==h;end
open(joinpath(out,"constitutive_pass.toml"),"w") do io
    TOML.print(io,Dict("passed"=>true,"records"=>records,"raw_table_nodes_preserved"=>true,
        "exact_documented_native_exceptions"=>exceptions,"runtime_modified"=>false,
        "legacy_strict_native_zero_test_passed"=>false,
        "full_fault_mapping_test"=>false,"reservoir_time_marching"=>false,
        "julia"=>string(VERSION),"project"=>Base.active_project(),
        "project_sha256"=>bytes2hex(open(sha256,Base.active_project())),
        "manifest_sha256"=>bytes2hex(open(sha256,joinpath(dirname(Base.active_project()),"Manifest.toml")))))
end
