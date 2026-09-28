using JutulDarcy, Jutul, MAT, ForwardDiff, SHA, TOML, LinearAlgebra, Test
BLAS.set_num_threads(1)
input,out,common_path,specific_path=ARGS
@assert !ispath(out);mkpath(out)
D=matread(input);T=Matrix{Float64}.(vec(D["exportTables"]))
slotmap=Int.(vec(D["tableSlotMap"]))
@assert length(T)==522&&sort(slotmap)==collect(1:522)
common=matread(common_path);specific=matread(specific_path)
contract=JutulDarcy.validate_mrst_combined_specific_identity(common,specific)
@assert contract["passed"]&&contract["saturation_region_count"]==522
sgof=specific["fault"]["fluid_tables"]["SGOF"]
indices=Int.(vec(sgof["source_curve_linear_indices"]))
@assert all(vec(sgof["tables"])[i]==T[indices[i]] for i in 1:522)
open(joinpath(out,"full_fault_contract_pass.toml"),"w") do io
    TOML.print(io,Dict("passed"=>true,"contract"=>contract,
        "candidate_specific_sha256"=>bytes2hex(open(sha256,specific_path)),
        "common_sha256"=>bytes2hex(open(sha256,common_path))))
end
common=nothing;specific=nothing;GC.gc()
before=deepcopy(T)
plateau=JutulDarcy.apply_mrst_all_active_pc_entry_plateau!(T,collect(1:522))
@assert T==before&&plateau["adjusted_tables"]==0
props=Dict("SGOF"=>T)
kr=JutulDarcy.deck_relperm(Dict(),props;oil=true,water=false,gas=true,satnum=collect(1:522))
pc=JutulDarcy.deck_pc(props;oil=true,water=false,gas=true,satnum=collect(1:522))
function expected(t,s,j)
    s<=t[1,1]&&return t[1,j]
    s>=t[end,1]&&return t[end,j]
    i=searchsortedlast(view(t,:,1),s);w=(s-t[i,1])/(t[i+1,1]-t[i,1])
    (1-w)*t[i,j]+w*t[i+1,j]
end
# Record actual interpolation errors before testing, including on a failed run.
endpoint_records=[Dict("slot"=>slotmap[i],"raw_krg0"=>t[1,2],
    "raw_krw0"=>t[1,3],"imported_krg0"=>kr.krg[i](0.0),
    "imported_krw0"=>kr.krog[i](1.0)) for (i,t) in enumerate(T)]
open(joinpath(out,"endpoint_diagnostic.toml"),"w") do io
    TOML.print(io,Dict("absolute_tolerance"=>1e-12,"records"=>endpoint_records))
end
@testset "Endpoint tolerance rejects material errors" begin
    @test abs(eps(Float64))<=1e-12
    @test !(abs(1e-9)<=1e-12)
end
records=[]
@testset "All hybrid tables: importer, endpoints and AD" begin
    for (i,t) in enumerate(T)
        fg=kr.krg[i];fw=s->kr.krog[i](1-s);fp=pc.pc[1][i]
        points=sort!(unique(vcat(collect(range(0,1;length=1001)),t[:,1],
            (t[1:end-1,1]+t[2:end,1])/2,[nextfloat(0.),t[end,1]+1e-8])))
        err=zeros(3)
        for s in points
            for (j,f) in enumerate((fg,fw,fp))
                value=f(s);derivative=ForwardDiff.derivative(f,s)
                @test isfinite(value)&&isfinite(derivative)
                err[j]=max(err[j],abs(value-expected(t,s,j+1)))
                if j<3
                    @test -1e-12<=value<=1+1e-12
                    @test j==1 ? derivative>=-1e-10 : derivative<=1e-10
                else
                    @test value>0&&derivative>=-1e-6
                end
            end
        end
        @test maximum(err[1:2])<1e-9
        @test err[3]<=1e-6+1e-12*maximum(t[:,4])
        @test fp(0.)==t[1,4]&&fp(1.)==t[end,4]
        @test abs(fg(1.)-t[end,2])<1e-12&&fw(1.)==0
        @test t[1,1]==0&&t[1,2]==0&&t[1,3]==1
        # Imported piecewise interpolation need not reproduce zero bit-for-bit.
        @test abs(fg(0.))<=1e-12&&abs(fw(0.)-1)<=1e-12
        push!(records,Dict("slot"=>slotmap[i],"max_krg_error"=>err[1],
            "max_krw_error"=>err[2],"max_pc_error_pa"=>err[3],
            "initial_krg_error"=>abs(fg(0.)),"initial_krw_error"=>abs(fw(0.)-1)))
    end
end
@assert T==before
open(joinpath(out,"constitutive_pass.toml"),"w") do io
    TOML.print(io,Dict("passed"=>true,"tables"=>522,
        "table_sha256"=>bytes2hex(open(sha256,input)),"plateau_adjustments"=>0,
        "raw_table_nodes_unchanged_by_import"=>true,"endpoint_absolute_tolerance"=>1e-12,
        "records"=>records))
end
