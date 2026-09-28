using Test, SHA, TOML, DelimitedFiles
include("plan_and_run.jl")
@assert length(ARGS)==2
manifest,report=ARGS
items=readdlm(manifest,'\t',String;header=true)[1]
records=Dict[]
@testset "All eight pilot pressure plans" begin
    @test size(items,1)==8
    for row in eachrow(items)
        slot=parse(Int,row[1]);x=P.read_input(row[2]);model,_=P.prepare_model(x)
        pp,pool=dictionary(x);old=P.fixed_from_dictionary(pool,100)
        endpoint=maximum(model.endpoint)
        @test length(pp)==100
        @test all(diff(pp).>0)
        @test pp[end]>=endpoint
        @test pp[1:end-1]==old[1:end-1]
        changed=findall(pp.!=old)
        if slot==295
            @test changed==[100]
            @test reinterpret(UInt64,endpoint)-reinterpret(UInt64,old[end])==4
            @test pp[end]===endpoint
        else
            @test isempty(changed)
            previous=P.read_input(row[3])
            @test reinterpret(UInt64,pp)==reinterpret(UInt64,previous.pressures)
            @test reinterpret(UInt64,pool)==reinterpret(UInt64,previous.dictionary)
            for f in (:perm,:phi,:volume,:edges,:inlet,:outlet,:material,:tables)
                @test getfield(x,f)==getfield(previous,f)
            end
        end
        push!(records,Dict("slot"=>slot,"changed_sample_indices"=>changed,
            "old_max_pa"=>old[end],"new_max_pa"=>pp[end],"endpoint_pa"=>endpoint,
            "relative_shortfall"=>(endpoint-old[end])/endpoint,
            "raw_input_sha256"=>bytes2hex(open(sha256,row[2]))))
    end
end
@assert !ispath(report)
open(report,"w") do io
    TOML.print(io,Dict("passed"=>true,"records"=>records))
end
