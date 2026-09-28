using Test, Random, SHA
include(joinpath(@__DIR__,"..","src","FaultCorePc.jl"))
using .FaultCorePc

function fixture(;n=8,disconnected=false)
    sand=[0.0 0 1 1e3; .001 .001 .99 1.1e3; .1 .1 .8 2e3; .95 1 0 1e7]
    clay=[0.0 0 1 2e5; .001 .001 .99 2.1e5; .1 .1 .8 4e5; .70 1 0 1e8]
    e=hcat(collect(1:n-1),collect(2:n))
    disconnected&&(e=e[1:end-1,:])
    CoreInput(0,fill(1e-14,n,1),fill(.25,n),ones(n),e,[1],[n],
        UInt8[isodd(i) ? 1 : 2 for i in 1:n],(sand,clay),[0.,1e10],Float64[],Settings())
end

@testset "Input, scaling, conventions" begin
    input=fixture();m,a=prepare_model(input)
    @test m.scale[1] == sqrt(Settings().reference_sand_k_m2*.25/(Settings().reference_sand_phi*1e-14))
    @test m.pv==fill(.25,m.n)
    @test a["cell_count"]==8 && a["sand_cells"]==4
    @test m.curves[1].sg[end]==.95 && m.curves[2].sg[end]==.70
    for nk in [3,6]
        perm=ones(8,nk);perm[:,nk].=1e-14
        z=CoreInput(0,perm,input.phi,input.volume,input.edges,input.inlet,input.outlet,
            input.material,input.tables,input.pressures,input.dictionary,input.settings)
        @test prepare_model(z)[1].scale==m.scale
    end
    zero=deepcopy(input);zero.phi[1]=0;zero.perm[2]=1e-30
    zm,za=prepare_model(zero)
    @test zm.pv[1]==0 && za["scaling_phi_floor_cells"]==1 && za["scaling_k_floor_cells"]==1
    for bad in [:negative_k,:nan_phi,:zero_vol,:material,:pressure,:empty_inlet]
        x=deepcopy(input)
        if bad==:negative_k;x.perm[1]=-1
        elseif bad==:nan_phi;x.phi[1]=NaN
        elseif bad==:zero_vol;x.volume[1]=0
        elseif bad==:material;x.material[1]=3
        elseif bad==:pressure;x.pressures[2]=-1
        else;empty!(x.inlet);end
        @test_throws AssertionError prepare_model(x)
    end
    r=upscale(m)
    @test r.rows[1].pressure_pa==0 && r.rows[1].sg_physical==0
    @test r.rows[end].gas_breakthrough && !r.reservoir_export_approved
    dm=prepare_model(fixture(disconnected=true))[1]
    @test prepare_paths(dm)===nothing
    dr=upscale(dm);@test isnan(dr.breakthrough_pa) && !dr.rows[end].gas_breakthrough
end

@testset "Independent path thresholds and exact fast path" begin
    rng=MersenneTwister(47)
    for trial in 1:60
        n=rand(rng,4:16);x=fixture(n=n)
        extra=[rand(rng,1:n,12) rand(rng,1:n,12)]
        e=vcat(x.edges,extra)
        m=prepare_model(CoreInput(0,x.perm,x.phi,x.volume,e,x.inlet,x.outlet,x.material,
            x.tables,x.pressures,x.dictionary,x.settings))[1]
        for gas in [true,false]
            th=Float64.(rand(rng,1:12,n));seeds=gas ? m.inlet : m.outlet
            got=FaultCorePc.path_thresholds(m,th,seeds;gas=gas)
            # Independent fixed-point edge relaxation, without heap or spanning tree.
            ref=fill(gas ? Inf : -Inf,n);ref[seeds].=th[seeds]
            for _ in 1:n
                for (u,v) in eachrow(vcat(e,e[:,[2,1]]))
                    candidate=gas ? max(ref[u],th[v]) : min(ref[u],th[v])
                    ref[v]=gas ? min(ref[v],candidate) : max(ref[v],candidate)
                end
            end
            @test got==ref
        end
        paths=prepare_paths(m)
        @test paths!==nothing
        pp=sort!(unique(vcat(0.,m.events,m.events.*1.001)))
        a=initial(n);b=initial(n);wa=Workspace(n);wb=Workspace(n)
        for p in pp
            advance_pc!(m,a,p,wa,paths);advance_pc!(m,b,p,wb,nothing)
            @test StandaloneMIP.same_state(a,b)
        end
    end
end

@testset "Sampling and streaming interchange" begin
    pool=vcat(0.,exp.(range(-10,20,length=4001)))
    pp=fixed_from_dictionary(pool)
    @test length(pp)==100 && pp[1]==0 && pp[end]==pool[end] && all(diff(pp).>0)
    @test_throws AssertionError fixed_from_dictionary(pool[1:40])
    m=prepare_model(fixture())[1];hash=sha256("synthetic input")
    mktempdir() do tmp
        path=joinpath(tmp,"states.bin");archive=StateArchive(path,m.n,length(m.pressures),hash)
        r=try
            upscale(m;callback=(j,s,w,row)->write_state!(archive,j,s,w))
        finally;close(archive);end
        count=Ref(0)
        read_archive(path;expected_model_hash=hash) do j,s,kw,kg
            count[]+=1
            @test s.pressure==m.pressures[j] && all(0 .<= kw .<=1) && all(0 .<=kg .<=1)
            if j==2;@test StandaloneMIP.same_state(s,r.final_state);end
        end
        @test count[]==2
        @test_throws AssertionError StateArchive(path,m.n,2,hash)
        @test_throws AssertionError read_archive(path,(args...)->nothing;expected_model_hash=zeros(UInt8,32))
        open(path,"a") do io;write(io,UInt8(0));end
        @test_throws AssertionError read_archive(path,(args...)->nothing)
    end
end
