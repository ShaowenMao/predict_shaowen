module FaultCorePc
using SHA, TOML
include(joinpath(@__DIR__, "..", "vendor", "StandaloneMIP.jl"))
using .StandaloneMIP
export Settings, CoreInput, prepare_model, fixed_from_dictionary, prepare_paths,
       advance_pc!, observe, upscale, read_input, StateArchive, write_state!, read_archive,
       initial, Workspace, phase!, StandaloneMIP, run_file

Base.@kwdef struct Settings
    entry_sg::Float64 = 0.001
    sand_endpoint::Float64 = 0.95
    clay_endpoint::Float64 = 0.70
    min_scaling_phi::Float64 = 0.0001
    min_scaling_k_md::Float64 = 1e-9
    md_to_m2::Float64 = 9.869233e-16
    reference_sand_k_m2::Float64 = 7.60393535652603e-13
    reference_sand_phi::Float64 = 0.289875
    clay_fit_slope::Float64 = -0.1992
    clay_fit_intercept::Float64 = 1.407
    co2_brine_sigma_mn_m::Float64 = 25.0
    co2_brine_theta_deg::Float64 = 30.0
    hg_air_sigma_mn_m::Float64 = 485.0
    hg_air_theta_deg::Float64 = 140.0
    clay_anchor_sg::Float64 = 0.10
end

struct CoreInput
    core::Int
    perm::Matrix{Float64}
    phi::Vector{Float64}
    volume::Vector{Float64}
    edges::Matrix{Int}
    inlet::Vector{Int}
    outlet::Vector{Int}
    material::Vector{UInt8}
    tables::Tuple{Matrix{Float64},Matrix{Float64}}
    pressures::Vector{Float64}
    dictionary::Vector{Float64}
    settings::Settings
end

function read_input(path)
    ENDIAN_BOM == 0x04030201 || error("Input format requires a little-endian host")
    open(path,"r") do io
        magic,core,n,nk,ne,ni,no,n1,n2,np,nd,nc=readvec(io,Int64,12)
        @assert magic==2026090904 && n>0 && nk in (1,3,6) && ne>=0
        @assert ni>0 && no>0 && n1>=2 && n2>=2 && np>=2 && nd>=0 && nc==15
        cfg=Settings(readvec(io,Float64,nc)...)
        perm=reshape(readvec(io,Float64,n*nk),n,nk)
        phi=readvec(io,Float64,n);vol=readvec(io,Float64,n)
        edges=reshape(readvec(io,Int64,2ne),ne,2)
        inlet=readvec(io,Int64,ni);outlet=readvec(io,Int64,no)
        material=readvec(io,UInt8,n)
        tables=(reshape(readvec(io,Float64,4n1),n1,4),reshape(readvec(io,Float64,4n2),n2,4))
        pp=readvec(io,Float64,np);dictionary=readvec(io,Float64,nd)
        @assert eof(io)
        CoreInput(core,perm,phi,vol,edges,inlet,outlet,material,tables,pp,dictionary,cfg)
    end
end

function fixed_from_dictionary(pool,count=100)
    @assert count>=3 && length(pool)>=count && pool[1]==0
    @assert all(isfinite,pool) && all(diff(pool).>0)
    positive=pool[2:end];logs=log.(positive);lo=logs[1];hi=logs[end]
    chosen=[0.0]
    for j in 0:count-2
        target=lo+j*(hi-lo)/(count-2)
        # findmin uses the first occurrence, matching the frozen lower-index tie rule.
        _,index=findmin(abs.(logs.-target));push!(chosen,positive[index])
    end
    @assert length(unique(chosen))==count "Pressure dictionary cannot support this fixed grid"
    chosen
end

function prepare_model(input::CoreInput)
    s=input.settings;n=length(input.phi);nk=size(input.perm,2)
    @assert size(input.perm,1)==n && nk in (1,3,6) && size(input.edges,2)==2 && n>0
    @assert length(input.volume)==length(input.material)==n
    @assert all(isfinite,input.perm) && all(x->isfinite(x)&&0<=x<=1,input.phi)
    @assert all(x->isfinite(x)&&x>0,input.volume)
    @assert all(x->x in (1,2),input.material)
    @assert !isempty(input.inlet) && !isempty(input.outlet)
    @assert all(1 .<= input.inlet .<= n) && all(1 .<= input.outlet .<= n)
    @assert all(x->isfinite(x)&&x>=0,input.pressures) && all(diff(input.pressures).>0)
    @assert input.pressures[1]==0 && length(input.pressures)>=2
    @assert all(isfinite,getfield.(Ref(s),fieldnames(Settings)))
    @assert s.min_scaling_phi>0 && s.min_scaling_k_md>0 && s.md_to_m2>0
    @assert s.reference_sand_k_m2>0 && s.reference_sand_phi>0
    @assert s.co2_brine_sigma_mn_m>0 && s.hg_air_sigma_mn_m>0
    kzz=input.perm[:,nk==1 ? 1 : nk==3 ? 3 : 6];@assert all(kzz.>0)
    curves=map(enumerate(input.tables)) do (material,t)
        endpoint=material==1 ? s.sand_endpoint : s.clay_endpoint
        @assert size(t,2)==4 && size(t,1)>=2 && all(isfinite,t)
        @assert t[1,1]==0 && t[end,1]==endpoint && t[end,3]==0
        @assert all(diff(t[:,1]).>0) && all(diff(t[:,4]).>0) && all(t[:,4].>=0)
        @assert all(0 .<= t[:,2:3] .<= 1) && all(diff(t[:,2]).>=0) && all(diff(t[:,3]).<=0)
        @assert 0<=s.entry_sg<=endpoint
        StandaloneMIP.Curve(copy(t[:,1]),copy(t[:,2]),copy(t[:,3]),copy(t[:,4]))
    end
    sand,clay=curves;@assert 0<=s.clay_anchor_sg<=s.clay_endpoint
    anchor=StandaloneMIP.interpolate(clay.sg,clay.pc,s.clay_anchor_sg);@assert anchor>0
    factor=abs(s.co2_brine_sigma_mn_m*cosd(s.co2_brine_theta_deg)/
               (s.hg_air_sigma_mn_m*cosd(s.hg_air_theta_deg)))
    @assert isfinite(factor)&&factor>0
    entry_ref=[StandaloneMIP.interpolate(c.sg,c.pc,s.entry_sg) for c in curves]
    scale=zeros(n);entry=zeros(n);endpoint=zeros(n);pv=input.volume.*input.phi
    @assert sum(pv)>0
    for i in 1:n
        k=max(kzz[i],s.min_scaling_k_md*s.md_to_m2);phi=max(input.phi[i],s.min_scaling_phi)
        if input.material[i]==1
            scale[i]=sqrt(s.reference_sand_k_m2*phi/(s.reference_sand_phi*k))
        else
            pb=1e5*10.0^(s.clay_fit_slope*log10(k/s.md_to_m2)+s.clay_fit_intercept)*factor
            scale[i]=pb/anchor
        end
        c=curves[input.material[i]]
        entry[i]=scale[i]*entry_ref[input.material[i]];endpoint[i]=scale[i]*c.pc[end]
    end
    @assert all(x->isfinite(x)&&x>0,scale) && all(x->isfinite(x)&&x>0,entry)
    @assert all(isfinite,endpoint) && all(endpoint.>=entry)
    row,neighbor=adjacency(n,input.edges)
    m=Model(n,row,neighbor,copy(input.inlet),copy(input.outlet),copy(input.material),scale,
        entry,endpoint,pv,Tuple(curves),copy(input.pressures),sort!(unique(vcat(entry,endpoint))))
    audit=Dict("cell_count"=>n,"sand_cells"=>count(==(1),input.material),
        "clay_cells"=>count(==(2),input.material),"pore_volume_m3"=>sum(pv),
        "scaling_phi_floor_cells"=>count(<(s.min_scaling_phi),input.phi),
        "scaling_k_floor_cells"=>count(<(s.min_scaling_k_md*s.md_to_m2),kzz),
        "clay_conversion_factor"=>factor,"clay_reference_anchor_pa"=>anchor)
    m,audit
end

include("paths.jl")
include("outputs.jl")
end
