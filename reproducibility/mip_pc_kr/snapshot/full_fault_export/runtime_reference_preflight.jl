using JutulDarcy, Jutul, MAT, TOML, SHA, LinearAlgebra
include(joinpath(@__DIR__, "contract.jl"))
using .GoMStep62EffectivePcGlobalPlateauContract
include(joinpath(@__DIR__, "hysteresis_validation.jl"))
common,specific,out=ARGS
@assert !ispath(out);mkpath(out)
start=time();BLAS.set_num_threads(1)
println("FULL_GRID_ASSEMBLY_BEGIN");flush(stdout)
setup=simulate_mrst_case(specific;common_mrst_path=common,specific_mrst_path=specific,
    do_sim=false,write_output=false,verbose=true,disable_hysteresis=false,
    hysteresis_s_min=0.05,use_mrst_transmissibility=false,fault_saturation_domain_mode="input",
    fault_pc_entry_treatment="plateau_all_active",explicit_fault_hysteresis_mode="reservoir",
    ds_max=0.05,max_nonlinear_iterations=10,max_timestep_cuts=25,
    well_volume_fraction=1.0e-3,nthreads=Threads.nthreads())
diag=GoMStep62EffectivePcGlobalPlateauContract.validate_assembled_case(setup)
D=matread(specific);sgof=D["fault"]["fluid_tables"]["SGOF"]
@assert D["wettability_scenario"]["clay_theta_deg"]==60
@assert length(sgof["tables"])==522
assembled=vec(setup.mrst["deck"]["PROPS"]["SGOF"])
for (region,table) in zip(Int.(vec(sgof["regions"])),vec(sgof["tables"]))
    @assert isequal(assembled[region],table)
end
raw_pressure=MAT.matopen(common) do f
    Float64.(vec(read(f,"state0")["pressure"]))
end
initial=setup.case.state0[:Reservoir]
pressure_error=maximum(abs.(vec(initial[:Pressure]).-raw_pressure))
@assert pressure_error<=1e-6
@assert all(isfinite,initial[:Pressure]) && minimum(initial[:Pressure])>0
@assert maximum(abs.(sum(initial[:Saturations];dims=1).-1))<1e-12
rm=setup.case.model.models[:Reservoir]
@assert JutulDarcy.get_reference_phase_index(rm.system)==JutulDarcy.phase_indices(rm.system).l
perm=diag.perm
kxx,kxy,kxz,kyy,kyz,kzz=(view(perm,:,i) for i in 1:6)
@assert all(>(0),kxx)&&all(>(0),kyy)&&all(>(0),kzz)
@assert all(>(0), kxx .* kyy .- kxy .^ 2)
determinant = kxx .* kyy .* kzz .+ 2 .* kxy .* kxz .* kyz .-
    kxx .* kyz .^ 2 .- kyy .* kxz .^ 2 .- kzz .* kxy .^ 2
@assert all(>(0), determinant)
h=CorrectedFaultValidation.validate!(setup,out)
open(joinpath(out,"PASS.toml"),"w") do io
    TOML.print(io,Dict("passed"=>true,"tables"=>522,"all_fault_tables_exact"=>true,
        "fault_hysteresis_disabled"=>h["fault_mobility_and_qoi_drainage_only"],
        "host_hysteresis_unchanged"=>h["host_matches_immutable_oracle"],
        "initial_pressure_max_error_pa"=>pressure_error,"positive_definite_permeability"=>true,
        "full_reservoir_simulation_run"=>false,"full_grid_runtime_assembly_passed"=>true,
        "wall_seconds"=>time()-start,"specific_sha256"=>bytes2hex(open(sha256,specific)),
        "common_sha256"=>bytes2hex(open(sha256,common)),"production_modified"=>false))
end
println("FULL_GRID_RUNTIME_PREFLIGHT_PASS");flush(stdout)
