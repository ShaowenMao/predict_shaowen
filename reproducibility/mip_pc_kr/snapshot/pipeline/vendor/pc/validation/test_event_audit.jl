using Test
include(joinpath(@__DIR__,"..","src","FaultCorePc.jl"))
using .FaultCorePc
include("event_audit.jl")

function model(;endpoint=20.,entry=1.)
    M=StandaloneMIP
    c=M.Curve([0.,.70],[0.,1.],[1.,0.],[.1,30.])
    e=[endpoint,10.,30.];en=[entry,1.,1.]
    row,nb=M.adjacency(3,[1 2;2 3])
    M.Model(3,row,nb,[1],[3],fill(UInt8(2),3),e./30.,en,e,ones(3),(c,c),
            [0.,15.,20.],sort(unique(vcat(en,e))))
end

function run_gate(m,ref;before_pressure=15.,pc=20.,wrong_native=0,wrong_ref=0,
                  drop_trapped=false,change_sg=false,no_paths=false)
    paths=prepare_paths(m);rp=prepare_paths(ref)
    s=initial(m.n);rs=initial(ref.n);w=Workspace(m.n);rw=Workspace(ref.n)
    advance_pc!(m,s,before_pressure,w,paths);advance_pc!(ref,rs,before_pressure,rw,rp)
    old=deepcopy(s);rold=deepcopy(rs)
    ne=advance_pc!(m,s,pc,w,paths);rn=advance_pc!(ref,rs,pc,rw,rp)
    drop_trapped&&(old.trapped[1]=0)
    change_sg&&(s.sg[1]=nextfloat(s.sg[1]))
    event_identity_gate(m,ref)
    event_schedule_gate(m,ref,before_pressure,pc,ne+wrong_native,rn+wrong_ref,
        old,rold,s,rs,no_paths ? nothing : paths,rp)
end

@testset "Exact native schedule and inactive endpoint audit" begin
    ref=model();m=model(endpoint=nextfloat(20.))
    @test isempty(run_gate(ref,ref))
    report=run_gate(m,ref)
    @test length(report)==1
    @test report[1]["gridblock"]==1 && report[1]["endpoint_difference_ulp"]==1
    @test report[1]["state_inactive_proved"] && report[1]["native_schedules_exact"]
    @test run_gate(ref,m)[1]["endpoint_difference_pa"]<0
    @test_throws AssertionError run_gate(m,ref;wrong_native=1)
    @test_throws AssertionError run_gate(m,ref;wrong_ref=-1)
    @test_throws AssertionError run_gate(m,ref;drop_trapped=true)
    @test_throws AssertionError run_gate(m,ref;change_sg=true)
    @test_throws AssertionError run_gate(m,ref;before_pressure=5.)
    @test_throws AssertionError run_gate(m,ref;no_paths=true)
    @test_throws AssertionError run_gate(model(endpoint=nextfloat(nextfloat(20.))),ref)
    @test_throws AssertionError run_gate(model(entry=nextfloat(1.)),ref;before_pressure=0.,pc=1.)
    @test_throws AssertionError event_identity_gate(model(entry=prevfloat(1.)),ref)
end
