# Per-core snapshots only. No global cache, cross-core reuse, or disk state archive.
struct DrainageFrame{S,R}
    state::S
    kw::Vector{Float64}
    kg::Vector{Float64}
    row::R
    seal::String
end
struct DrainageCache{I}
    identity::I
    cells::Int
    frames::Vector{DrainageFrame}
    rows::Vector{NamedTuple}
    payload_bytes::Int
end

function drainage_row(j,p,ne,o,state,w)
    (sample=j,pressure_pa=p,sg_physical=o.sg_physical,
     trapped_pv_fraction=o.trapped_pv_fraction,immobile_pv_fraction=o.immobile_pv_fraction,
     gas_breakthrough=Int(o.gas_breakthrough),invaded_pv_fraction=o.invaded_pv_fraction,
     events=ne,state_hash=digest_state(state,w))
end

function frame_seal(identity,row,state,w)
    # Bind metadata omitted from the legacy state hash, including breakthrough.
    bytes2hex(sha256(string(repr(identity),repr(row),repr(state.pressure),
                           repr(state.breakthrough),digest_state(state,w))))
end

function checked_frame(c,j,identity)
    @assert isequal(identity,c.identity) "Drainage cache input identity mismatch"
    @assert length(c.frames)==length(c.rows)==100 && 1<=j<=100
    f=c.frames[j];s=f.state
    @assert all(length(a)==c.cells for a in (s.sg,s.invaded,s.trapped,f.kw,f.kg))
    @assert f.row==c.rows[j] && f.row.sample==j && s.pressure==f.row.pressure_pa
    @assert digest_state(s,f)==f.row.state_hash "Cached microstate changed"
    @assert frame_seal(identity,f.row,s,f)==f.seal "Cached metadata changed"
    f
end

function cached_pc_pass(m,paths,identity;max_bytes=512*1024^2)
    @assert length(m.pressures)==100 && !isempty(identity.raw_sha256)
    bytes=100*(26*m.n+16)
    @assert bytes<=max_bytes "Drainage cache exceeds bounded memory allowance"
    state=Pc.initial(m.n);w=Pc.Workspace(m.n)
    frames=DrainageFrame[];rows=NamedTuple[]
    for (j,p) in enumerate(m.pressures)
        ne=Pc.advance_pc!(m,state,p,w,paths);o=Pc.observe(m,state,w)
        row=drainage_row(j,p,ne,o,state,w)
        f=DrainageFrame(deepcopy(state),copy(w.kw),copy(w.kg),row,
                        frame_seal(identity,row,state,w))
        push!(rows,row);push!(frames,f)
        println("DRAINAGE_CACHE sample=",j,"/100");flush(stdout)
    end
    c=DrainageCache(identity,m.n,frames,rows,bytes)
    for j in eachindex(rows);checked_frame(c,j,identity);end
    c
end

function restore_drainage(c,m,j,identity)
    @assert m.n==c.cells && m.pressures[j]==c.rows[j].pressure_pa
    f=checked_frame(c,j,identity)
    s=deepcopy(f.state);w=Pc.Workspace(m.n)
    o=Pc.observe(m,s,w)
    @assert drainage_row(j,m.pressures[j],f.row.events,o,s,w)==c.rows[j]
    @assert reinterpret(UInt8,w.kw)==reinterpret(UInt8,f.kw)
    @assert reinterpret(UInt8,w.kg)==reinterpret(UInt8,f.kg)
    s,w
end

function verify_forward_reference(c,m,paths,identity)
    # Qualification only: independent forward replay, not an adaptive query replay.
    s=Pc.initial(m.n);w=Pc.Workspace(m.n);start=time_ns()
    for (j,p) in enumerate(m.pressures)
        f=checked_frame(c,j,identity)
        ne=Pc.advance_pc!(m,s,p,w,paths);o=Pc.observe(m,s,w)
        @assert drainage_row(j,p,ne,o,s,w)==c.rows[j]
        @assert isequal(s.breakthrough,f.state.breakthrough) && s.pressure==f.state.pressure
        @assert all(reinterpret(UInt8,a)==reinterpret(UInt8,b) for (a,b) in
                    zip((s.sg,s.invaded,s.trapped,w.kw,w.kg),
                        (f.state.sg,f.state.invaded,f.state.trapped,f.kw,f.kg)))
        println("REFERENCE_MATCH sample=",j,"/100");flush(stdout)
    end
    (time_ns()-start)/1e9
end
