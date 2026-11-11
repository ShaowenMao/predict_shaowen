# Cross-language local-Pc preparation permits roundoff, but each engine must
# process exactly its own event schedule. An inactive threshold may fall on
# opposite sides of an output pressure without changing the physical state.
function event_identity_gate(m,reference)
    a=vcat(m.entry,m.endpoint);b=vcat(reference.entry,reference.endpoint)
    ia=sortperm(a);ib=sortperm(b)
    @assert ia==ib "Event identity/order changed"
    @assert (diff(a[ia]).==0)==(diff(b[ib]).==0) "Event grouping changed"
    true
end

samebits(a::Float64,b::Float64)=reinterpret(UInt64,a)==reinterpret(UInt64,b)
function event_schedule_gate(m,reference,previous,pc,actual,reference_count,
                             before,reference_before,state,reference_state,paths,reference_paths)
    own=searchsortedlast(m.events,pc)-searchsortedlast(m.events,previous)
    other=searchsortedlast(reference.events,pc)-searchsortedlast(reference.events,previous)
    @assert actual==own "Julia did not process its exact native event schedule"
    @assert reference_count==other "MATLAB event-count metadata disagrees with its model"
    reports=Dict{String,Any}[]
    for kind in (:entry,:endpoint)
        a=getfield(m,kind);b=getfield(reference,kind)
        for i in eachindex(a)
            ((previous<a[i]<=pc)==(previous<b[i]<=pc)) && continue
            @assert kind==:endpoint "Gas-entry event moved across a sample boundary"
            @assert isfinite(a[i]) && isfinite(b[i]) && min(a[i],b[i])>0
            @assert nextfloat(min(a[i],b[i]))==max(a[i],b[i]) "Endpoint-bin difference exceeds one ULP"
            @assert paths!==nothing && reference_paths!==nothing "Inactive endpoint requires a proved path-loss threshold"
            @assert before.trapped[i]!=0 && reference_before.trapped[i]!=0 "Endpoint was not already trapped"
            @assert before.invaded[i]!=0 && reference_before.invaded[i]!=0
            @assert state.trapped[i]!=0 && reference_state.trapped[i]!=0
            @assert samebits(before.sg[i],state.sg[i]) && samebits(reference_before.sg[i],reference_state.sg[i]) "Inactive cell saturation changed"
            @assert max(paths.loss[i],reference_paths.loss[i])<=previous
            @assert max(paths.loss[i],reference_paths.loss[i])<min(a[i],b[i]) "Endpoint can still close a drainage path"
            push!(reports,Dict("gridblock"=>i,"kind"=>"already_trapped_endpoint",
                "previous_pressure_pa"=>previous,"output_pressure_pa"=>pc,
                "julia_endpoint_pa"=>a[i],"matlab_endpoint_pa"=>b[i],
                "endpoint_difference_pa"=>a[i]-b[i],"endpoint_difference_ulp"=>1,
                "julia_path_loss_pa"=>paths.loss[i],"matlab_path_loss_pa"=>reference_paths.loss[i],
                "julia_sg"=>state.sg[i],"matlab_sg"=>reference_state.sg[i],
                "julia_events"=>actual,"matlab_events"=>reference_count,
                "native_schedules_exact"=>true,"state_inactive_proved"=>true))
        end
    end
    @assert actual==reference_count || !isempty(reports) "Unexplained event-count difference"
    reports
end
