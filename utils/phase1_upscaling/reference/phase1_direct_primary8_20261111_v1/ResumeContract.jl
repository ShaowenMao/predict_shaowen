module ResumeContract
export Ledger, take_saved!, finished
mutable struct Ledger
    rows::Vector
    next::Int
end
Ledger(rows) = Ledger(rows,1)
function take_saved!(ledger,j,phase)
    ledger.next > length(ledger.rows) && return nothing
    row=ledger.rows[ledger.next]
    @assert row["sample"]==j && row["phase"]==phase "Saved-call prefix diverged; never recompute"
    @assert isfinite(row["q"]) && row["q"]>=0
    ledger.next+=1
    row
end
finished(ledger)=ledger.next==length(ledger.rows)+1
end
