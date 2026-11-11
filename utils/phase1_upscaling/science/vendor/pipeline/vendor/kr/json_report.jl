function json_value(io,x::AbstractString)
    print(io,'"')
    for c in x
        if c=='"' || c=='\\';print(io,'\\',c)
        elseif UInt32(c)<0x20;print(io,"\\u",string(UInt32(c);base=16,pad=4))
        else;print(io,c)
        end
    end
    print(io,'"')
end
json_value(io,x::Bool)=print(io,x ? "true" : "false")
function json_value(io,x::Real)
    @assert isfinite(x);print(io,x)
end
function json_value(io,x::AbstractVector)
    print(io,'[')
    for (i,v) in enumerate(x);i>1 && print(io,',');json_value(io,v);end
    print(io,']')
end
function json_value(io,x::AbstractDict)
    print(io,'{')
    for (i,k) in enumerate(sort!(collect(keys(x))))
        i>1 && print(io,',');json_value(io,k);print(io,':');json_value(io,x[k])
    end
    print(io,'}')
end
