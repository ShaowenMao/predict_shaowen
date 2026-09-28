include("PhaseFlow.jl");include("json_report.jl")
using .PhaseFlow,LinearAlgebra,SparseArrays
BLAS.set_num_threads(2)
function report(path,data)
    @assert !isfile(path);open(path,"w") do io;json_value(io,data);println(io);end
end
function load_packet(path)
    open(path,"r") do io
        magic,core,n,ne,ni,no,nu,np=readvec(io,Int64,8);@assert magic==2026090902&&np==100
        len,area,qa=readvec(io,Float64,3)
        e=reshape(readvec(io,Int64,2ne),ne,2);h1=readvec(io,Float64,ne);h2=readvec(io,Float64,ne)
        ci=readvec(io,Int64,ni);co=readvec(io,Int64,no);hi=readvec(io,Float64,ni);ho=readvec(io,Float64,no)
        net=Net(n,e,h1,h2,ci,co,hi,ho)
        pc=readvec(io,Float64,np);expected=reshape(readvec(io,Float64,3np),np,3)
        owner=readvec(io,Int64,np);sample=readvec(io,Int64,nu)
        kw=reshape(readvec(io,Float64,n*nu),n,nu);kg=reshape(readvec(io,Float64,n*nu),n,nu)
        @assert eof(io)&&all(1 .<=owner.<=nu)&&all(diff(sample).>0)&&isfinite(qa)&&qa>0
        (;core,net,len,area,qa,pc,expected,owner,sample,kw,kg)
    end
end
function verify_parts(P,prep,out)
    checks=0
    open(joinpath(prep,"parts_checks.bin"),"r") do io
        n=read(io,Int64);@assert n==2length(P.sample)+1
        for _=1:n
            sample,phase,nn=readvec(io,Int64,3);wanted=bytes2hex(readvec(io,UInt8,32))
            if phase==0;@assert sample==0;kr=ones(P.net.n)
            else
                k=findfirst(==(sample),P.sample);@assert k!==nothing&&phase in (1,2)
                kr=phase==1 ? P.kw[:,k] : P.kg[:,k]
            end
            p,_,_,keep,_=prepare(P.net,kr)
            @assert p.n==nn&&parts_hash(p,keep)==wanted "MATLAB conservative operator differs"
            checks+=1
        end
        @assert eof(io)
    end
    report(joinpath(out,"assembly_checks.json"),Dict("core"=>P.core,"checks"=>checks,"all_parts_bitwise_equal"=>true))
    println("PASS ",checks," conservative operator fingerprints")
end
function run(prep,lib,out,mode,repetition)
    start=time_ns();bridge!(lib);P=load_packet(joinpath(prep,"phase_packet.bin"))
    if mode=="check";verify_parts(P,prep,out);return;end
    @assert mode=="run"
    small=Net(3,[1 2;2 3],[2.,2.],[2.,2.],[1],[3],[2.],[2.])
    conductance(small,[1.,.01,1.]);conductance(small,[1.,0.,1.]);conductance(small,[1.,.01,1.];force_repair=true)
    setup_seconds=(time_ns()-start)/1e9
    report(joinpath(out,"runtime.json"),Dict("julia"=>string(VERSION),"cpu"=>Sys.CPU_NAME,
        "threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),"blas"=>string(BLAS.get_config()),
        "iterative_solvers"=>string(Base.pkgversion(PhaseFlow.IterativeSolvers)),
        "setup_and_compilation_seconds"=>setup_seconds,"umfpack_iterative_refinement_steps"=>2))
    @assert !isfile(joinpath(out,"calls.jsonl"));Q=zeros(length(P.sample),2);phase_seconds=0.0;absolute_seconds=0.0
    open(joinpath(out,"calls.jsonl"),"w") do io
        function one(kr,sample,phase,wanted)
            began=time_ns()
            try
                q,info=conductance(P.net,kr);seconds=(time_ns()-began)/1e9
                relative=abs(q-wanted)/max(abs(q),abs(wanted),floatmin(Float64))
                matched=(q==0)==(wanted==0)&&relative<=1e-5
                row=Dict("sample"=>sample,"phase"=>phase,"q"=>q,"seconds"=>seconds,"accepted"=>true,
                    "reference_matches"=>matched,"relative_reference_error"=>relative,"info"=>info)
                json_value(io,row);println(io);flush(io)
                @assert matched "Reference conductance mismatch; no timing accepted"
                println("JULIA core",P.core," rep",repetition," sample",sample," ",phase," ",seconds,"s match");flush(stdout)
                q,seconds
            catch e
                report(joinpath(out,"failure.json"),Dict("sample"=>sample,"phase"=>phase,"error"=>sprint(showerror,e)))
                rethrow()
            end
        end
        qa,absolute_seconds=one(ones(P.net.n),0,"absolute",P.qa)
        for k in eachindex(P.sample),phase=1:2
            sample=P.sample[k];kr=phase==1 ? P.kw[:,k] : P.kg[:,k];label=phase==1 ? "brine" : "gas"
            q,seconds=one(kr,sample,label,P.expected[sample,phase+1]*P.qa)
            Q[k,phase]=q;phase_seconds+=seconds
        end
    end
    curves=[Dict("sample"=>k,"pc"=>P.pc[k],"sg"=>P.expected[k,1],"krw"=>Q[P.owner[k],1]/P.qa,
                 "krg"=>Q[P.owner[k],2]/P.qa) for k in eachindex(P.pc)]
    report(joinpath(out,"curves.json"),Dict("curves"=>curves))
    report(joinpath(out,"summary.json"),Dict("core"=>P.core,"repeat"=>repetition,"fixed100_points"=>100,
        "unique_states"=>length(P.sample),"phase_seconds"=>phase_seconds,"absolute_seconds"=>absolute_seconds,
        "all_calls_accepted"=>true,"worker_seconds"=>(time_ns()-start)/1e9,"production_promotion"=>false))
end
run(ARGS[1],ARGS[2],ARGS[3],ARGS[4],length(ARGS)>4 ? parse(Int,ARGS[5]) : 0)
