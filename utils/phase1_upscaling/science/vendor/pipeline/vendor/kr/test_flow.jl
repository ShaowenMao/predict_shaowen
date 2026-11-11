include("PhaseFlow.jl")
using .PhaseFlow,Test,Random,LinearAlgebra,SparseArrays
bridge!(ARGS[1]);Random.seed!(14921)
@testset "Analytic and disconnected controls" begin
    for n in (2,3,9,20), contrast in (1.0,1e-3,1e-8,1e-14)
        e=hcat(collect(1:n-1),collect(2:n));net=Net(n,e,fill(2.,n-1),fill(2.,n-1),[1],[n],[2.],[2.])
        k=fill(contrast,n);k[1]=1.0
        q,info=conductance(net,k);wanted=1/sum(1.0./k)
        @test abs(q-wanted)/wanted<=1e-9
        @test info["metrics"][5]<=1e-5
        k[end]=0;q,info=conductance(net,k);@test q==0
    end
    net=Net(3,[1 2;2 3],[2.,2.],[2.,2.],[1],[3],[2.],[2.])
    q,info=conductance(net,[1.,.01,1.];force_repair=true)
    @test info["fallback_used"]&&abs(q-1/102)/(1/102)<1e-9
end
@testset "Independent connectivity and algebra controls" begin
    for trial in 1:80
        n=12;e=hcat(rand(1:n,30),rand(1:n,30));net=Net(n,e,ones(30),ones(30),[1,2],[11,12],[1.,1.],[1.,1.])
        kr=rand(n);kr[rand(1:n,3)].=0
        p,A,b,keep,_=prepare(net,kr)
        # Independent Boolean transitive closure, including isolated nodes.
        reach=Matrix{Bool}(I,n,n)
        for k in axes(e,1)
            u,v=e[k,1],e[k,2]
            if kr[u]>0&&kr[v]>0;reach[u,v]=reach[v,u]=true;end
        end
        for k=1:n,i=1:n,j=1:n;reach[i,j]|=reach[i,k]&&reach[k,j];end
        incoming=net.ci[kr[net.ci].>0];outgoing=net.co[kr[net.co].>0]
        expected=[any(reach[i,j] for j in incoming)&&any(reach[i,j] for j in outgoing) for i=1:n]
        @test keep==expected
        if p.n>0
            x=rand(p.n);y=apply_operator(p,x)
            @test norm(y-(Diagonal(p.scale)*A*Diagonal(p.scale))*x)<=1e-12*max(norm(y),1.)
        end
    end
end
if length(ARGS)>1
    @testset "Unchanged MATLAB binary128 kernel equivalence" begin
        open(ARGS[2],"r") do io
            @test read(io,Int64)==2026090903;count=read(io,Int64);@test count==32
            for _=1:count
                n,ne,ni,no=readvec(io,Int64,4)
                p=Parts(n,readvec(io,Float64,ne),readvec(io,Float64,ne),readvec(io,Float64,ne),
                    readvec(io,Float64,ni),readvec(io,Float64,no),readvec(io,Float64,ni),readvec(io,Float64,no),readvec(io,Float64,n))
                x=reshape(readvec(io,Float64,2n),n,2);dx=readvec(io,Float64,n);v=readvec(io,Float64,n)
                r=readvec(io,Float64,n);m=readvec(io,Float64,8);added=reshape(readvec(io,Float64,2n),n,2);applied=readvec(io,Float64,n)
                ra,ma=audit(p,x)
                @test reinterpret(UInt64,ra)==reinterpret(UInt64,r)
                @test reinterpret(UInt64,ma)==reinterpret(UInt64,m)
                @test reinterpret(UInt64,pressure_add(x,dx))==reinterpret(UInt64,added)
                @test reinterpret(UInt64,apply_operator(p,v))==reinterpret(UInt64,applied)
            end
            @test eof(io)
        end
    end
end
