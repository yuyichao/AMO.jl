#

module TestUtils

using AMO.Utils: ThreadObjectPool, ObjectPool, eachobj

using Test
using Base.Threads

println("Testing with $(nthreads()) threads.")

mutable struct Counter{T}
    v::T
    Counter{T}() where T = new{T}(0)
end
Base.getindex(c::Counter) = c.v
Base.setindex!(c::Counter, v) = (c.v = v)

@testset "$(Pool)" for Pool in [ThreadObjectPool, ObjectPool]
    for cb in [()->Ref(0), Counter{Int}]
        p = @inferred Pool(cb)
        @test fieldtype(typeof(p), :cb) != DataType
        N = 10000
        @threads :greedy for i in 1:N
            o = @inferred get(p)
            o[] += 1
            put!(p, o)
        end
        @test sum(v[] for v in eachobj(p)) == N
        @threads :greedy for i in 1:N * 2
            o = @inferred get(p)
            o[] += 1
            put!(p, o)
        end
        @test sum(v[] for v in eachobj(p)) == 3N
        empty!(p)
        @test sum((v[] for v in eachobj(p)), init=0) == 0
    end
end

end
