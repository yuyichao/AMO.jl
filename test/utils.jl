#

module TestUtils

using AMO.Utils: ThreadObjectPool, eachobj

using Test
using Base.Threads

println("Testing with $(nthreads()) threads.")

@testset "ThreadObjectPool" begin
    p = ThreadObjectPool() do
        return Ref(0)
    end
    N = 10000
    @threads :greedy for i in 1:N
        o = get(p)
        o[] += 1
        put!(p, o)
    end
    @test sum(v[] for v in eachobj(p)) == N
    @threads :greedy for i in 1:N * 2
        o = get(p)
        o[] += 1
        put!(p, o)
    end
    @test sum(v[] for v in eachobj(p)) == 3N
end

end
