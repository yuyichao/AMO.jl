#

module TestRand

using AMO: g_sum, g_s, g_l

using Test

@testset "g-factor" begin
    @test g_s ≈ 2 + 1 / 137 / π atol=0.5e-4
    @test g_l == 1

    @test g_sum(2, 1/2, 2, 3/2, 0) == 1/2
    @test g_sum(1, 1/2, 2, 3/2, 0) == -1/2
    @test g_sum(2, 3/2, 0, 1/2, 2) == 1/2
    @test g_sum(1, 3/2, 0, 1/2, 2) == -1/2

    @test g_sum(3/2, 1, 1, 1/2, 2) ≈ 4/3
    @test g_sum(3/2, 1/2, 2, 1, 1) ≈ 4/3
end

end
