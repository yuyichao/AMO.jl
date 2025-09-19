#

module TestMath

using Test

import AMO.Math as M

function test_assoc_laguerre(x, α, nmax)
    bx = big(x)
    bα = big(α)
    bα_1 = bα - 1
    L0 = big(1.0)
    L1 = 1 + bα - bx
    @test M.assoc_laguerre(x, 0, α) ≈ Float64(L0)
    @test M.assoc_laguerre(x, 1, α) ≈ Float64(L1)

    L_2 = L0
    L_1 = L1
    for n in 2:nmax
        L = ((2n + bα_1 - bx) * L_1 - (n + bα_1) * L_2) / n
        @test M.assoc_laguerre(x, n, α) ≈ L atol=2e-7 rtol=1e-4
        L_2, L_1 = L_1, L
    end
end

@testset "Laguerre" begin
    @test M.assoc_laguerre(1.0, -1, 0.0) == 0
    @test M.assoc_laguerre(NaN, -1, NaN) == 0
    @test M.assoc_laguerre(NaN, 0, NaN) == 1
    @test isnan(M.assoc_laguerre(NaN, 1, 0.0))
    @test isnan(M.assoc_laguerre(0.0, 1, NaN))

    test_assoc_laguerre(0.0, 0.0, 10)

    for v in (0.1, 0.6, 1.0, 1.2, 2.3)
        test_assoc_laguerre(v, 0.0, 10000)
        test_assoc_laguerre(0.0, v, 10000)
        test_assoc_laguerre(v, v, 10000)
        test_assoc_laguerre(-v, 0.0, 10000)
        test_assoc_laguerre(0.0, -v, 10000)
        test_assoc_laguerre(-v, -v, 10000)
    end
end

end
