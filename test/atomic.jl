#

using AMO: g_sum, g_s, g_l, Atomic

using Test
using HalfIntegers
using RationalRoots
using WignerSymbols

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

function coupling_kernel(j0, m0, j1, m1, j2, m2, j1′, m1′, j2′, m2′, k, q)
    return clebschgordan(Float64, j1, m1, j0, m0, j1′, m1′) *
        clebschgordan(Float64, j2, m2, j0, m0, j2′, m2′) *
        clebschgordan(Float64, j2, m2, k, q, j1, m1)
end

function coupling0(j0, j1, j2, j1′, m1′, j2′, m2′, k, q)
    return sum(coupling_kernel(j0, m0, j1, m1, j2, m2, j1′, m1′, j2′, m2′, k, q)
               for m0 in -j0:j0, m1 in -j1:j1, m2 in -j2:j2; init=0.0)
end

function check_coupling_js(j0, j1, j2, k)
    has_nonzero = false
    vtotal = 0.0
    for j1′ in abs(j1 - j0):(j1 + j0)
        for j2′ in abs(j2 - j0):(j2 + j0)
            vbig = Atomic.couple_reduced_element(j1′, j2′, j0, j1, j2, k)
            @test vbig isa RationalRoot{BigInt}
            vf = Atomic.couple_reduced_element(Float64, j1′, j2′, j0, j1, j2, k)
            @test vf isa Float64
            if abs(j1′ - j2′) > k || j1′ + j2′ < k
                @test vbig == 0
                @test vf == 0
                continue
            end
            @test vf ≈ vbig atol=sqrt(eps())
            vtotal_j = 0.0
            for m1′ in -j1′:j1′
                for m2′ in -j2′:j2′
                    q = m1′ - m2′
                    if q < -k || q > k
                        continue
                    end
                    v0 = coupling0(j0, j1, j2, j1′, m1′, j2′, m2′, k, q)
                    v1 = vf * clebschgordan(Float64, j2′, m2′, k, q, j1′, m1′)
                    @test v0 ≈ v1 atol=sqrt(eps())
                    has_nonzero |= !(isapprox(v0, 0, atol=sqrt(eps())))
                    vtotal_j += abs2(v1)
                end
            end
            @test vtotal_j ≈ abs2(vf) * (2 * j1′ + 1)
            vtotal += vtotal_j
        end
    end
    @test has_nonzero
    @test vtotal ≈ (2 * j1 + 1) * (2 * j0 + 1)
end

function check_coupling(jmax, kmax)
    for dk in 0:twice(kmax)
        k = half(dk)
        for dj0 in 0:twice(jmax)
            j0 = half(dj0)
            for dj1 in 0:twice(jmax)
                j1 = half(dj1)
                for dj2 in 0:twice(jmax)
                    j2 = half(dj2)
                    if abs(j1 - j2) > k || j1 + j2 < k || (dj1 - dj2 - dk) % 2 != 0
                        continue
                    end
                    check_coupling_js(j0, j1, j2, k)
                end
            end
        end
    end
end

@testset "Reduced matrix element" begin
    check_coupling(4, 2)
end
