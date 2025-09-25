#

module TestTrap

using Test
using Base.Iterators

using AMO: Trap

@testset "Lamb-Dicke" begin
    @test_throws ArgumentError Trap.η(0.1)
    @test_throws ArgumentError Trap.η(0.1, fm=0.1)
    @test_throws ArgumentError Trap.η(0.1, ωm=0.1)
    @test_throws ArgumentError Trap.η(0.1, kp=0.1)
    @test_throws ArgumentError Trap.η(0.1, λp=0.1)
    @test_throws ArgumentError Trap.η(0.1, νp=0.1)

    @test_throws ArgumentError Trap.η(0.1, fm=0.1, ωm=0.1, kp=0.1)
    @test_throws ArgumentError Trap.η(0.1, fm=0.1, kp=0.1, λp=0.1)

    m = 2.84e-25
    λ = 435e-9
    f = 124e3
    η_exp = 0.22297375295764688

    for kw1 in ((;fm=f), (;ωm=2π * f))
        for kw2 in ((;kp=2π / λ), (;λp=λ), (;νp=299792458 / λ))
            @test Trap.η(m; kw1..., kw2...) ≈ η_exp
        end
    end
end

function test_sideband(nmax, η)
    N = nmax * 2 + 10
    M = zeros(ComplexF64, N, N)
    for i in 1:N - 1
        M[i + 1, i] = M[i, i + 1] = im * η * sqrt(i)
    end
    M = exp(M)
    for n1 in 0:nmax
        for n2 in 0:nmax
            ele = M[n1 + 1, n2 + 1]
            @test Trap.sideband(n1, n2, η) ≈ ele atol=sqrt(eps()) rtol=sqrt(eps())
            @test Trap.sideband(n1, n2, η, phase=true) ≈ ele atol=sqrt(eps()) rtol=sqrt(eps())
            @test abs(Trap.sideband(n1, n2, η, phase=false)) ≈ abs(ele) atol=sqrt(eps()) rtol=sqrt(eps())

            if n1 > 100 || n2 > 100
                # Avoid laguerre polynomial overflow
                continue
            end
            s32 = Trap.sideband(n1, n2, Float32(η))
            @test s32 isa Complex{Float32}
            @test s32 ≈ ele atol=sqrt(eps(Float32)) rtol=sqrt(eps(Float32))
            s32 = Trap.sideband(n1, n2, Float32(η), phase=false)
            @test s32 isa Float32
            @test abs(s32) ≈ abs(ele) atol=sqrt(eps(Float32)) rtol=sqrt(eps(Float32))
        end
    end

    for Δn in -nmax:nmax
        n₋max = nmax - abs(Δn)
        if Δn < 0
            nstart = -Δn
            nend = nmax
        else
            nstart = 0
            nend = n₋max
        end
        it_c = Iterators.take(Trap.SidebandIter(Δn, η), n₋max + 1)
        it_r = Iterators.take(Trap.SidebandIter(Δn, η, phase=false), n₋max + 1)
        it_c32 = Iterators.take(Trap.SidebandIter(Δn, Float32(η)), n₋max + 1)
        it_r32 = Iterators.take(Trap.SidebandIter(Δn, Float32(η), phase=false), n₋max + 1)
        @test eltype(it_c) == ComplexF64
        @test eltype(it_r) == Float64
        @test eltype(it_c32) == ComplexF32
        @test eltype(it_r32) == Float32

        vc = collect(it_c)
        vr = collect(it_r)
        vc32 = collect(it_c32)
        vr32 = collect(it_r32)

        for (i, n1) in enumerate(nstart:nend)
            n2 = n1 + Δn
            ele = M[n1 + 1, n2 + 1]

            @test vc[i] ≈ ele atol=sqrt(eps()) rtol=sqrt(eps())
            @test abs(vr[i]) ≈ abs(ele) atol=sqrt(eps()) rtol=sqrt(eps())
            if n1 > 100 || n2 > 100
                # Avoid laguerre polynomial overflow
                continue
            end
            @test vc32[i] ≈ ele atol=sqrt(eps(Float32)) rtol=sqrt(eps(Float32))
            @test abs(vr32[i]) ≈ abs(ele) atol=sqrt(eps(Float32)) rtol=sqrt(eps(Float32))
        end
    end
end

@testset "Sideband" begin
    for n1 in 0:10
        for n2 in 0:10
            @test Trap.sideband(n1, n2, 0) == (n1 == n2)
        end
    end
    @test Trap.sideband(0, -1, 0.2) == 0
    @test Trap.sideband(-1, 0, 0.3) == 0
end

@testset "Sideband η=$η" for η in Number[0.1, 0.2, 0.5, 1, 1.2, 2.5]
    test_sideband(150, η)
end

@testset "ThermalPopulation nbar=$nbar" for nbar in Number[0, 0.1, 0.3, 1, 5.0, 10]
    nmax = ceil(Int, (nbar + 1) * 100)
    ps1 = [Trap.thermal_population(nbar, n) for n in 0:nmax]
    it = Trap.ThermalPopulationIter(nbar)
    ps2 = collect(Iterators.take(it, nmax + 1))
    it32 = Trap.ThermalPopulationIter(Float32(nbar))
    ps32 = collect(Iterators.take(it32, nmax + 1))

    @test eltype(it) == Float64
    @test eltype(it32) == Float32

    @test ps1 ≈ ps2
    @test ps32 ≈ ps2
    @test sum(ps1) ≈ 1
    @test sum(ps2) ≈ 1
    @test sum(ps32) ≈ 1
    ns = 0:nmax
    @test sum(ns .* ps1) ≈ nbar
    @test sum(ns .* ps2) ≈ nbar
    @test sum(ns .* ps32) ≈ nbar
end

end
