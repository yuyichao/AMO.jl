#

module TestTrap

using Test
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
end

@testset "Sideband" begin
    test_sideband(150, 0.1)
    test_sideband(150, 0.2)
    test_sideband(150, 0.5)
    test_sideband(150, 1.2)
    test_sideband(150, 2.5)
end

end
