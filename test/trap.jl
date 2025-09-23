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

end
