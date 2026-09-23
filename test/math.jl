#

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

@testset "Imaginary" begin
    Im = M.Imaginary
    for v in (0.0, 1.0, -2.5, 3), z in (1.0 + 2.0im, -0.3 + 0.0im, 0.0 - 4im, 2 + 3im)
        a = Im(v)
        ref = v * im
        @test real(a) == 0
        @test imag(a) == v
        @test iszero(a) == iszero(v)
        @test Complex(a) == ref
        @test ComplexF64(a) == ref && ComplexF64(a) isa ComplexF64
        @test a * z == ref * z
        @test z * a == z * ref
        @test a * 2.5 == Im(v * 2.5)
        @test 2.5 * a == Im(v * 2.5)
        @test a * 2 == Im(v * 2)
        @test a * true == Im(v)
        @test iszero(a * false)
        @test true * a == Im(v)
        @test iszero(false * a)
        @test a / 2 == Im(v / 2)
        @test a * Im(3.0) == -(v * 3.0)
        @test -a == Im(-v)
        @test exp(a) ≈ exp(ref)
        @test promote(a, 1.0) == (ref, 1.0)
        @test promote(a, 1.0im) == (ref, 1.0im)
    end
    @test zero(Im(1.0)) == Im(0.0)
    @test zero(Im{Float64}) == Im(0.0)
    @test sprint(show, Im(1.5)) == "Imaginary(1.5)"
end

using LinearAlgebra
using Random

@testset "heevd!" begin
    rng = Xoshiro(99)
    alloc_heevd!(ws, A) = @allocated M.heevd!(ws, A)
    for T in (ComplexF64, ComplexF32), n in (1, 2, 3, 5, 12)
        ws = M.HermEigenWorkspace{T}(n)
        for _ in 1:5
            A = Matrix(Hermitian(randn(rng, T, n, n)))
            λ, V = M.heevd!(ws, A)
            @test λ isa Vector{real(T)}
            @test V isa Matrix{T}
            @test issorted(λ)
            @test V * Diagonal(λ) * V' ≈ A
            @test V' * V ≈ I
            E = eigen(Hermitian(A))
            @test λ ≈ E.values
            # Only the upper triangle is accessed
            B = copy(A)
            B[2:end, 1] .= 0
            λ2, _ = M.heevd!(ws, B)
            @test λ2 ≈ E.values
        end
        A = Matrix(Hermitian(randn(rng, T, n, n)))
        alloc_heevd!(ws, A)
        @test alloc_heevd!(ws, A) == 0
    end
end
