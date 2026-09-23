#

import AMO.TimeSequence as TS

using Test
using Random
using LinearAlgebra
using StaticArrays
using QuantumToolbox
using SciMLOperators: MatrixOperator

const MT = Matrix{ComplexF64}
const σx = sigmax()
const σy = sigmay()
const σz = sigmaz()
# Tight solver tolerances so that the results can be compared precisely
const tols = (; reltol=1e-10, abstol=1e-12)

function fd_gradient(step, p, δ=1e-4)
    OP = typeof(step).parameters[1]
    grads = OP[]
    for k in 1:length(p)
        pp = copy(p)
        pp[k] += δ
        TS.set_params!(step, pp)
        Up = copy(TS.compute(step, OP[]))
        pm = copy(p)
        pm[k] -= δ
        TS.set_params!(step, pm)
        Um = copy(TS.compute(step, OP[]))
        push!(grads, (Up - Um) / (2δ))
    end
    TS.set_params!(step, p)
    return grads
end

@testset "QobjEvoStep constant operator" begin
    H = 0.3 * σz + 0.7 * σx
    tspan = (0.2, 1.5)
    Uref = exp(-im * (tspan[2] - tspan[1]) * Matrix(H.data))
    step = TS.QobjEvoStep{MT}(H; nparams=0, tspan=tspan, tols...)
    @test TS.nparams(step) == 0
    @test TS.support_inplace_compute(typeof(step))
    @test TS.get_init(step)() isa MT
    @test iszero(TS.get_init(step)())
    TS.set_params!(step, Float64[])
    @test TS.compute(step, MT[]) ≈ Uref rtol=1e-8
    res = zeros(ComplexF64, 2, 2)
    @test TS.compute!(res, step, MT[]) === res
    @test res ≈ Uref rtol=1e-8

    # Default matrix type
    step = TS.QobjEvoStep(H; nparams=0, tspan=tspan, tols...)
    @test step isa TS.QobjEvoStep{MT}
    @test TS.compute(step, MT[]) ≈ Uref rtol=1e-8

    # Static matrix type
    SOP = SMatrix{2,2,ComplexF64,4}
    step = TS.QobjEvoStep{SOP}(H; nparams=0, tspan=tspan, tols...)
    @test !TS.support_inplace_compute(typeof(step))
    @test TS.get_init(step)() isa SOP
    U = TS.compute(step, SOP[])
    @test U isa SOP
    @test U ≈ Uref rtol=1e-8
    @test_throws ArgumentError TS.compute!(SOP(zeros(2, 2)), step, SOP[])

    # Constant QobjEvo, with unused parameters
    step = TS.QobjEvoStep{MT}(QobjEvo(H); nparams=2, tspan=tspan, tols...)
    TS.set_params!(step, [1.0, 2.0])
    grads = [zeros(ComplexF64, 2, 2) for _ in 1:2]
    @test TS.compute(step, grads) ≈ Uref rtol=1e-8
    @test all(g->norm(g) < 1e-9, grads)
end

@testset "QobjEvoStep driven qubit [$OP]" for OP in (MT, SMatrix{2,2,ComplexF64,4})
    H = QobjEvo((σz / 2, (σx / 2, (p, t)->p[1] * cos(p[2] * t)), (σy / 2, (p, t)->p[3])))
    p = [0.8, 1.3, 0.4]
    T = 2.0
    step = TS.QobjEvoStep{OP}(H; nparams=3, tspan=(0, T), tols...)
    @test TS.nparams(step) == 3
    TS.set_params!(step, p)
    grads = [OP(zeros(2, 2)) for _ in 1:3]
    U = TS.compute(step, grads)
    @test U isa OP
    @test U' * U ≈ I

    ψ0 = basis(2, 0)
    sol = sesolve(H, ψ0, [0, T]; params=p, progress_bar=Val(false), tols...)
    @test U * ψ0.data ≈ sol.states[end].data rtol=1e-8

    for (g, gfd) in zip(grads, fd_gradient(step, p))
        @test g isa OP
        @test g ≈ gfd rtol=1e-5
    end

    # Without gradient the result should be the same
    @test TS.compute(step, OP[]) ≈ U rtol=1e-8

    if OP <: Matrix
        res = zeros(ComplexF64, 2, 2)
        grads2 = [zeros(ComplexF64, 2, 2) for _ in 1:3]
        grads2_ids = copy(grads2)
        @test TS.compute!(res, step, grads2) === res
        @test res ≈ U
        for (g1, g2, gid) in zip(grads, grads2, grads2_ids)
            @test g2 === gid
            @test g1 ≈ g2
        end
    end
end

@testset "QobjEvoStep NamedTuple parameters" begin
    H = QobjEvo((σz / 2, (σx / 2, (p, t)->p.Ω * cos(p.ω * t))))
    pnt = (Ω=0.8, ω=1.3)
    p = [pnt.Ω, pnt.ω]
    T = 1.5
    step = TS.QobjEvoStep{MT}(H; nparams=2, tspan=(0, T), param_map=p->(Ω=p[1], ω=p[2]),
                              tols...)
    TS.set_params!(step, p)
    grads = [zeros(ComplexF64, 2, 2) for _ in 1:2]
    U = TS.compute(step, grads)
    ψ0 = basis(2, 0)
    sol = sesolve(H, ψ0, [0, T]; params=pnt, progress_bar=Val(false), tols...)
    @test U * ψ0.data ≈ sol.states[end].data rtol=1e-8
    for (g, gfd) in zip(grads, fd_gradient(step, p))
        @test g ≈ gfd rtol=1e-5
    end
end

@testset "QobjEvoStep complex coefficients" begin
    # Driven oscillator: H = ω a†a + f(t) a + conj(f(t)) a†
    N = 4
    a = destroy(N)
    f(p, t) = p[1] * exp(im * p[2] * t)
    H = QobjEvo((0.7 * a' * a, (a, f), (a', (p, t)->conj(f(p, t)))))
    p = [0.5, 1.1]
    T = 1.2
    step = TS.QobjEvoStep{MT}(H; nparams=2, tspan=(0, T), tols...)
    TS.set_params!(step, p)
    grads = [zeros(ComplexF64, N, N) for _ in 1:2]
    U = TS.compute(step, grads)
    @test U' * U ≈ I
    ψ0 = basis(N, 0)
    sol = sesolve(H, ψ0, [0, T]; params=p, progress_bar=Val(false), tols...)
    @test U * ψ0.data ≈ sol.states[end].data rtol=1e-8
    for (g, gfd) in zip(grads, fd_gradient(step, p))
        @test g ≈ gfd rtol=1e-5
    end
end

@testset "QobjEvoStep Liouvillian" begin
    H = QobjEvo((σz / 2, (σx / 2, (p, t)->p[1] * cos(p[2] * t))))
    c_ops = [sqrt(0.3) * sigmam(), sqrt(0.1) * σz]
    L = liouvillian(H, c_ops)
    @test issuper(L)
    p = [0.9, 1.7]
    T = 1.8
    step = TS.QobjEvoStep{MT}(L; nparams=2, tspan=(0, T), tols...)
    @test step.n == 4
    TS.set_params!(step, p)
    grads = [zeros(ComplexF64, 4, 4) for _ in 1:2]
    UL = TS.compute(step, grads)
    ρ0 = ket2dm((basis(2, 0) + basis(2, 1)) / sqrt(2))
    sol = mesolve(H, ρ0, [0, T], c_ops; params=p, progress_bar=Val(false), tols...)
    @test UL * mat2vec(ρ0.data) ≈ mat2vec(sol.states[end].data) rtol=1e-8
    for (g, gfd) in zip(grads, fd_gradient(step, p))
        @test g ≈ gfd rtol=1e-5
    end
    # Constant Liouvillian
    Lc = liouvillian(0.5 * σz, c_ops)
    stepc = TS.QobjEvoStep{MT}(Lc; nparams=0, tspan=(0, T), tols...)
    TS.set_params!(stepc, Float64[])
    @test TS.compute(stepc, MT[]) ≈ exp(T * Matrix(Lc.data)) rtol=1e-8
end

@testset "QobjEvoStep Sequence" begin
    H1 = QobjEvo((σz / 2, (σx / 2, (p, t)->p[1] * cos(p[2] * t))))
    H2 = 0.4 * σy
    step1 = TS.QobjEvoStep{MT}(H1; nparams=2, tspan=(0, 1.0), tols...)
    step2 = TS.QobjEvoStep{MT}(H2; nparams=0, tspan=(0, 0.7), tols...)
    step3 = TS.ConstMatrixStep{MT}((Matrix(σx.data),); param_t=true)
    for kws in ((;), (; mul=nothing, mul! = LinearAlgebra.mul!))
        s = TS.Sequence{MT}((step1, step2, step3); kws...)
        @test TS.nparams(s) == 4
        p = [0.6, 1.2, 0.3, 0.5]
        TS.set_params!(s, p)
        grads = [zeros(ComplexF64, 2, 2) for _ in 1:4]
        U = copy(TS.compute(s, grads))
        TS.set_params!(step1, p[1:2])
        TS.set_params!(step3, p[3:4])
        Uref = TS.compute(step1, MT[]) * TS.compute(step2, MT[]) * TS.compute(step3, MT[])
        @test U ≈ Uref rtol=1e-8
        for (g, gfd) in zip(grads, fd_gradient(s, p))
            @test g ≈ gfd rtol=1e-5
        end
    end
end

@testset "QobjEvoStep errors" begin
    # Composed operators are not supported
    H = QobjEvo(MatrixOperator(Matrix(σx.data)) * MatrixOperator(Matrix(σz.data)))
    @test_throws ArgumentError TS.QobjEvoStep{MT}(H; nparams=0, tspan=(0, 1))
    # Time-dependent MatrixOperator is not supported
    H = QobjEvo(MatrixOperator(Matrix(σx.data); update_func=(A, u, p, t)->A))
    @test_throws ArgumentError TS.QobjEvoStep{MT}(H; nparams=0, tspan=(0, 1))
    # Real matrix type
    @test_throws ArgumentError TS.QobjEvoStep{Matrix{Float64}}(σx; nparams=0, tspan=(0, 1))
    # Kets are not operators
    @test_throws ArgumentError TS.QobjEvoStep{MT}(basis(2, 0); nparams=0, tspan=(0, 1))
    # Parameter count
    step = TS.QobjEvoStep{MT}(σx; nparams=1, tspan=(0, 1))
    @test_throws AssertionError TS.set_params!(step, [1.0, 2.0])
    @test_throws AssertionError TS.compute(step, [zeros(ComplexF64, 2, 2) for _ in 1:2])
    @test_throws AssertionError TS.compute!(zeros(ComplexF64, 2, 2), step,
                                            [zeros(ComplexF64, 2, 2) for _ in 1:2])
end
