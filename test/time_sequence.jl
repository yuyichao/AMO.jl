#

import AMO.TimeSequence as TS

using Test
using Random

mutable struct F64Ref
    v::Float64
end
TS.get_init(::Type{F64Ref}) = ()->F64Ref(0.0)
TS.get_mul(::Type{F64Ref}) = nothing
TS.get_mul!(::Type{F64Ref}) = (out::F64Ref, a::F64Ref, b::F64Ref)->(out.v = a.v * b.v)
Base.Float64(v::F64Ref) = v.v

mutable struct F64Ref2
    v::Float64
end
Base.Float64(v::F64Ref2) = v.v

mutable struct Step1{VT} <: TS.AbstractStep{VT,1}
    v1::Float64
    Step1{VT}() where VT = new()
end
function TS.set_params!(op::Step1, params)
    op.v1 = params[1]
    return
end
function TS.compute(op::Step1{VT}, grad) where VT
    if !isempty(grad)
        grad[1] = VT(1)
    end
    return VT(op.v1)
end
mutable struct Step2{VT} <: TS.AbstractStep{VT,2}
    v1::Float64
    v2::Float64
    Step2{VT}() where VT = new()
end
function TS.set_params!(op::Step2, params)
    op.v1 = params[1]
    op.v2 = params[2]
    return
end
function TS.compute(op::Step2{VT}, grad) where VT
    if !isempty(grad)
        grad[1] = VT(op.v2)
        grad[2] = VT(op.v1)
    end
    return VT(op.v1 * op.v2)
end
mutable struct Step3{VT} <: TS.AbstractStep{VT,3}
    v1::Float64
    v2::Float64
    v3::Float64
    Step3{VT}() where VT = new()
end
function TS.set_params!(op::Step3, params)
    op.v1 = params[1]
    op.v2 = params[2]
    op.v3 = params[3]
    return
end
function TS.compute(op::Step3{VT}, grad) where VT
    if !isempty(grad)
        grad[1] = VT(op.v2 * op.v3)
        grad[2] = VT(op.v1 * op.v3)
        grad[3] = VT(op.v1 * op.v2)
    end
    return VT(op.v1 * op.v2 * op.v3)
end

mutable struct IStep1{VT} <: TS.AbstractStep{VT,1}
    v1::Float64
    IStep1{VT}() where VT = new()
end
function TS.set_params!(op::IStep1, params)
    op.v1 = params[1]
    return
end
function TS.compute(op::IStep1{VT}, grad) where VT
    if !isempty(grad)
        grad[1] = VT(1)
    end
    return VT(op.v1)
end
function TS.compute!(res::VT, op::IStep1{VT}, grad) where VT
    if !isempty(grad)
        grad[1].v = 1
    end
    res.v = op.v1
    return res
end
TS.support_inplace_compute(::Type{<:IStep1}) = true
mutable struct IStep2{VT} <: TS.AbstractStep{VT,2}
    v1::Float64
    v2::Float64
    IStep2{VT}() where VT = new()
end
function TS.set_params!(op::IStep2, params)
    op.v1 = params[1]
    op.v2 = params[2]
    return
end
function TS.compute(op::IStep2{VT}, grad) where VT
    if !isempty(grad)
        grad[1] = VT(op.v2)
        grad[2] = VT(op.v1)
    end
    return VT(op.v1 * op.v2)
end
function TS.compute!(res::VT, op::IStep2{VT}, grad) where VT
    if !isempty(grad)
        grad[1].v = op.v2
        grad[2].v = op.v1
    end
    res.v = op.v1 * op.v2
    return res
end
TS.support_inplace_compute(::Type{<:IStep2}) = true
mutable struct IStep3{VT} <: TS.AbstractStep{VT,3}
    v1::Float64
    v2::Float64
    v3::Float64
    IStep3{VT}() where VT = new()
end
function TS.set_params!(op::IStep3, params)
    op.v1 = params[1]
    op.v2 = params[2]
    op.v3 = params[3]
    return
end
function TS.compute(op::IStep3{VT}, grad) where VT
    if !isempty(grad)
        grad[1] = VT(op.v2 * op.v3)
        grad[2] = VT(op.v1 * op.v3)
        grad[3] = VT(op.v1 * op.v2)
    end
    return VT(op.v1 * op.v2 * op.v3)
end
function TS.compute!(res::VT, op::IStep3{VT}, grad) where VT
    if !isempty(grad)
        grad[1].v = op.v2 * op.v3
        grad[2].v = op.v1 * op.v3
        grad[3].v = op.v1 * op.v2
    end
    res.v = op.v1 * op.v2 * op.v3
    return res
end
TS.support_inplace_compute(::Type{<:IStep3}) = true

function test_scalar_sequence(s::TS.AbstractStep{VT}) where VT
    NP = TS.nparams(typeof(s))
    @test TS.nparams(s) == NP
    if VT === Float64
        @test !TS.support_inplace_compute(typeof(s))
    else
        @test TS.support_inplace_compute(typeof(s)) == (length(s.steps) != 1 || TS.support_inplace_compute(typeof(s.steps[1])))
    end

    for _ in 1:10
        params = rand(NP) .+ 0.4
        TS.set_params!(s, params)
        @test Float64(TS.compute(s, VT[])) ≈ prod(params)
    end

    grads = [VT(0.0) for i in 1:NP]
    for _ in 1:100
        params = rand(NP) .+ 0.4
        TS.set_params!(s, params)
        if VT === Float64
            grads .= 0
        else
            for v in grads
                v.v = 0
            end
        end
        @test Float64(TS.compute(s, grads)) ≈ prod(params)
        for i in 1:NP
            @test Float64(grads[i]) ≈ prod((p for (pidx, p) in enumerate(params)
                                                 if pidx != i), init=1.0)
        end
    end

    if !TS.support_inplace_compute(typeof(s))
        return
    end

    for _ in 1:10
        params = rand(NP) .+ 0.4
        TS.set_params!(s, params)
        res = VT(0.0)
        TS.compute!(res, s, VT[])
        @test Float64(res) ≈ prod(params)
    end

    grads2 = copy(grads)
    for _ in 1:100
        params = rand(NP) .+ 0.4
        TS.set_params!(s, params)
        for v in grads
            v.v = 0
        end
        res = VT(0.0)
        TS.compute!(res, s, grads)
        @test Float64(res) ≈ prod(params)
        for (v1, v2) in zip(grads, grads2)
            @test v1 === v2
        end
        for i in 1:NP
            @test Float64(grads[i]) ≈ prod((p for (pidx, p) in enumerate(params)
                                                 if pidx != i), init=1.0)
        end
    end
end

@testset "Scalar Time Sequence [$(VT.name.name)]" for VT in (Float64, F64Ref, F64Ref2)
    if VT === F64Ref2
        kws = (;init=()->F64Ref2(0.0), mul=nothing,
               mul! = (out::F64Ref2, a::F64Ref2, b::F64Ref2)->(out.v = a.v * b.v))
    else
        kws = ()
    end

    ops = (Step1{VT}(), Step2{VT}(), Step3{VT}(), Step1{VT}(),
           Step1{VT}(), Step2{VT}(), Step3{VT}(), Step2{VT}(),
           Step2{VT}(), Step1{VT}(), Step3{VT}(), Step3{VT}())
    s = TS.Sequence{VT}(ops; kws...)
    @test TS.nparams(s) == 24
    test_scalar_sequence(s)

    s1 = TS.Sequence{VT}((Step1{VT}(),); kws...)
    @test TS.nparams(s1) == 1
    test_scalar_sequence(s1)

    s2 = TS.Sequence{VT}((Step2{VT}(),); kws...)
    @test TS.nparams(s2) == 2
    test_scalar_sequence(s2)

    s3 = TS.Sequence{VT}((Step3{VT}(),); kws...)
    @test TS.nparams(s3) == 3
    test_scalar_sequence(s3)

    if VT === Float64
        continue
    end

    ops = (IStep1{VT}(), IStep2{VT}(), IStep3{VT}(), IStep1{VT}(),
           IStep1{VT}(), IStep2{VT}(), IStep3{VT}(), IStep2{VT}(),
           IStep2{VT}(), IStep1{VT}(), IStep3{VT}(), IStep3{VT}())
    s = TS.Sequence{VT}(ops; kws...)
    @test TS.nparams(s) == 24
    test_scalar_sequence(s)

    s1 = TS.Sequence{VT}((IStep1{VT}(),); kws...)
    @test TS.nparams(s1) == 1
    test_scalar_sequence(s1)

    s2 = TS.Sequence{VT}((IStep2{VT}(),); kws...)
    @test TS.nparams(s2) == 2
    test_scalar_sequence(s2)

    s3 = TS.Sequence{VT}((IStep3{VT}(),); kws...)
    @test TS.nparams(s3) == 3
    test_scalar_sequence(s3)
end

mutable struct SStep1 <: TS.AbstractStep{String,1}
    v1::String
    SStep1() = new()
end
function TS.set_params!(op::SStep1, params)
    op.v1 = params[1]
    return
end
function TS.compute(op::SStep1, grad)
    if !isempty(grad)
        grad[1] = "d"
    end
    return op.v1
end
mutable struct SStep2 <: TS.AbstractStep{String,2}
    v1::String
    v2::String
    SStep2() = new()
end
function TS.set_params!(op::SStep2, params)
    op.v1 = params[1]
    op.v2 = params[2]
    return
end
function TS.compute(op::SStep2, grad)
    if !isempty(grad)
        grad[1] = "d" * op.v2
        grad[2] = op.v1 * "d"
    end
    return op.v1 * op.v2
end
mutable struct SStep3 <: TS.AbstractStep{String,3}
    v1::String
    v2::String
    v3::String
    SStep3() = new()
end
function TS.set_params!(op::SStep3, params)
    op.v1 = params[1]
    op.v2 = params[2]
    op.v3 = params[3]
    return
end
function TS.compute(op::SStep3, grad)
    if !isempty(grad)
        grad[1] = "d" * op.v2 * op.v3
        grad[2] = op.v1 * "d" * op.v3
        grad[3] = op.v1 * op.v2 * "d"
    end
    return op.v1 * op.v2 * op.v3
end

function test_string_sequence(s)
    NP = TS.nparams(typeof(s))
    @test TS.nparams(s) == NP
    @test !TS.support_inplace_compute(typeof(s))

    grads = fill("", NP)
    for _ in 1:100
        params = [randstring(2) for _ in 1:NP]
        TS.set_params!(s, params)
        @test TS.compute(s, String[]) == prod(params)
        grads .= ""
        @test TS.compute(s, grads) == prod(params)
        for i in 1:NP
            @test grads[i] == prod(((pidx == i ? "d" : p)
                                    for (pidx, p) in enumerate(params)))
        end
    end
end

@testset "String Time Sequence" begin
    ops = (SStep1(), SStep2(), SStep3(), SStep1(),
           SStep1(), SStep2(), SStep3(), SStep2(),
           SStep2(), SStep1(), SStep3(), SStep3())
    s = TS.Sequence{String}(ops)
    @test TS.nparams(s) == 24
    test_string_sequence(s)

    s1 = TS.Sequence{String}((SStep1(),))
    @test TS.nparams(s1) == 1
    test_string_sequence(s1)

    s2 = TS.Sequence{String}((SStep2(),))
    @test TS.nparams(s2) == 2
    test_string_sequence(s2)

    s3 = TS.Sequence{String}((SStep3(),))
    @test TS.nparams(s3) == 3
    test_string_sequence(s3)
end

using LinearAlgebra
using StaticArrays

const σx = ComplexF64[0 1; 1 0]
const σy = ComplexF64[0 -im; im 0]
const σz = ComplexF64[1 0; 0 -1]

@testset "sinhc" begin
    Im = TS.Imaginary
    for v in (0.0, 1.0, -2.5, 3)
        a = Im(v)
        ref = v * im
        @test TS._sinhc(a) ≈ (iszero(v) ? 1 : sinh(ref) / ref)
        @test TS._sinhc(a) isa Real
    end
    for z in (0.0 + 0.0im, 1.0 + 2.0im, 0.0 - 4im)
        @test TS._sinhc(z) ≈ (iszero(z) ? 1 : sinh(z) / z)
    end
end

# Exact Fréchet derivative of `exp` at `M` in the direction `E` via the block matrix trick.
function exp_frechet(M, E)
    n = size(M, 1)
    B = [M E; zero(M) M]
    return exp(B)[1:n, n + 1:2n]
end

function ti_reference(Hs, H0, coeffs, t)
    n = size(Hs === () ? H0 : Hs[1], 1)
    A = zeros(ComplexF64, n, n)
    if H0 !== nothing
        A .+= H0
    end
    for (c, H) in zip(coeffs, Hs)
        A .+= c .* H
    end
    M = -im * t * A
    U = exp(M)
    dU = [exp_frechet(M, -im * t * H) for H in Hs]
    return A, U, dU, -im * A * U
end

function test_ti_step(::Type{OP}, Hs, H0, param_t; rng, hermitian=nothing) where OP
    NH = length(Hs)
    NP = NH + param_t
    t0 = 0.7
    step = TS.ConstMatrixStep{OP}(Hs; H0=H0, t=t0, param_t=param_t, hermitian=hermitian)
    @test TS.nparams(step) == NP
    @test TS.nparams(typeof(step)) == NP
    @test TS.support_inplace_compute(typeof(step)) == ismutabletype(OP)
    n = size(NH > 0 ? Hs[1] : H0, 1)
    init = TS.get_init(step)
    buf1 = init()
    buf2 = init()
    @test buf1 isa OP
    @test size(buf1) == (n, n)
    @test iszero(buf1)
    @test !ismutabletype(OP) || buf1 !== buf2
    approx(a, b) = isapprox(a, b, rtol=1e-8, atol=1e-10)

    for trial in 1:20
        params = randn(rng, NP)
        if trial == 1 && param_t
            params[end] = 0 # t = 0 should give the identity
        end
        TS.set_params!(step, params)
        coeffs = params[1:NH]
        t = param_t ? params[end] : t0
        A, Uref, dUref, dUdt = ti_reference(Hs, H0, coeffs, t)
        if trial == 1 && param_t
            @test Uref ≈ I
        end

        U = @inferred TS.compute(step, OP[])
        @test U isa OP
        @test approx(U, Uref)

        grads = [OP(zeros(n, n)) for _ in 1:NP]
        U = @inferred TS.compute(step, grads)
        @test approx(U, Uref)
        for k in 1:NH
            @test grads[k] isa OP
            @test approx(grads[k], dUref[k])
        end
        if param_t
            @test grads[end] isa OP
            @test approx(grads[end], dUdt)
        end

        if !ismutabletype(OP)
            @test_throws ArgumentError TS.compute!(OP(zeros(n, n)), step, OP[])
            continue
        end

        res = OP(zeros(n, n))
        @test TS.compute!(res, step, OP[]) === res
        @test approx(res, Uref)

        grads2 = [OP(zeros(n, n)) for _ in 1:NP]
        grads2_ids = copy(grads2)
        res = OP(zeros(n, n))
        @test TS.compute!(res, step, grads2) === res
        @test approx(res, Uref)
        for (g1, g2, gid) in zip(grads, grads2, grads2_ids)
            @test g2 === gid
            @test approx(g1, g2)
        end
    end
end

@testset "ConstMatrixStep" begin
    rng = Xoshiro(12345)
    I2 = Matrix{ComplexF64}(I, 2, 2)

    @testset "Hermitian 2x2 [$OP]" for OP in (Matrix{ComplexF64}, SMatrix{2,2,ComplexF64,4},
                                             MMatrix{2,2,ComplexF64,4})
        for param_t in (false, true), H0 in (nothing, σz / 4)
            test_ti_step(OP, (σx / 2, σy / 2, σz / 2), H0, param_t; rng)
            test_ti_step(OP, (σx / 2, σy / 2), H0, param_t; rng)
            test_ti_step(OP, (σx / 2,), H0, param_t; rng)
            test_ti_step(OP, (), σz, param_t; rng)
            # Generic path forced on Hermitian input
            test_ti_step(OP, (σx / 2, σy / 2, σz / 2), H0, param_t; rng, hermitian=false)
        end
        # Vector input for `Hs`
        step = TS.ConstMatrixStep{OP}([σx, σy])
        @test TS.nparams(step) == 2
    end

    @testset "Degenerate spectrum [$OP]" for OP in (Matrix{ComplexF64}, SMatrix{4,4,ComplexF64,16})
        for param_t in (false, true)
            # Eigenvalues ±sqrt(c1² + c2²), each doubly degenerate
            test_ti_step(OP, (kron(σx, I2), kron(σz, I2)), nothing, param_t; rng)
            # Exactly degenerate diagonal generator
            test_ti_step(OP, (ComplexF64.(diagm([1, 1, 2, 2])),), nothing, param_t; rng)
            # Nearly degenerate
            test_ti_step(OP, (ComplexF64.(diagm([1, 1 + 1e-9, 2, 2 + 1e-12])),),
                         kron(σx, σx), param_t; rng)
        end
    end

    @testset "Real symmetric generators [$OP]" for OP in (Matrix{ComplexF64},
                                                        SMatrix{3,3,ComplexF64,9})
        H1 = randn(rng, 3, 3); H1 = H1 + H1'
        H2 = randn(rng, 3, 3); H2 = H2 + H2'
        H0 = randn(rng, 3, 3); H0 = H0 + H0'
        for param_t in (false, true)
            test_ti_step(OP, (H1, H2), nothing, param_t; rng)
            test_ti_step(OP, (H1, H2), H0, param_t; rng)
            test_ti_step(OP, (), H0, param_t; rng)
        end
    end

    @testset "General [$OP]" for OP in (Matrix{ComplexF64}, SMatrix{3,3,ComplexF64,9})
        Hs = (randn(rng, ComplexF64, 3, 3), randn(rng, ComplexF64, 3, 3))
        H0 = randn(rng, ComplexF64, 3, 3)
        for param_t in (false, true)
            test_ti_step(OP, Hs, nothing, param_t; rng)
            test_ti_step(OP, Hs, H0, param_t; rng)
            test_ti_step(OP, (), H0, param_t; rng)
            # Real non-symmetric generators
            test_ti_step(OP, (real(Hs[1]), real(Hs[2])), real(H0), param_t; rng)
        end
    end

    @testset "Errors" begin
        MT = Matrix{ComplexF64}
        @test_throws ArgumentError TS.ConstMatrixStep{MT}(())
        @test_throws DimensionMismatch TS.ConstMatrixStep{MT}((σx, zeros(ComplexF64, 3, 3)))
        @test_throws DimensionMismatch TS.ConstMatrixStep{MT}((zeros(ComplexF64, 2, 3),))
        @test_throws DimensionMismatch TS.ConstMatrixStep{MT}((σx,); H0=zeros(ComplexF64, 3, 3))
        @test_throws DimensionMismatch TS.ConstMatrixStep{MT}((); H0=zeros(ComplexF64, 2, 3))
        # Real matrix type
        @test_throws ArgumentError TS.ConstMatrixStep{Matrix{Float64}}((real(σx),))
        @test_throws ArgumentError TS.ConstMatrixStep{SMatrix{2,2,Float64,4}}((real(σx),))
        step = TS.ConstMatrixStep{MT}((σx,))
        @test_throws AssertionError TS.set_params!(step, [1.0, 2.0])
        @test_throws AssertionError TS.compute(step, [zeros(ComplexF64, 2, 2) for _ in 1:2])
        @test_throws AssertionError TS.compute!(zeros(ComplexF64, 2, 2), step,
                                                [zeros(ComplexF64, 2, 2) for _ in 1:2])
    end
end

function test_ti_sequence(::Type{OP}, s, rng; inplace) where OP
    @test TS.nparams(s) == 7
    @test TS.get_init(s) === s.init
    @test TS.get_init(s)() isa OP
    approx(a, b) = isapprox(a, b, rtol=1e-8, atol=1e-10)
    for _ in 1:20
        params = randn(rng, 7)
        TS.set_params!(s, params)

        A1 = params[1] * σx / 2 + params[2] * σy / 2 + σz / 4
        t1 = params[3]
        A2 = σz / 2
        t2 = params[4]
        A3 = params[5] * σx / 2 + params[6] * σy / 2 + params[7] * σz / 2
        t3 = 1.0
        M1 = -im * t1 * A1
        M2 = -im * t2 * A2
        M3 = -im * t3 * A3
        U1 = exp(M1)
        U2 = exp(M2)
        U3 = exp(M3)
        Uref = U1 * U2 * U3
        dUref = [exp_frechet(M1, -im * t1 * σx / 2) * U2 * U3,
                 exp_frechet(M1, -im * t1 * σy / 2) * U2 * U3,
                 (-im * A1 * U1) * U2 * U3,
                 U1 * (-im * A2 * U2) * U3,
                 U1 * U2 * exp_frechet(M3, -im * t3 * σx / 2),
                 U1 * U2 * exp_frechet(M3, -im * t3 * σy / 2),
                 U1 * U2 * exp_frechet(M3, -im * t3 * σz / 2)]

        @test approx(TS.compute(s, OP[]), Uref)
        grads = [OP(zeros(2, 2)) for _ in 1:7]
        @test approx(TS.compute(s, grads), Uref)
        for k in 1:7
            @test approx(grads[k], dUref[k])
        end

        if !inplace
            continue
        end
        res = OP(zeros(2, 2))
        TS.compute!(res, s, OP[])
        @test approx(res, Uref)
        grads = [OP(zeros(2, 2)) for _ in 1:7]
        grads_ids = copy(grads)
        TS.compute!(res, s, grads)
        @test approx(res, Uref)
        for k in 1:7
            @test grads[k] === grads_ids[k]
            @test approx(grads[k], dUref[k])
        end
    end
end

@testset "ConstMatrixStep Sequence [$OP]" for OP in (Matrix{ComplexF64},
                                                        SMatrix{2,2,ComplexF64,4})
    rng = Xoshiro(54321)
    make_steps() = (TS.ConstMatrixStep{OP}((σx / 2, σy / 2); H0=σz / 4, param_t=true),
                    TS.ConstMatrixStep{OP}((); H0=σz / 2, param_t=true),
                    TS.ConstMatrixStep{OP}((σx / 2, σy / 2, σz / 2)))
    # Default: init from the steps, allocating multiplication
    s = TS.Sequence{OP}(make_steps())
    @test s.init !== nothing
    @test !TS.support_inplace_compute(typeof(s))
    test_ti_sequence(OP, s, rng; inplace=false)
    # Nested sequence gets its init from the inner sequence
    s2 = TS.Sequence{OP}((s, make_steps()[1]))
    @test s2.init === s.init
    @test TS.nparams(s2) == 10
    if OP <: Matrix
        # In-place multiplication with init from the steps
        s = TS.Sequence{OP}(make_steps(); mul=nothing, mul! = LinearAlgebra.mul!)
        @test TS.support_inplace_compute(typeof(s))
        test_ti_sequence(OP, s, rng; inplace=true)
        # Explicit init still takes precedence
        init = ()->zeros(ComplexF64, 2, 2)
        s = TS.Sequence{OP}(make_steps(); init=init, mul=nothing, mul! = LinearAlgebra.mul!)
        @test s.init === init
        test_ti_sequence(OP, s, rng; inplace=true)
    end
end
