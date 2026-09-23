#

module TimeSequence

using LinearAlgebra
using StaticArrays

using ..Math: Imaginary

public AbstractStep, support_inplace_compute, compute, compute!, set_params!,
    get_init, get_mul, get_mul!, Sequence, ConstMatrixStep, QobjEvoStep

abstract type AbstractStep{OP,NParams} end

@inline support_inplace_compute(T) = false

"""
    compute(step::AbstractStep{OP,NParams}, grad::AbstractVector{OP})::OP

Compute the operator for the `step`.
If `grad` is not empty, it should be an array of size `NParams` to store the
gradient WRT each parameter.
"""
function compute end

"""
    compute!(output::OP, step::AbstractStep{OP,NParams}, grad::AbstractVector{OP})

Similar to `compute` but does everything in-place by mutating the `OP` object
in `output` and the `grad` array.
"""
function compute! end

"""
    set_params!(step::AbstractStep{OP,NParams}, params::AbstractVector)::Nothing

Set the parameters for the `step` to `params`, which should have length `NParams`.
"""
function set_params! end

(nparams(::Type{T} where T<:AbstractStep{OP,NParams}) where {OP,NParams}) = NParams
(nparams(::AbstractStep{OP,NParams}) where {OP,NParams}) = NParams

"""
    get_init(::Type{OP})
    get_init(step::AbstractStep)

Return a function that creates a new zero-initialized `OP` object, used by
[`Sequence`](@ref) to populate its buffers, or `nothing` if none is available.
`Sequence` uses `get_init(OP)` and falls back to the first non-`nothing`
`get_init(step)` of its steps. An `init` is required when in-place multiplication
(`mul!`) is used or when any of the steps supports in-place compute.
"""
get_init(T) = nothing
get_mul(T) = *
get_mul!(T) = nothing

_first_init(::Tuple{}) = nothing
function _first_init(steps::Tuple)
    init = get_init(steps[1])
    return init === nothing ? _first_init(Base.tail(steps)) : init
end
function _default_init(::Type{OP}, steps) where OP
    init = get_init(OP)
    return init === nothing ? _first_init(steps) : init
end

struct Sequence{OP,NSteps,Steps<:NTuple{NSteps,AbstractStep},NParams,Init,Mul,Mul!,ValBuf,GradBuf,PartialBuf,TmpBuf} <: AbstractStep{OP,NParams}
    steps::Steps
    val_buf::ValBuf
    grad_buf::GradBuf
    prefix_buf::PartialBuf
    suffix_buf::PartialBuf
    tmp_buf::TmpBuf # [Result, intermediate for gradient]

    init::Init
    mul::Mul
    mul!::Mul!

    function Sequence{OP}(steps::Steps; init::Init=_default_init(OP, steps),
                          mul::Mul=get_mul(OP), mul!::(Mul!)=get_mul!(OP)) where Steps<:NTuple{NSteps,AbstractStep{OP}} where {OP,NSteps,Init,Mul,Mul!}
        @assert NSteps > 0
        if mul! !== nothing
            @assert init !== nothing
            @assert mul === nothing
        end
        NParams = sum(nparams, steps)
        op_isbits = isbitstype(OP)

        function op_mem(n)
            mem = Memory{OP}(undef, n)
            if init !== nothing
                @inbounds for i in 1:n
                    mem[i] = init()::OP
                end
            end
            return mem
        end
        op_mvec(n) = (init !== nothing ? MVector(ntuple(_->init()::OP, n)) :
            MVector{n,OP}(undef))
        op_buf(n) = if n <= 0
            return nothing
        elseif !op_isbits
            return op_mem(n)
        else
            return op_mvec(n)
        end

        val_buf = op_buf(NSteps)
        grad_buf = op_buf(NParams)
        prefix_buf = op_buf(NSteps - 2)
        suffix_buf = op_buf(NSteps - 2)
        tmp_buf = mul! === nothing ? nothing : (init()::OP, init()::OP)
        s = new{OP,NSteps,Steps,NParams,Init,Mul,Mul!,typeof(val_buf),typeof(grad_buf),
                typeof(prefix_buf),typeof(tmp_buf)}(
                    steps, val_buf, grad_buf, prefix_buf, suffix_buf,
                    tmp_buf, init, mul, mul!)
        return s
    end
end

get_init(s::Sequence) = s.init

Base.@assume_effects :foldable function support_inplace_compute(::Type{<:Sequence{OP,NSteps,Steps,NParams,Init,Mul,Mul!}}) where {OP,NSteps,Steps,NParams,Init,Mul,Mul!}
    if NSteps == 1
        return support_inplace_compute(Steps.parameters[1])
    end
    return Mul! !== Nothing
end

@generated function _param_range(::Type{<:Sequence{OP,NSteps,Steps}}) where {OP,NSteps,Steps}
    StepTypes = (Steps.parameters...,)
    step_nparams = nparams.(StepTypes)
    cum_nparams = cumsum(step_nparams)
    starts = (0, cum_nparams[1:end - 1]...) .+ 1
    return starts, cum_nparams
end

@generated function set_params!(s::Sequence, params::AbstractVector)
    ex = quote
        @assert length(params) == $(nparams(s))
        steps = s.steps
    end
    starts, ends = _param_range(s)
    for (i, (start_idx, end_idx)) in enumerate(zip(starts, ends))
        push!(ex.args, :(@inline set_params!(@inbounds(steps[$i]),
                                             @view params[$start_idx:$end_idx])))
    end
    push!(ex.args, :(return))
    return ex
end

# Insert a `nothing` prefix to the array to avoid the compiler optimize out
# the line number node when interpolating the expression to the parent function.
__eval_compute_body(starts, ends, has_grad) = insert!(Any[
    quote
        step = steps[$i]
        stepgrad = $(has_grad ? :(@view(grad_buf[$start_idx:$end_idx])) : :dummy_grad)
        if support_inplace_compute(typeof(step))
            compute!(val_buf[$i], step, stepgrad)
        else
            val_buf[$i] = compute(step, stepgrad)
        end
    end for (i, (start_idx, end_idx)) in enumerate(zip(starts, ends))], 1, :nothing)

@generated function _eval_compute(s::Sequence{OP}, grads, has_grad) where OP
    starts, ends = _param_range(s)
    ex = quote
        steps = s.steps
        grad_buf = s.grad_buf
        val_buf = s.val_buf
        @inbounds @inline if has_grad
            $(__eval_compute_body(starts, ends, true)...)
        else
            dummy_grad = SVector{0,OP}()
            $(__eval_compute_body(starts, ends, false)...)
        end
    end
    return ex
end

macro pick_mul(mul!, mul, out, a, b)
    mul! = esc(mul!)
    mul = esc(mul)
    out = esc(out)
    a = esc(a)
    b = esc(b)
    # Use an dummy expression to make sure the original source info is preserved
    :(nothing;
      if $mul! === nothing
          $mul($a, $b)
      else
          v = $out
          $mul!(v, $a, $b)
          v
      end)
end

macro pick_mulass(mul!, mul, out, a, b)
    mul! = esc(mul!)
    mul = esc(mul)
    out = esc(out)
    a = esc(a)
    b = esc(b)
    # Use an dummy expression to make sure the original source info is preserved
    :(nothing;
      if $mul! === nothing
          v = $mul($a, $b)
          $out = v
      else
          v = $out
          $mul!(v, $a, $b)
      end;
      v)
end

@inline function _eval_grads(s::Sequence{OP,NSteps,Steps,NParams}, grads, mul, mul!, first_val, last_val) where {OP,NSteps,Steps,NParams}
    @assert length(grads) == NParams
    prev = last_val
    @inbounds for i in NSteps - 2:-1:1
        prev = @pick_mulass(mul!, mul, s.suffix_buf[i], s.val_buf[i + 1], prev)
    end
    tmp2 = NSteps > 2 && mul! !== nothing ? @inbounds(s.tmp_buf[2]) : nothing
    starts, ends = _param_range(typeof(s))
    @inbounds for step_idx in 1:NSteps
        pstart = starts[step_idx]
        pend = ends[step_idx]
        if step_idx == 1
            suffix = step_idx == NSteps - 1 ? last_val : s.suffix_buf[step_idx]
            for param_idx in pstart:pend
                @pick_mulass(mul!, mul, grads[param_idx], s.grad_buf[param_idx], suffix)
            end
        elseif step_idx == NSteps
            prefix = step_idx == 2 ? first_val : s.prefix_buf[step_idx - 2]
            for param_idx in pstart:pend
                @pick_mulass(mul!, mul, grads[param_idx], prefix, s.grad_buf[param_idx])
            end
        else
            prefix = step_idx == 2 ? first_val : s.prefix_buf[step_idx - 2]
            suffix = step_idx == NSteps - 1 ? last_val : s.suffix_buf[step_idx]
            for param_idx in pstart:pend
                tmp = @pick_mul(mul!, mul, tmp2, prefix, s.grad_buf[param_idx])
                @pick_mulass(mul!, mul, grads[param_idx], tmp, suffix)
            end
        end
    end
    return
end

function compute(s::Sequence{OP,NSteps,Steps,NParams}, grads) where {OP,NSteps,Steps,NParams}
    if NSteps == 1
        return compute(s.steps[1], grads)
    end
    mul = s.mul
    mul! = s.mul!
    has_grad = !isempty(grads)
    @inline _eval_compute(s, grads, has_grad)
    first_val = @inbounds s.val_buf[1]
    last_val = @inbounds s.val_buf[NSteps]

    prev = first_val
    @inbounds for i in 2:NSteps - 1
        prev = @pick_mulass(mul!, mul, s.prefix_buf[i - 1],
                            prev, s.val_buf[i])
    end
    res = @pick_mul(mul!, mul, s.tmp_buf[1], prev, last_val)
    has_grad && _eval_grads(s, grads, mul, mul!, first_val, last_val)
    return res
end

function compute!(res::OP, s::Sequence{OP,NSteps,Steps,NParams}, grads) where {OP,NSteps,Steps,NParams}
    if NSteps == 1
        compute!(res, s.steps[1], grads)
        return res
    end
    mul! = s.mul!
    @assert mul! !== nothing
    has_grad = !isempty(grads)
    @inline _eval_compute(s, grads, has_grad)
    first_val = @inbounds s.val_buf[1]
    last_val = @inbounds s.val_buf[NSteps]

    prev = first_val
    @inbounds for i in 2:NSteps - 1
        next = s.prefix_buf[i - 1]
        mul!(next, prev, s.val_buf[i])
        prev = next
    end
    mul!(res, prev, last_val)
    has_grad && _eval_grads(s, grads, nothing, mul!, first_val, last_val)
    return res
end

##### Time-independent parametrized matrix operator

const _NEG_IM = Imaginary(-1)

# Buffers used by the in-place `compute!` for mutable matrix types.
struct _TIBuffers{MT<:AbstractMatrix,VT<:AbstractVector}
    A::MT    # assembled generator
    Φ::MT    # divided differences of exp in the eigenbasis
    tmp1::MT
    tmp2::MT
    h::VT    # exp(-im * t * λ / 2)
    eλ::VT   # exp(-im * t * λ)
end

"""
    ConstMatrixStep{OP}(Hs; H0=nothing, t=1, param_t=false, hermitian=nothing)

A step whose operator is the time evolution under a time-independent, linearly
parametrized Hamiltonian. With `NH = length(Hs)`, coefficients `c` and duration `t`,
the operator is (with `ħ = 1`)

    U = exp(-im * t * (H0 + c[1] * Hs[1] + ... + c[NH] * Hs[NH]))

The step has `NH` parameters, the coefficients `c`, and, if `param_t=true`,
one additional trailing parameter for the duration `t`.
Otherwise `t` is fixed to the value of the keyword argument.
`Hs` may be empty, e.g. to describe a free evolution under `H0` for a variable time.

`OP` is the matrix type of the result (e.g. `Matrix{ComplexF64}` or
`SMatrix{2,2,ComplexF64,4}`) and must have a complex element type.
`Hs` (a tuple) and `H0` are converted to `OP`.

If `hermitian` is not given, it is inferred from the generator matrices.
For a Hermitian generator (real coefficients are assumed), the exponential and
its derivatives are computed with a Hermitian eigendecomposition (only the upper
triangle of the assembled generator is accessed when `hermitian=true` is forced).
Otherwise a general eigendecomposition is used, which requires the generator to be
diagonalizable.

The gradient WRT `c[k]` is the Fréchet derivative of the matrix exponential in the
direction `-im * t * Hs[k]`, evaluated in the eigenbasis using divided differences
of `exp`, which is well-behaved for (nearly) degenerate eigenvalues.
The gradient WRT `t` is `-im * A * U`, where `A` is the assembled generator.

In-place evaluation (`compute!`) is supported when `OP` is a mutable matrix type
(e.g. `Matrix`), in which case the working buffers are allocated once at construction.
The eigendecomposition itself still allocates its result arrays.
The step provides [`get_init`](@ref), so a [`Sequence`](@ref) of these steps
can be constructed without an explicit `init`.
"""
mutable struct ConstMatrixStep{OP<:AbstractMatrix,NParams,NH,PT,Herm,H0T,Buf} <: AbstractStep{OP,NParams}
    const H0::H0T
    const Hs::NTuple{NH,OP}
    coeffs::SVector{NH,PT}
    t::PT
    const buf::Buf

    function ConstMatrixStep{OP}(Hs; H0=nothing, t=1, param_t::Bool=false,
                                     hermitian::Union{Bool,Nothing}=nothing) where OP<:AbstractMatrix
        T = eltype(OP)
        if !(T <: Complex)
            throw(ArgumentError("`OP` must have a complex element type, got $OP"))
        end
        PT = real(T)
        Hs = map(H->convert(OP, H)::OP, Tuple(Hs))
        NH = length(Hs)
        H0 = H0 === nothing ? nothing : convert(OP, H0)::OP
        ref = NH > 0 ? Hs[1] : H0
        if ref === nothing
            throw(ArgumentError("At least one generator matrix (`Hs` or `H0`) is required"))
        end
        n = size(ref, 1)
        for H in (H0 === nothing ? Hs : (H0, Hs...))
            if size(H) != (n, n)
                throw(DimensionMismatch("All generator matrices must be square and of the same size"))
            end
        end
        if hermitian === nothing
            hermitian = all(ishermitian, Hs) && (H0 === nothing || ishermitian(H0))
        end
        t = convert(PT, t)
        NParams = NH + Int(param_t)
        if ismutabletype(OP)
            buf = _TIBuffers(similar(ref, T), similar(ref, T), similar(ref, T), similar(ref, T),
                             similar(ref, T, n), similar(ref, T, n))
        else
            buf = nothing
        end
        return new{OP,NParams,NH,PT,hermitian,typeof(H0),typeof(buf)}(
            H0, Hs, zero(SVector{NH,PT}), t, buf)
    end
end

@inline _ref_matrix(step::ConstMatrixStep{OP,NParams,NH}) where {OP,NParams,NH} =
    NH == 0 ? step.H0 : step.Hs[1]

function get_init(step::ConstMatrixStep)
    ref = _ref_matrix(step)
    return ()->zero(ref)
end

support_inplace_compute(::Type{<:ConstMatrixStep{OP}}) where OP = ismutabletype(OP)

function set_params!(step::ConstMatrixStep{OP,NParams,NH,PT},
                     params::AbstractVector) where {OP,NParams,NH,PT}
    @assert length(params) == NParams
    step.coeffs = SVector{NH,PT}(ntuple(k->PT(params[k]), Val(NH)))
    if NParams > NH
        step.t = PT(params[NParams])
    end
    return
end

# sinh(z) / z, continuous at 0
@inline _sinhc(z) = iszero(z) ? one(z) : sinh(z) / z
@inline function _sinhc(z::Imaginary)
    y = z.v
    return iszero(y) ? one(y) : sin(y) / y
end
# s * (exp(s λi) - exp(s λj)) / (s λi - s λj), with h = exp(s λ / 2), continuous at λi == λj
@inline _phi(s, hi, hj, λi, λj) = s * hi * hj * _sinhc(s * (λi - λj) / 2)

@inline function _eigen_decomp(A, ::Val{true})
    E = eigen(Hermitian(A))
    V = E.vectors
    return E.values, V, V'
end
@inline function _eigen_decomp(A, ::Val{false})
    E = eigen(convert(Matrix, A))
    V = E.vectors
    return E.values, V, inv(V)
end

@inline _to_op(::Type{OP}, m) where OP = convert(OP, m)

@inline function _assemble(step::ConstMatrixStep{OP,NParams,NH}) where {OP,NParams,NH}
    H0 = step.H0
    if NH == 0
        return H0
    end
    Hs = step.Hs
    coeffs = step.coeffs
    A = coeffs[1] * Hs[1]
    for k in 2:NH
        A = A + coeffs[k] * Hs[k]
    end
    if H0 !== nothing
        A = A + H0
    end
    return A
end

function compute(step::ConstMatrixStep{OP,NParams,NH,PT,Herm}, grad) where {OP,NParams,NH,PT,Herm}
    A = _assemble(step)
    s = Imaginary(-step.t)
    λ, V, W = _eigen_decomp(A, Val(Herm))
    h = exp.((s / 2) .* λ)
    eλ = h .* h
    U = (V .* transpose(eλ)) * W
    if !isempty(grad)
        @assert length(grad) == NParams
        Φ = _phi.(s, h, transpose(h), λ, transpose(λ))
        Hs = step.Hs
        for k in 1:NH
            grad[k] = _to_op(OP, V * (Φ .* (W * Hs[k] * V)) * W)
        end
        if NParams > NH
            grad[NParams] = _to_op(OP, _NEG_IM .* (A * U))
        end
    end
    return _to_op(OP, U)
end

function _assemble!(A, step::ConstMatrixStep{OP,NParams,NH}) where {OP,NParams,NH}
    H0 = step.H0
    if H0 === nothing
        fill!(A, zero(eltype(A)))
    else
        copyto!(A, H0)
    end
    Hs = step.Hs
    coeffs = step.coeffs
    for k in 1:NH
        axpy!(coeffs[k], Hs[k], A)
    end
    return A
end

function _fill_phi!(Φ, s, h, λ)
    n = length(λ)
    @inbounds for j in 1:n
        hj = h[j]
        λj = λ[j]
        for i in 1:n
            Φ[i, j] = _phi(s, h[i], hj, λ[i], λj)
        end
    end
    return Φ
end

function compute!(res::OP, step::ConstMatrixStep{OP,NParams,NH,PT,Herm}, grad) where {OP,NParams,NH,PT,Herm}
    buf = step.buf
    if buf === nothing
        throw(ArgumentError("In-place compute is not supported for immutable matrix type $OP"))
    end
    A = _assemble!(buf.A, step)
    s = Imaginary(-step.t)
    λ, V, W = _eigen_decomp(A, Val(Herm))
    h = buf.h
    eλ = buf.eλ
    h .= exp.((s / 2) .* λ)
    eλ .= h .* h
    tmp1 = buf.tmp1
    tmp2 = buf.tmp2
    copyto!(tmp1, V)
    rmul!(tmp1, Diagonal(eλ))
    mul!(res, tmp1, W)
    if !isempty(grad)
        @assert length(grad) == NParams
        Φ = _fill_phi!(buf.Φ, s, h, λ)
        Hs = step.Hs
        for k in 1:NH
            mul!(tmp1, Hs[k], V)
            mul!(tmp2, W, tmp1)
            tmp2 .*= Φ
            mul!(tmp1, V, tmp2)
            mul!(grad[k], tmp1, W)
        end
        if NParams > NH
            # -im * A * U. The scaling is done by BLAS for strided matrices.
            mul!(grad[NParams], A, res, -im, false)
        end
    end
    return res
end


##### Step defined by a (time-dependent) QuantumToolbox operator
# The constructor and `compute`/`compute!` are implemented in the `AMOQuantumToolboxExt`
# package extension, which is loaded together with `QuantumToolbox`.

"""
    QobjEvoStep{OP}(H::AbstractQuantumObject; nparams, tspan, param_map=identity, kwargs...)
    QobjEvoStep(H::AbstractQuantumObject; kwargs...)

A step whose operator is the time evolution under a `QuantumObject` or a (time-dependent)
`QuantumObjectEvolution` (`QobjEvo`) `H` from `QuantumToolbox`, from `tspan[1]` to
`tspan[2]`. This requires the `QuantumToolbox` package to be loaded.

The time-dependent coefficients `f(p, t)` of `H` receive `param_map(p)` as their first
argument, where `p` is an `SVector` of the `nparams` parameters of the step
(possibly containing `ForwardDiff.Dual` numbers). Use `param_map` to, e.g., convert the
parameters into a `NamedTuple` expected by the coefficient functions.
The coefficient functions must be generic in the element type of `p`.

For an `Operator` (a Hamiltonian ``H``), the operator of the step is the propagator
``U = T\\exp(-i∫H dt)``. For a `SuperOperator` (a Liouvillian ``L``), it is
``T\\exp(∫L dt)`` acting on the vectorized (column stacking) density matrix.

The propagator is computed with `QuantumToolbox.sesolve` using the identity operator as
the initial state. The gradient WRT the parameters is computed by solving the sensitivity
equations ``∂_t (∂_k U) = -i (H ∂_k U + (∂_k H) U)`` together with the propagator as a
single (vectorized) `sesolve` problem, where ``∂_k H`` is obtained by forward-mode
automatic differentiation of the coefficient functions. The accuracy of both the
operator and its gradient are therefore determined by the ODE solver options,
which can be passed as keyword arguments (e.g. `alg`, `reltol`, `abstol`) and are
forwarded to `sesolve`.

`OP` is the matrix type of the result (e.g. `Matrix{ComplexF64}` or
`SMatrix{2,2,ComplexF64,4}`) and defaults to a dense `Matrix` with the complex element
type of `H`. The step provides [`get_init`](@ref), so a [`Sequence`](@ref) of these steps
can be constructed without an explicit `init`.
"""
mutable struct QobjEvoStep{OP<:AbstractMatrix,NParams,PT,HP,SP,HA,SA,KW} <: AbstractStep{OP,NParams}
    const H_prop::HP  # operator for the propagator
    const ψ0_prop::SP # initial state (identity) for `H_prop`
    const H_aug::HA   # (vectorized) operator for the propagator and its sensitivities
    const ψ0_aug::SA  # initial state for `H_aug`
    const n::Int      # size of the operator
    const t0::PT
    const t1::PT
    const kwargs::KW  # forwarded to the solver
    params::SVector{NParams,PT}
end

support_inplace_compute(::Type{<:QobjEvoStep{OP}}) where OP = ismutabletype(OP)

function get_init(step::QobjEvoStep{OP}) where OP
    n = step.n
    return ()->convert(OP, zeros(eltype(OP), n, n))
end

function set_params!(step::QobjEvoStep{OP,NParams,PT}, params::AbstractVector) where {OP,NParams,PT}
    @assert length(params) == NParams
    step.params = SVector{NParams,PT}(ntuple(k->PT(params[k]), Val(NParams)))
    return
end

end
