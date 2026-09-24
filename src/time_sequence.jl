#

module TimeSequence

using LinearAlgebra
using LinearAlgebra: BlasFloat
using StaticArrays

using ..Math: Imaginary, HermEigenWorkspace, heevd!

public AbstractStep, support_inplace_compute, compute, compute!, compute_right, compute_right!, set_params!,
    get_init, get_mul, get_mul!, Sequence, ConstMatrixStep, QobjEvoStep

abstract type AbstractStep{OP,NParams} end

@inline support_inplace_compute(T) = false

"""
    compute(step::AbstractStep{OP,NParams}, grad::AbstractVector{OP})::OP
    compute(step::AbstractStep{OP,NParams}, grad::AbstractVector, L, R)

Compute the operator `U` for the `step`.
If `grad` is not empty, it should be an array of size `NParams` to store the
gradient WRT each parameter.

In the second form, `L * U * R` is returned and `grad[k]` is set to
`L * ∂U/∂p_k * R`, where `L` and `R` are left/right multipliers (e.g. matrices,
possibly non-square, or vectors), or `nothing` for the identity. The multipliers allow
steps and sequences to avoid forming the full operator (e.g. when only matrix elements
or the evolution of a few states are needed). Step implementations must provide the
first form; the second form is optional but required for the step to be used in a
[`Sequence`](@ref) with multipliers, in which case it is called with exactly one of
the multipliers (the other being `nothing`). The returned object should have the type
of the corresponding product of the multipliers with the operator type (`L * U`,
`U * R` or `(L * U) * R`), which allows sequences to type their intermediate storage.
"""
function compute end

"""
    compute!(output::OP, step::AbstractStep{OP,NParams}, grad::AbstractVector{OP})
    compute!(output, step::AbstractStep{OP,NParams}, grad::AbstractVector, L, R)

Similar to `compute` but does everything in-place by mutating the object
in `output` and the objects in the `grad` array.
"""
function compute! end

"""
    compute_right(step::AbstractStep, grad, L, R, gradR)
    compute_right!(res, step::AbstractStep, grad, L, R, gradR)

Compute the right-multiplied operator `U * R` of the `step` (into `res` for the in-place
version) and set `grad[k]` to the gradient multiplied on both sides, `L * ∂U/∂p_k * R`.

This is the form used by [`Sequence`](@ref) for the gradients with left/right multipliers
(its only caller, so both multipliers are always given): the returned product is what is
needed to propagate the right multiplier through the sequence, while the gradients with
both multipliers can be much cheaper to evaluate than with either multiplier alone
(e.g. for [`ConstMatrixStep`](@ref), where the cost per parameter drops from `O(n³)` to
`O(n²)` for thin multipliers).

The default implementation uses the multiplied form of [`compute`](@ref) with `R` only
and then multiplies the gradients by `L`, for steps that do not benefit from evaluating
both multipliers at once. For this, `gradR` is an array of at least `length(grad)`
elements for the intermediate `∂U/∂p_k * R`, allocated by the sequence: buffers of the
shape of `U * R` for the in-place version, and an array of the appropriate element type
(possibly uninitialized) for the allocating version. Implementations that do not need
it may ignore it.
"""
function compute_right(step::AbstractStep, grad, L, R, gradR)
    gr = @view gradR[1:length(grad)]
    UR = compute(step, gr, nothing, R)
    @inbounds for k in eachindex(grad)
        grad[k] = L * gr[k]
    end
    return UR
end
function compute_right!(res, step::AbstractStep, grad, L, R, gradR)
    gr = @view gradR[1:length(grad)]
    compute!(res, step, gr, nothing, R)
    @inbounds for k in eachindex(grad)
        mul!(grad[k], L, gr[k])
    end
    return res
end

# An (uninitialized) object that can hold the product `A * B` (the output of
# `mul!(out, A, B)`), used to allocate the buffers of `Sequence` for the in-place
# computation with left/right multipliers.
_product_buffer(A::AbstractMatrix, B::AbstractMatrix) =
    similar(B, promote_type(eltype(A), eltype(B)), (size(A, 1), size(B, 2)))
_product_buffer(A::AbstractMatrix, B::AbstractVector) =
    similar(B, promote_type(eltype(A), eltype(B)), (size(A, 1),))

@inline _lmul(::Nothing, X) = X
@inline _lmul(L, X) = L * X
@inline _rmul(X, ::Nothing) = X
@inline _rmul(X, R) = X * R
@inline _lrmul(L, X, R) = _lmul(L, _rmul(X, R))
# out = L * M * R in-place (at least one multiplier), `T1` a scratch for `M * R`
@inline _rmul!(out, M, ::Nothing) = copyto!(out, M)
@inline _rmul!(out, M, R) = mul!(out, M, R)
@inline _lrmul!(out, ::Nothing, M, R, T1) = mul!(out, M, R)
@inline _lrmul!(out, L, M, ::Nothing, T1) = mul!(out, L, M)
@inline function _lrmul!(out, L, M, R, T1)
    mul!(T1, M, R)
    return mul!(out, L, T1)
end

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

# The sizes of the multipliers (`nothing` for non-arrays), their types are recorded as
# type parameters of the buffers below.
_lr_size(x) = x isa AbstractArray ? size(x) : nothing

# Buffers for the in-place computation with both left and right multipliers of a
# `Sequence`, allocated at construction from prototypes of the multipliers of types
# `LT` and `RT`.
struct _LRBuffers{LT,RT,SZ,PB,SB,TR}
    sizes::SZ          # sizes of the prototypes
    prefix::Vector{PB} # P_j = L * U_1 * ... * U_(j-1) for j in 2:N
    suf1::SB           # two alternating buffers for the suffixes U_j * ... * U_N * R
    suf2::SB
    gradR::Vector{SB}  # per parameter buffers for ∂U_j * R, for the steps' `compute_right!`
    res::TR            # result buffer for the allocating `compute`
end
function _LRBuffers(::Type{OP}, NSteps, NParams, init, L, R) where OP
    U = init()::OP
    prefix = [_product_buffer(L, U) for _ in 1:NSteps]
    suf1 = _product_buffer(U, R)
    suf2 = _product_buffer(U, R)
    gradR = [_product_buffer(U, R) for _ in 1:NParams]
    res = _product_buffer(@inbounds(prefix[1]), suf1)
    sizes = (_lr_size(L), _lr_size(R))
    return _LRBuffers{typeof(L),typeof(R),typeof(sizes),eltype(prefix),typeof(suf1),typeof(res)}(
        sizes, prefix, suf1, suf2, gradR, res)
end

# Buffers for a single left or right multiplier (the other one being `Nothing`): the
# full operator and gradients are computed and then multiplied.
struct _LRFullBuffers{LT,RT,SZ,OP,TR}
    sizes::SZ
    full_res::OP
    full_grads::Vector{OP}
    res::TR            # result buffer for the allocating `compute`
end
function _LRFullBuffers(::Type{OP}, NParams, init, L, R) where OP
    U = init()::OP
    res = L === nothing ? _product_buffer(U, R) : _product_buffer(L, U)
    sizes = (_lr_size(L), _lr_size(R))
    return _LRFullBuffers{typeof(L),typeof(R),typeof(sizes),OP,typeof(res)}(
        sizes, init()::OP, [init()::OP for _ in 1:NParams], res)
end

"""
    Sequence{OP}(steps; init=..., mul=..., mul!=..., left=nothing, right=nothing)

The time-ordered product `U = U_1 * U_2 * ... * U_N` of the operators of the `steps`
(a tuple of [`AbstractStep`](@ref)s with operator type `OP`), whose parameters are the
concatenated parameters of the steps. The operator and its gradient are computed with
[`compute`](@ref) / [`compute!`](@ref), with a cost linear in the number of steps and
the number of parameters.

The products use `mul` (allocating, default `get_mul(OP)`) or `mul!` (in-place, default
`get_mul!(OP)`, with `mul=nothing`). In-place multiplication requires `init`, a function
creating new `OP` objects for the buffers (default [`get_init`](@ref) of `OP` or of the
steps).

`left` and `right` are prototypes (objects of the same type and size) of the left/right
multipliers that will be passed to `compute`/`compute!` when using in-place
multiplication, from which the corresponding buffers are allocated. Give both for the
computation of `L * U * R`, or only one of them for `L * U` or `U * R`.
"""
struct Sequence{OP,NSteps,Steps<:NTuple{NSteps,AbstractStep},NParams,Init,Mul,Mul!,ValBuf,GradBuf,PartialBuf,TmpBuf,LRBuf}
    steps::Steps
    val_buf::ValBuf
    grad_buf::GradBuf
    prefix_buf::PartialBuf
    suffix_buf::PartialBuf
    tmp_buf::TmpBuf # [Result, intermediate for gradient]
    lr_buf::LRBuf   # `_LRBuffers`/`_LRFullBuffers` for left/right multipliers (in-place only)

    init::Init
    mul::Mul
    mul!::Mul!

    function Sequence{OP}(steps::Steps; init::Init=_default_init(OP, steps),
                          mul::Mul=get_mul(OP), mul!::(Mul!)=get_mul!(OP),
                          left=nothing, right=nothing) where Steps<:NTuple{NSteps,AbstractStep{OP}} where {OP,NSteps,Init,Mul,Mul!}
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
        lr_buf = if mul! === nothing || (left === nothing && right === nothing)
            nothing
        elseif left !== nothing && right !== nothing
            _LRBuffers(OP, NSteps, NParams, init, left, right)
        else
            _LRFullBuffers(OP, NParams, init, left, right)
        end
        s = new{OP,NSteps,Steps,NParams,Init,Mul,Mul!,typeof(val_buf),typeof(grad_buf),
                typeof(prefix_buf),typeof(tmp_buf),typeof(lr_buf)}(
                    steps, val_buf, grad_buf, prefix_buf, suffix_buf,
                    tmp_buf, lr_buf, init, mul, mul!)
        return s
    end
end

(nparams(::Type{<:Sequence{OP,NSteps,Steps,NParams}}) where {OP,NSteps,Steps,NParams}) = NParams
nparams(s::Sequence) = nparams(typeof(s))

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

# Computation with both left and right multipliers. Without gradients:
#   suffixes  S_j = U_j * S_(j+1)  (S_(N+1) = R)  for j = N, ..., 2,   result L * U_1 * S_2
# With gradients, each step is called at most twice and the last step once:
#   forward   P_(j+1) = P_j * U_j  (P_1 = L)      for j = 1, ..., N-1  (no gradients)
#   backward  S_j = U_j * S_(j+1) with the gradients P_j * ∂U_j * S_(j+1) (`compute_right`)
#                                                  for j = N, ..., 1
#   result    L * S_1
# Only thin products (with a multiplier on each side) are formed by the sequence itself.
function __lr_body_nograd(NSteps, inplace)
    body = Any[:nothing, :(S = R)]
    for j in NSteps:-1:2
        sbuf = isodd(j) ? :suf1 : :suf2
        push!(body, inplace ? quote
            if support_inplace_compute(typeof(steps[$j]))
                compute!($sbuf, steps[$j], dummy_grad, nothing, S)
                S = $sbuf
            else
                S = compute(steps[$j], dummy_grad, nothing, S)
            end
        end : :(S = compute(steps[$j], dummy_grad, nothing, S)))
    end
    push!(body, inplace ? quote
        if support_inplace_compute(typeof(steps[1]))
            compute!(prefix[2], steps[1], dummy_grad, L, nothing)
            P = prefix[2]
        else
            P = compute(steps[1], dummy_grad, L, nothing)
        end
        mul!(res, P, S)
    end : quote
        P = compute(steps[1], dummy_grad, L, nothing)
        res = mul(P, S)
    end)
    return body
end

function __lr_body_grad(starts, ends, NSteps, inplace)
    body = Any[:nothing, :(P = L)]
    for j in 1:NSteps - 1
        push!(body, inplace ? quote
            if support_inplace_compute(typeof(steps[$j]))
                compute!(prefix[$(j + 1)], steps[$j], dummy_grad, P, nothing)
                P = prefix[$(j + 1)]
            else
                P = compute(steps[$j], dummy_grad, P, nothing)
                prefix[$(j + 1)] = P
            end
        end : quote
            P = compute(steps[$j], dummy_grad, P, nothing)
            prefix[$(j + 1)] = P
        end)
    end
    push!(body, :(S = R))
    for j in NSteps:-1:1
        Pj = j == 1 ? :L : :(prefix[$j])
        gj = :(@view(grads[$(starts[j]):$(ends[j])]))
        sbuf = isodd(j) ? :suf1 : :suf2
        grj = :(@view(gradR[$(starts[j]):$(ends[j])]))
        push!(body, inplace ? quote
            if support_inplace_compute(typeof(steps[$j]))
                compute_right!($sbuf, steps[$j], $gj, $Pj, S, $grj)
                S = $sbuf
            else
                S = compute_right(steps[$j], $gj, $Pj, S, $grj)
            end
        end : :(S = compute_right(steps[$j], $gj, $Pj, S, $grj)))
    end
    push!(body, inplace ? :(mul!(res, L, S)) : :(res = mul(L, S)))
    return body
end

@generated function _compute_lr!(res, s::Sequence{OP,NSteps}, grads, L, R, prefix, suf1, suf2,
                                 gradR, ::Val{inplace}, ::Val{has_grad}) where {OP,NSteps,inplace,has_grad}
    starts, ends = _param_range(s)
    body = has_grad ? __lr_body_grad(starts, ends, NSteps, inplace) :
        __lr_body_nograd(NSteps, inplace)
    return quote
        steps = s.steps
        mul = s.mul
        mul! = s.mul!
        dummy_grad = SVector{0,OP}()
        @inbounds begin
            $(body...)
        end
        return res
    end
end

# Products with a single multiplier
@inline _mul_lr(mul, ::Nothing, X, R) = mul(X, R)
@inline _mul_lr(mul, L, X, ::Nothing) = mul(L, X)
@inline _mul_lr!(mul!, out, ::Nothing, X, R) = mul!(out, X, R)
@inline _mul_lr!(mul!, out, L, X, ::Nothing) = mul!(out, L, X)

# The buffers of the sequence for the multipliers `L` and `R`, checked against the
# prototypes: the types by dispatch, the sizes at runtime.
function _lr_buffers(s::Sequence, L, R)
    lr = s.lr_buf
    if lr === nothing
        throw(ArgumentError("Left/right multipliers with in-place multiplication require the `left`/`right` prototypes at construction"))
    end
    return _check_lr_buffers(lr, L, R)
end
function _check_lr_buffers(lr::Union{_LRBuffers{LT,RT},_LRFullBuffers{LT,RT}}, L::LT, R::RT) where {LT,RT}
    if lr.sizes != (_lr_size(L), _lr_size(R))
        throw(ArgumentError("The left/right multipliers must have the size of the `left`/`right` prototypes given at construction"))
    end
    return lr
end
_check_lr_buffers(lr, L, R) =
    throw(ArgumentError("The left/right multipliers must have the type of the `left`/`right` prototypes given at construction"))

"""
    compute(s::Sequence, grads, L, R)
    compute!(res, s::Sequence, grads, L, R)

Compute `L * U * R` for the operator `U` of the sequence and the corresponding
gradients `L * ∂U/∂p_k * R` (see [`compute`](@ref)). `L` and `R` may be `nothing`
(identity) or any objects supported by the multiplied forms of `compute` of the steps
and by the multiplication of the sequence.

When both multipliers are given, the full operators of the steps are never formed:
each step is called at most twice (once with the left multiplier to propagate it, and
once with [`compute_right`](@ref) to propagate the right multiplier and evaluate the
gradients with both multipliers; the last step only once), and the sequence only forms
thin products. This is much cheaper than computing the full operator when `L` and `R`
are "thin" (e.g. for a matrix element or a state). The cost scales with the number of
steps plus the number of parameters.
When only one multiplier is given, the full operator and gradients are computed and
then multiplied.

With in-place multiplication, the sequence must be constructed with the prototypes
`left`/`right` of the multipliers that are used (see [`Sequence`](@ref)), and the
allocating version returns an internal buffer.
"""
function compute(s::Sequence{OP,NSteps,Steps,NParams}, grads, L, R) where {OP,NSteps,Steps,NParams}
    if L === nothing && R === nothing
        return compute(s, grads)
    end
    if NSteps == 1
        return compute(s.steps[1], grads, L, R)
    end
    if s.mul! !== nothing
        lr = _lr_buffers(s, L, R)
        return compute!(lr.res, s, grads, L, R)
    end
    has_grad = !isempty(grads)
    has_grad && @assert length(grads) == NParams
    if L === nothing || R === nothing
        # Full operator and gradients, then multiplied
        full = Vector{OP}(undef, has_grad ? NParams : 0)
        U = compute(s, full)
        mul = s.mul
        @inbounds for k in eachindex(full)
            grads[k] = _mul_lr(mul, L, full[k], R)
        end
        return _mul_lr(mul, L, U, R)
    end
    if has_grad
        # Arrays for the prefixes and for the intermediate gradients of the steps'
        # `compute_right` fallback (allocated once per call), typed from the products
        # of the multipliers with the operator type (see `compute`)
        mul = s.mul
        prefix = Vector{Base.promote_op(mul, typeof(L), OP)}(undef, NSteps)
        gradR = Vector{Base.promote_op(mul, OP, typeof(R))}(undef, NParams)
        return _compute_lr!(nothing, s, grads, L, R, prefix, nothing, nothing, gradR,
                            Val(false), Val(true))
    end
    return _compute_lr!(nothing, s, grads, L, R, nothing, nothing, nothing, nothing,
                        Val(false), Val(false))
end

function compute!(res, s::Sequence{OP,NSteps,Steps,NParams}, grads, L, R) where {OP,NSteps,Steps,NParams}
    if L === nothing && R === nothing
        return compute!(res, s, grads)
    end
    if NSteps == 1
        return compute!(res, s.steps[1], grads, L, R)
    end
    if s.mul! === nothing
        throw(ArgumentError("In-place compute with left/right multipliers requires in-place multiplication (`mul!`)"))
    end
    if !isempty(grads)
        @assert length(grads) == NParams
    end
    return _compute_lr_buffered!(res, s, grads, L, R, _lr_buffers(s, L, R))
end

function _compute_lr_buffered!(res, s::Sequence, grads, L, R, lr::_LRBuffers)
    if isempty(grads)
        _compute_lr!(res, s, grads, L, R, lr.prefix, lr.suf1, lr.suf2, lr.gradR,
                     Val(true), Val(false))
    else
        _compute_lr!(res, s, grads, L, R, lr.prefix, lr.suf1, lr.suf2, lr.gradR,
                     Val(true), Val(true))
    end
    return res
end

function _compute_lr_buffered!(res, s::Sequence, grads, L, R, fb::_LRFullBuffers)
    mul! = s.mul!
    if isempty(grads)
        compute!(fb.full_res, s, grads)
    else
        full_grads = fb.full_grads
        compute!(fb.full_res, s, full_grads)
        @inbounds for k in eachindex(grads)
            _mul_lr!(mul!, grads[k], L, full_grads[k], R)
        end
    end
    _mul_lr!(mul!, res, L, fb.full_res, R)
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
struct _TIBuffers{T,MT<:AbstractMatrix{T},VT<:AbstractVector,EW,GE}
    A::MT    # assembled generator
    Φ::MT    # divided differences of exp in the eigenbasis
    tmp1::MT
    tmp2::MT
    h::VT    # exp(-im * t * λ / 2)
    eλ::VT   # exp(-im * t * λ)
    eig::EW  # `HermEigenWorkspace` for dense matrices with BLAS element types, else `nothing`
    geig::GE # cache of the general eigendecomposition (λ, V, V⁻¹) for dense matrices
    scratch::Base.RefValue{Memory{T}} # scratch memory for the left/right multiplied products
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
For dense matrices with BLAS element types, the Hermitian eigendecomposition uses a
preallocated workspace and is cached until the parameters change.
The step provides [`get_init`](@ref), so a [`Sequence`](@ref) of these steps
can be constructed without an explicit `init`.

The left/right multiplied forms `compute(step, grad, L, R)` and
`compute!(res, step, grad, L, R)` are supported for `L`, `R` being `nothing` or arrays
(matrices, or a vector for `R`), and are computed directly from the eigenfactors
without forming the full operator. For few rows of `L` and columns of `R`,
the cost of the gradients is dominated by one inner product of two `n × n` matrices
per parameter.
"""
mutable struct ConstMatrixStep{OP<:AbstractMatrix,NParams,NH,PT,Herm,H0T,Buf} <: AbstractStep{OP,NParams}
    const H0::H0T
    const Hs::NTuple{NH,OP}
    coeffs::SVector{NH,PT}
    t::PT
    const buf::Buf
    eig_valid::Bool # whether the eigendecomposition in `buf.eig` is that of the current `coeffs`

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
            eig = (OP <: Matrix && T <: BlasFloat && hermitian) ? HermEigenWorkspace{T}(n) : nothing
            geig = (OP <: Matrix && !hermitian) ?
                Ref{Union{Nothing,Tuple{Vector{T},Matrix{T},Matrix{T}}}}(nothing) : nothing
            buf = _TIBuffers(similar(ref, T), similar(ref, T), similar(ref, T), similar(ref, T),
                             similar(ref, T, n), similar(ref, T, n), eig, geig,
                             Ref(Memory{T}(undef, 0)))
        else
            buf = nothing
        end
        return new{OP,NParams,NH,PT,hermitian,typeof(H0),typeof(buf)}(
            H0, Hs, zero(SVector{NH,PT}), t, buf, false)
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
    step.coeffs = SVector{NH,PT}(ntuple(k->PT(@inbounds(params[k])), Val(NH)))
    if NParams > NH
        step.t = PT(@inbounds params[NParams])
    end
    step.eig_valid = false
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
# Static matrices: the dense decomposition is converted back to static types so that
# the results keep the types of the products with the operator.
@inline function _eigen_decomp(A::StaticMatrix{N,N}, ::Val{false}) where N
    E = eigen(convert(Matrix, A))
    V = E.vectors
    return SVector{N}(E.values), SMatrix{N,N}(V), SMatrix{N,N}(inv(V))
end

# In-place variants for `compute!`: use the preallocated LAPACK workspace when available,
# and a buffer for the input copy of the general eigendecomposition.
@inline _eigen_decomp!(step::ConstMatrixStep, buf::_TIBuffers, A, ::Val{true}) =
    _eigen_decomp_herm!(step, buf.eig, A)
@inline function _eigen_decomp_herm!(step::ConstMatrixStep, ws::HermEigenWorkspace, A)
    # The eigendecomposition only depends on the coefficients, reuse it when valid
    if step.eig_valid
        V = ws.V
        return ws.λ, V, V'
    end
    λ, V = heevd!(ws, A)
    step.eig_valid = true
    return λ, V, V'
end
@inline _eigen_decomp_herm!(::ConstMatrixStep, ::Nothing, A) = _eigen_decomp(A, Val(true))
@inline function _eigen_decomp!(step::ConstMatrixStep, buf::_TIBuffers, A::Matrix, ::Val{false})
    cache = buf.geig
    if step.eig_valid
        c = cache[]
        if c !== nothing
            return c
        end
    end
    # `Φ` is not needed until after the eigendecomposition
    E = eigen!(copyto!(buf.Φ, A))
    V = E.vectors
    c = (E.values, V, inv(V))
    cache[] = c
    step.eig_valid = true
    return c
end
@inline _eigen_decomp!(::ConstMatrixStep, buf::_TIBuffers, A, ::Val{false}) =
    _eigen_decomp(A, Val(false))

@inline _to_op(::Type{OP}, m) where OP = convert(OP, m)

@inline function _assemble(step::ConstMatrixStep{OP,NParams,NH}) where {OP,NParams,NH}
    H0 = step.H0
    if NH == 0
        return H0
    end
    Hs = step.Hs
    coeffs = step.coeffs
    A = @inbounds coeffs[1] * Hs[1]
    @inbounds for k in 2:NH
        A = muladd.(coeffs[k], Hs[k], A)
    end
    if H0 !== nothing
        A = A + H0
    end
    return A
end

##### Closed form for 2x2 matrices
#
# With `s A = x0 I + x⃗·σ⃗` and `r² = x⃗·x⃗`,
#     exp(s A) = e^x0 (f0 I + f1 x⃗·σ⃗),   f0 = cosh(r),  f1 = sinh(r) / r.
# Since this is an exact analytic expression in the matrix entries, its derivative along
# a direction `E = e0 I + e⃗·σ⃗` is the Fréchet derivative of the exponential:
#     e^x0 [(e0 f0 + f1 x⃗·e⃗) I + (e0 f1 + f2 x⃗·e⃗) x⃗·σ⃗ + f1 e⃗·σ⃗],   f2 = (f0 - f1) / r².
# For a Hermitian generator `r² = -t² |a⃗|²` and all the `f`'s are real trigonometric functions.

# Taylor coefficients in r² of f0, f1 and f2
const _EXPM2_F0 = Tuple(Float64(1 // factorial(big(2n))) for n in 0:7)
const _EXPM2_F1 = Tuple(Float64(1 // factorial(big(2n + 1))) for n in 0:7)
const _EXPM2_F2 = Tuple(Float64((2n + 2) // factorial(big(2n + 3))) for n in 0:7)

@inline function _expm2_coefs_direct(r2::Real)
    if r2 < 0
        θ = sqrt(-r2)
        sθ, cθ = sincos(θ)
        f1 = sθ / θ
        return cθ, f1, (cθ - f1) / r2
    end
    r = sqrt(r2)
    f0 = cosh(r)
    f1 = sinh(r) / r
    return f0, f1, (f0 - f1) / r2
end
@inline function _expm2_coefs_direct(r2::Complex)
    r = sqrt(r2)
    f0 = cosh(r)
    f1 = sinh(r) / r
    return f0, f1, (f0 - f1) / r2
end
# f0, f1, f2 as (even) functions of r². The series avoids the cancellation in f2 near 0.
@inline function _expm2_coefs(r2)
    if abs(r2) < 0.25
        return evalpoly(r2, _EXPM2_F0), evalpoly(r2, _EXPM2_F1), evalpoly(r2, _EXPM2_F2)
    end
    return _expm2_coefs_direct(r2)
end

# Pauli decomposition A = a0 I + ax σx + ay σy + az σz
@inline function _pauli2(A, ::Val{true})
    # Hermitian: real coefficients from the upper triangle
    a11 = real(@inbounds A[1, 1])
    a22 = real(@inbounds A[2, 2])
    a12 = @inbounds A[1, 2]
    return (a11 + a22) / 2, real(a12), -imag(a12), (a11 - a22) / 2
end
@inline function _pauli2(A, ::Val{false})
    a11 = @inbounds A[1, 1]
    a22 = @inbounds A[2, 2]
    a12 = @inbounds A[1, 2]
    a21 = @inbounds A[2, 1]
    return (a11 + a22) / 2, (a12 + a21) / 2, Imaginary(-1 / 2) * (a21 - a12), (a11 - a22) / 2
end
# Entries (column major) of c0 I + cx σx + cy σy + cz σz
@inline _pauli2_entries(c0, cx, cy, cz) =
    (c0 + cz, cx + Imaginary(1) * cy, cx + _NEG_IM * cy, c0 - cz)

# Entries of exp(s * A) for a 2x2 matrix `A`, and the quantities needed for the derivatives
@inline function _expm2(A, s, ::Val{Herm}) where Herm
    a0, ax, ay, az = _pauli2(A, Val(Herm))
    s2 = s * s
    r2 = s2 * muladd(ax, ax, muladd(ay, ay, az * az))
    f0, f1, f2 = _expm2_coefs(r2)
    ex0 = exp(s * a0)
    g = ex0 * (f1 * s)
    U = _pauli2_entries(ex0 * f0, g * ax, g * ay, g * az)
    return U, (ax, ay, az, s2, f0, f1, f2, ex0)
end
# Entries of the Fréchet derivative of exp(s * A) in the direction s * H
@inline function _expm2_frechet(H, s, ::Val{Herm}, (ax, ay, az, s2, f0, f1, f2, ex0)) where Herm
    h0, hx, hy, hz = _pauli2(H, Val(Herm))
    xe = s2 * muladd(ax, hx, muladd(ay, hy, az * hz))
    e0 = s * h0
    d0 = ex0 * (e0 * f0 + f1 * xe)
    β = e0 * f1 + f2 * xe
    γ = ex0 * s
    return _pauli2_entries(d0, γ * muladd(β, ax, f1 * hx), γ * muladd(β, ay, f1 * hy),
                           γ * muladd(β, az, f1 * hz))
end
@inline function _set2!(M, (m11, m21, m12, m22))
    @inbounds M[1, 1] = m11
    @inbounds M[2, 1] = m21
    @inbounds M[1, 2] = m12
    @inbounds M[2, 2] = m22
    return M
end

function _compute2(step::ConstMatrixStep{OP,NParams,NH,PT,Herm}, A, s, grad) where {OP,NParams,NH,PT,Herm}
    U, parts = _expm2(A, s, Val(Herm))
    Um = _to_op(OP, SMatrix{2,2}(U))
    if !isempty(grad)
        @assert length(grad) == NParams
        Hs = step.Hs
        @inbounds for k in 1:NH
            grad[k] = _to_op(OP, SMatrix{2,2}(_expm2_frechet(Hs[k], s, Val(Herm), parts)))
        end
        if NParams > NH
            @inbounds grad[NParams] = _to_op(OP, _NEG_IM .* (A * Um))
        end
    end
    return Um
end

function _compute2!(res::OP, step::ConstMatrixStep{OP,NParams,NH,PT,Herm}, A, s, grad) where {OP,NParams,NH,PT,Herm}
    U, parts = _expm2(A, s, Val(Herm))
    _set2!(res, U)
    if !isempty(grad)
        @assert length(grad) == NParams
        Hs = step.Hs
        @inbounds for k in 1:NH
            _set2!(grad[k], _expm2_frechet(Hs[k], s, Val(Herm), parts))
        end
        if NParams > NH
            mul!(@inbounds(grad[NParams]), A, res, -im, false)
        end
    end
    return res
end

function compute(step::ConstMatrixStep{OP,NParams,NH,PT,Herm}, grad) where {OP,NParams,NH,PT,Herm}
    A = _assemble(step)
    s = Imaginary(-step.t)
    if size(A) == (2, 2)
        return _compute2(step, A, s, grad)
    end
    λ, V, W = _eigen_decomp(A, Val(Herm))
    h = exp.((s / 2) .* λ)
    eλ = h .* h
    U = (V .* transpose(eλ)) * W
    if !isempty(grad)
        @assert length(grad) == NParams
        Φ = _phi.(s, h, transpose(h), λ, transpose(λ))
        Hs = step.Hs
        @inbounds for k in 1:NH
            grad[k] = _to_op(OP, V * (Φ .* (W * Hs[k] * V)) * W)
        end
        if NParams > NH
            @inbounds grad[NParams] = _to_op(OP, _NEG_IM .* (A * U))
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
    @inbounds for k in 1:NH
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

# Left/right multiplied operator and gradients through the eigenfactors:
#     L U R = (L V) e^{sλ} (V⁻¹ R)
#     L ∂ₖU R = (L V) (Φ ∘ (V⁻¹ Hₖ V)) (V⁻¹ R)
# For few rows/columns of `L`/`R`, the gradients can be computed without any
# `n × n` product per parameter: with `Wab = (L V)[a, :] Φ (V⁻¹ R)[:, b]ᵀ` (elementwise),
#     (L ∂ₖU R)[a, b] = Σ_cd Wab[c, d] (V⁻¹ Hₖ V)[c, d] = Σ_pq (V⁻ᵀ Wab Vᵀ)[p, q] Hₖ[p, q]
# (with V⁻ᵀ the transpose of V⁻¹),
# so that each parameter only costs an inner product of two `n × n` matrices.
function compute(step::ConstMatrixStep, grad, L, R)
    if L === nothing && R === nothing
        return compute(step, grad)
    end
    return _compute_lr(step, grad, L, R, Val(false))
end
# The gradient buffers are not needed (the gradients are evaluated with both multipliers)
compute_right(step::ConstMatrixStep, grad, L, R, gradR) = _compute_lr(step, grad, L, R, Val(true))

# `RightOnly` selects the returned operator: `U * R` (`true`) or `L * U * R` (`false`);
# the gradients are always multiplied on both sides.
function _compute_lr(step::ConstMatrixStep{OP,NParams,NH,PT,Herm}, grad, L, R,
                     ::Val{RightOnly}) where {OP,NParams,NH,PT,Herm,RightOnly}
    A = _assemble(step)
    s = Imaginary(-step.t)
    if size(A) == (2, 2)
        # Closed form (converted to the operator type so that the products with the
        # multipliers have the same types as for the full operator), then multiply
        U, parts = _expm2(A, s, Val(Herm))
        Um = _to_op(OP, SMatrix{2,2}(U))
        if !isempty(grad)
            @assert length(grad) == NParams
            Hs = step.Hs
            @inbounds for k in 1:NH
                D = _to_op(OP, SMatrix{2,2}(_expm2_frechet(Hs[k], s, Val(Herm), parts)))
                grad[k] = _lrmul(L, D, R)
            end
            if NParams > NH
                @inbounds grad[NParams] = _lrmul(L, _NEG_IM .* (A * Um), R)
            end
        end
        return RightOnly ? _rmul(Um, R) : _lrmul(L, Um, R)
    end
    λ, V, W = _eigen_decomp(A, Val(Herm))
    hv = exp.((s / 2) .* λ)
    eλ = hv .* hv
    # The results are formed by products with the multipliers as the last operations,
    # so that their types are those of the corresponding products with the operator.
    Ve = V .* transpose(eλ) # V * Diagonal(eλ)
    AL = _lmul(L, V)
    BR = _rmul(W, R)
    U = RightOnly ? Ve * BR : _lmul(L, Ve) * BR
    if !isempty(grad)
        @assert length(grad) == NParams
        Φ = _phi.(s, hv, transpose(hv), λ, transpose(λ))
        Hs = step.Hs
        @inbounds for k in 1:NH
            grad[k] = AL * (Φ .* (W * Hs[k] * V)) * BR
        end
        if NParams > NH
            # -im * L * A * U * R = (L * A) * (V * Diagonal(-im * eλ)) * (W * R)
            Vt = V .* transpose(_NEG_IM .* eλ)
            @inbounds grad[NParams] = _lmul(_lmul(L, A), Vt) * BR
        end
    end
    return U
end

# Arrays of the given sizes backed by the scratch memory of the step (reallocated when
# too small). Plain `Array`s are used rather than reshaped views so that indexing and
# BLAS calls take their fast paths.
function _scratch_memory!(buf::_TIBuffers{T}, total) where T
    mem = buf.scratch[]
    if length(mem) < total
        mem = Memory{T}(undef, total)
        buf.scratch[] = mem
    end
    return mem
end
function _scratch_arrays(buf::_TIBuffers{T}, dims::Vararg{Dims,N}) where {T,N}
    # `mem` is assigned once so that the closure below does not box it
    mem = _scratch_memory!(buf, sum(prod, dims))
    offsets = cumsum((0, map(prod, dims)...))
    return ntuple(i->_wrap_array(mem, offsets[i] + 1, dims[i]), Val(N))
end
@inline _wrap_array(mem::Memory{T}, offset, dims::Dims{N}) where {T,N} =
    Base.wrap(Array, memoryref(mem, offset), dims)::Array{T,N}

function compute!(res, step::ConstMatrixStep, grad, L, R)
    if L === nothing && R === nothing
        return compute!(res, step, grad)
    end
    return _compute_lr!(res, step, grad, L, R, Val(false))
end
compute_right!(res, step::ConstMatrixStep, grad, L, R, gradR) =
    _compute_lr!(res, step, grad, L, R, Val(true))

function _compute_lr!(res, step::ConstMatrixStep{OP,NParams,NH,PT,Herm}, grad, L, R,
                      ::Val{RightOnly}) where {OP,NParams,NH,PT,Herm,RightOnly}
    buf = step.buf
    if buf === nothing
        throw(ArgumentError("In-place compute is not supported for immutable matrix type $OP"))
    end
    A = _assemble!(buf.A, step)
    s = Imaginary(-step.t)
    n = size(A, 1)
    k = R === nothing ? n : size(R, 2)
    rdims = R isa AbstractVector ? (n,) : (n, k)
    if n == 2
        # Closed form into the 2x2 buffers, then multiply
        U, parts = _expm2(A, s, Val(Herm))
        tmp1 = _set2!(buf.tmp1, U)
        tmp2 = buf.tmp2
        T1 = R === nothing ? nothing : _scratch_arrays(buf, rdims)[1]
        if RightOnly
            _rmul!(res, tmp1, R)
        else
            _lrmul!(res, L, tmp1, R, T1)
        end
        if !isempty(grad)
            @assert length(grad) == NParams
            Hs = step.Hs
            @inbounds for kk in 1:NH
                _set2!(tmp2, _expm2_frechet(Hs[kk], s, Val(Herm), parts))
                _lrmul!(grad[kk], L, tmp2, R, T1)
            end
            if NParams > NH
                mul!(tmp2, A, tmp1, -im, false)
                _lrmul!(@inbounds(grad[NParams]), L, tmp2, R, T1)
            end
        end
        return res
    end
    λ, V, W = _eigen_decomp!(step, buf, A, Val(Herm))
    hv = buf.h
    eλ = buf.eλ
    hv .= exp.((s / 2) .* λ)
    eλ .= hv .* hv
    m = L === nothing ? n : size(L, 1)
    ALbuf, ALe, BRbuf, T1 = _scratch_arrays(buf, (m, n), (m, n), rdims, rdims)
    AL = L === nothing ? V : mul!(ALbuf, L, V)
    BR = R === nothing ? W : mul!(BRbuf, W, R)
    if RightOnly
        tmp1 = copyto!(buf.tmp1, V)
        rmul!(tmp1, Diagonal(eλ))
        mul!(res, tmp1, BR)
    else
        copyto!(ALe, AL)
        rmul!(ALe, Diagonal(eλ))
        mul!(res, ALe, BR)
    end
    if !isempty(grad)
        @assert length(grad) == NParams
        Φ = _fill_phi!(buf.Φ, s, hv, λ)
        Hs = step.Hs
        tmp1 = buf.tmp1
        tmp2 = buf.tmp2
        if m * k < NH
            # Per matrix element: tmp2 = W' conj(Wab) V' = conj(Wᵀ Wab Vᵀ),
            # so that Σ_pq (Wᵀ Wab Vᵀ)[p, q] Hₖ[p, q] = dot(tmp2, Hₖ)
            @inbounds for b in 1:k, a in 1:m
                for d in 1:n, c in 1:n
                    tmp2[c, d] = conj(AL[a, c] * Φ[c, d] * BR[d, b])
                end
                mul!(tmp1, tmp2, V')
                mul!(tmp2, W', tmp1)
                for kk in 1:NH
                    grad[kk][a, b] = dot(tmp2, Hs[kk])
                end
            end
        else
            @inbounds for kk in 1:NH
                mul!(tmp1, Hs[kk], V)
                mul!(tmp2, W, tmp1)
                tmp2 .*= Φ
                mul!(T1, tmp2, BR)
                mul!(grad[kk], AL, T1)
            end
        end
        if NParams > NH
            # -im * L * A * U * R = -im * (L A V) e^{sλ} (V⁻¹ R)
            if L === nothing
                mul!(ALe, A, V)
            else
                mul!(ALbuf, L, A)
                mul!(ALe, ALbuf, V)
            end
            rmul!(ALe, Diagonal(eλ))
            mul!(@inbounds(grad[NParams]), ALe, BR, -im, false)
        end
    end
    return res
end

function compute!(res::OP, step::ConstMatrixStep{OP,NParams,NH,PT,Herm}, grad) where {OP,NParams,NH,PT,Herm}
    buf = step.buf
    if buf === nothing
        throw(ArgumentError("In-place compute is not supported for immutable matrix type $OP"))
    end
    A = _assemble!(buf.A, step)
    s = Imaginary(-step.t)
    if size(A) == (2, 2)
        return _compute2!(res, step, A, s, grad)
    end
    λ, V, W = _eigen_decomp!(step, buf, A, Val(Herm))
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
        @inbounds for k in 1:NH
            mul!(tmp1, Hs[k], V)
            mul!(tmp2, W, tmp1)
            tmp2 .*= Φ
            mul!(tmp1, V, tmp2)
            mul!(grad[k], tmp1, W)
        end
        if NParams > NH
            # -im * A * U. The scaling is done by BLAS for strided matrices.
            mul!(@inbounds(grad[NParams]), A, res, -im, false)
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

The propagator and the sensitivity equations ``∂_t (∂_k U) = -i (H ∂_k U + (∂_k H) U)``
(where ``∂_k H`` is obtained by forward-mode automatic differentiation of the coefficient
functions) are integrated column by column with ODE integrators that are initialized once
at construction and reused. With left/right multipliers, only the multipliers are
propagated: the columns of `R` (and their sensitivities) as states, and the rows of `L`
as states of the time-reversed transposed generator (whose propagator is ``U^T``), so that
the cost scales with the number of rows and columns of the multipliers instead of the
dimension. The accuracy of both the operator and its gradient are therefore determined by
the ODE solver options, which can be passed as keyword arguments (`alg`, default
`Vern7(lazy=false)` as for `QuantumToolbox.sesolve`, and e.g. `reltol`, `abstol`,
forwarded to the integrator).

`OP` is the matrix type of the result (e.g. `Matrix{ComplexF64}` or
`SMatrix{2,2,ComplexF64,4}`) and defaults to a dense `Matrix` with the complex element
type of `H`. The step provides [`get_init`](@ref), so a [`Sequence`](@ref) of these steps
can be constructed without an explicit `init`. Left/right multipliers (`nothing` or
arrays, like for [`ConstMatrixStep`](@ref)) are applied to the full operator and
gradients.
"""
mutable struct QobjEvoStep{OP<:AbstractMatrix,NParams,T,PT,HF,HR,GF,GR,KW,QB} <: AbstractStep{OP,NParams}
    const H_fwd::HF   # ODE generator -im * H (n × n)
    const H_rev::HR   # time-reversed and transposed generator, whose propagator is Uᵀ
    const G_fwd::GF   # sensitivity system ((K+1)n) for a state and its gradients
    const G_rev::GR   # the same for the reversed generator
    const n::Int      # size of the operator
    const t0::PT
    const t1::PT
    const kwargs::KW  # solver options
    const buf::QB     # ODE integrators and work buffers (see the extension)
    params::SVector{NParams,PT}
    # Cached full propagator and full gradients ([vec(U); vec(∂₁U); ...]) for the current
    # parameters (the multiplied forms propagate only the multipliers and are not cached)
    const U_cache::Matrix{T}
    const Y_cache::Vector{T}
    U_valid::Bool
    Y_valid::Bool
end

support_inplace_compute(::Type{<:QobjEvoStep{OP}}) where OP = ismutabletype(OP)

function get_init(step::QobjEvoStep{OP}) where OP
    n = step.n
    return ()->convert(OP, zeros(eltype(OP), n, n))
end

function set_params!(step::QobjEvoStep{OP,NParams,T,PT}, params::AbstractVector) where {OP,NParams,T,PT}
    @assert length(params) == NParams
    step.params = SVector{NParams,PT}(ntuple(k->PT(@inbounds(params[k])), Val(NParams)))
    step.U_valid = false
    step.Y_valid = false
    return
end

end
