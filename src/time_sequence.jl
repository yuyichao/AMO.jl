#

module TimeSequence

using StaticArrays

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

get_init(T) = nothing
get_mul(T) = *
get_mul!(T) = nothing

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

    function Sequence{OP}(steps::Steps; init::Init=get_init(OP),
                          mul::Mul=get_mul(OP), mul!::(Mul!)=get_mul!(OP)) where Steps<:NTuple{NSteps,AbstractStep{OP}} where {OP,NSteps,Init,Mul,Mul!}
        @assert NSteps > 0
        if mul! !== nothing
            @assert init !== nothing
            @assert mul === nothing
        end
        NParams = sum(nparams, steps)
        all_inplace = all(s->support_inplace_compute(typeof(s)), steps)
        op_isbits = isbitstype(OP)

        _op_vec(n) = init !== nothing ? [init()::OP for _ in 1:n] : Vector{OP}(undef, n)
        _op_mvec(n) = (init !== nothing ? MVector(ntuple(_->init()::OP, n)) :
            MVector{n,OP}(undef))
        op_vec(n, assign) = if n <= 0
            return nothing
        elseif !op_isbits
            return _op_vec(n)
        else
            return _op_mvec(n)
        end

        val_buf = op_vec(NSteps, !all_inplace)
        grad_buf = op_vec(NParams, !all_inplace)
        prefix_buf = op_vec(NSteps - 2, mul! === nothing)
        suffix_buf = op_vec(NSteps - 2, mul! === nothing)
        tmp_buf = op_vec(2, false)
        s = new{OP,NSteps,Steps,NParams,Init,Mul,Mul!,typeof(val_buf),typeof(grad_buf),
                typeof(prefix_buf),typeof(tmp_buf)}(
                    steps, val_buf, grad_buf, prefix_buf, suffix_buf,
                    tmp_buf, init, mul, mul!)
        return s
    end
end

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

@generated function _eval_compute(s::Sequence{OP}, grads, has_grad) where OP
    ex = quote
        steps = s.steps
        grad_buf = s.grad_buf
        val_buf = s.val_buf
    end
    starts, ends = _param_range(s)
    ex1 = quote end
    ex2 = quote end
    for (i, (start_idx, end_idx)) in enumerate(zip(starts, ends))
        push!(ex.args, :(
            @inbounds @inline begin
                step = steps[$i]
                stepgrad = (has_grad ? @view(grad_buf[$start_idx:$end_idx]) :
                    SVector{0,OP}())
                if support_inplace_compute(typeof(step))
                    compute!(val_buf[$i], step, stepgrad)
                else
                    val_buf[$i] = compute(step, stepgrad)
                end
            end))
    end
    return ex
end

macro pick_mul(mul!, mul, out, a, b)
    mul! = esc(mul!)
    mul = esc(mul)
    out = esc(out)
    a = esc(a)
    b = esc(b)
    quote
        if $mul! === nothing
            $mul($a, $b)
        else
            v = $out
            $mul!(v, $a, $b)
            v
        end
    end
end

macro pick_mulass(mul!, mul, out, a, b)
    mul! = esc(mul!)
    mul = esc(mul)
    out = esc(out)
    a = esc(a)
    b = esc(b)
    quote
        if $mul! === nothing
            v = $mul($a, $b)
            $out = v
        else
            v = $out
            $mul!(v, $a, $b)
        end
        v
    end
end

@inline function _eval_grads(s::Sequence{OP,NSteps,Steps,NParams}, grads, mul, mul!, first_val, last_val) where {OP,NSteps,Steps,NParams}
    @assert length(grads) == NParams
    prev = last_val
    @inbounds for i in NSteps - 2:-1:1
        prev = @pick_mulass(mul!, mul, s.suffix_buf[i], s.val_buf[i + 1], prev)
    end
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
                tmp = @pick_mul(mul!, mul, s.tmp_buf[2], prefix, s.grad_buf[param_idx])
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
    res = @inbounds @pick_mul(mul!, mul, s.tmp_buf[1], prev, last_val)
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

end
