#

module TimeSequence

using StaticArrays

abstract type AbstractStep{OP,NParams} end

"""
    compute(step::AbstractStep{OP,NParams}, grad::AbstractVector{OP})::OP

Compute the operator for the `step`.
If `grad` is not empty, it should be an array of size `NParams` to store the
gradient WRT each parameter.
"""
function compute end

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

struct Sequence{OP,NSteps,Steps<:NTuple{NSteps,AbstractStep},NParams,ValBuf,GradBuf,PartialBuf,TmpBuf,Mul,Mul!} <: AbstractStep{OP,NParams}
    steps::Steps
    val_buf::ValBuf
    grad_buf::GradBuf
    prefix_buf::PartialBuf
    suffix_buf::PartialBuf
    tmp_buf::TmpBuf # [Result, intermediate for gradient]
    mul::Mul
    mul!::Mul!

    function Sequence{OP}(steps::Steps; @specialize(init=get_init(OP)),
                          mul::Mul=get_mul(OP), mul!::(Mul!)=get_mul!(OP)) where Steps<:NTuple{NSteps,AbstractStep{OP}} where {OP,NSteps,Mul,Mul!}
        @assert NSteps > 0
        if mul! !== nothing
            @assert init !== nothing
            @assert mul === nothing
        else
            @assert init === nothing
        end
        NParams = sum(nparams, steps)
        tmp_buf = nothing
        if init !== nothing
            val_buf = MVector(ntuple(_->init()::OP, NSteps))
            grad_buf = MVector(ntuple(_->init()::OP, NParams))
            prefix_buf = MVector(ntuple(_->init()::OP, NSteps - 1))
            suffix_buf = MVector(ntuple(_->init()::OP, NSteps - 1))
            tmp_buf = MVector(ntuple(_->init()::OP, 2))
        elseif isbitstype(OP)
            val_buf = MVector{NSteps,OP}(undef)
            grad_buf = MVector{NParams,OP}(undef)
            prefix_buf = MVector{NSteps - 1,OP}(undef)
            suffix_buf = MVector{NSteps - 1,OP}(undef)
        else
            val_buf = Vector{OP}(undef, NSteps)
            grad_buf = Vector{OP}(undef, NParams)
            prefix_buf = Vector{OP}(undef, NSteps - 1)
            suffix_buf = Vector{OP}(undef, NSteps - 1)
        end
        s = new{OP,NSteps,Steps,NParams,typeof(val_buf),typeof(grad_buf),
                typeof(prefix_buf),typeof(tmp_buf),Mul,Mul!}(
                    steps, val_buf, grad_buf, prefix_buf, suffix_buf,
                    tmp_buf, mul, mul!)
        return s
    end
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

@generated function _eval_compute(s::Sequence{OP}, grads) where OP
    ex = quote
        steps = s.steps
        grad_buf = s.grad_buf
        val_buf = s.val_buf
    end
    starts, ends = _param_range(s)
    ex1 = quote end
    ex2 = quote end
    for (i, (start_idx, end_idx)) in enumerate(zip(starts, ends))
        push!(ex1.args, :(
            @inbounds begin
                @inline val_buf[$i] = compute(steps[$i], SVector{0,OP}())
            end))
        push!(ex2.args, :(
            @inbounds begin
                @inline val_buf[$i] =
                    compute(steps[$i], @view grad_buf[$start_idx:$end_idx])
            end))
    end
    push!(ex.args, quote
              if isempty(grads)
                  $ex1
              else
                  $ex2
              end
              return
          end)
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
            v = $mul!($out, $a, $b)
        end
        v
    end
end

function compute(s::Sequence{OP,NSteps,Steps,NParams}, grads) where {OP,NSteps,Steps,NParams}
    if NSteps == 1
        return compute(s.steps[1], grads)
    end
    mul = s.mul
    mul! = s.mul!
    starts, ends = _param_range(typeof(s))
    @inline _eval_compute(s, grads)
    @inbounds s.prefix_buf[1] = prev = s.val_buf[1]
    @inbounds for i in 2:NSteps - 1
        prev = @pick_mulass(mul!, mul, s.prefix_buf[i],
                            prev, s.val_buf[i])
    end
    last_val = @inbounds s.val_buf[NSteps]
    res = @inbounds @pick_mul(mul!, mul, s.tmp_buf[1], prev, last_val)
    if isempty(grads)
        return res
    end
    @assert length(grads) == NParams
    @inbounds s.suffix_buf[NSteps - 1] = prev = last_val
    @inbounds for i in NSteps - 2:-1:1
        prev = @pick_mulass(mul!, mul, s.suffix_buf[i],
                            s.val_buf[i + 1], prev)
    end
    @inbounds for step_idx in 1:NSteps
        pstart = starts[step_idx]
        pend = ends[step_idx]
        if step_idx == 1
            suffix = s.suffix_buf[step_idx]
            for param_idx in pstart:pend
                @pick_mulass(mul!, mul, grads[param_idx], s.grad_buf[param_idx], suffix)
            end
        elseif step_idx == NSteps
            prefix = s.prefix_buf[step_idx - 1]
            for param_idx in pstart:pend
                @pick_mulass(mul!, mul, grads[param_idx], prefix, s.grad_buf[param_idx])
            end
        else
            prefix = s.prefix_buf[step_idx - 1]
            suffix = s.suffix_buf[step_idx]
            for param_idx in pstart:pend
                tmp = @pick_mul(mul!, mul, s.tmp_buf[2], prefix, s.grad_buf[param_idx])
                @pick_mulass(mul!, mul, grads[param_idx], tmp, suffix)
            end
        end
    end
    return res
end

end
