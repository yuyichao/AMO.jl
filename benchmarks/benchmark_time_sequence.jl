#

import AMO.TimeSequence as TS
using BenchmarkTools

mutable struct F64Ref
    v::Float64
end
TS.get_init(::Type{F64Ref}) = ()->F64Ref(0.0)
TS.get_mul(::Type{F64Ref}) = nothing
TS.get_mul!(::Type{F64Ref}) = (out::F64Ref, a::F64Ref, b::F64Ref)->(out.v = a.v * b.v)
Base.Float64(v::F64Ref) = v.v

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

NP = 12
params = rand(NP) .+ 0.4
VT = Float64
ops = (Step1{VT}(), Step2{VT}(), Step3{VT}(),
       Step1{VT}(), Step2{VT}(), Step3{VT}())
s1 = TS.Sequence{VT}(ops)
grads1_0 = VT[]
grads1 = zeros(NP)
TS.set_params!(s1, params)

VT = F64Ref
ops = (Step1{VT}(), Step2{VT}(), Step3{VT}(),
       Step1{VT}(), Step2{VT}(), Step3{VT}())
s2 = TS.Sequence{VT}(ops)
grads2_0 = VT[]
grads2 = [VT(0.0) for _ in 1:NP]
TS.set_params!(s2, params)

VT = F64Ref
ops = (IStep1{VT}(), IStep2{VT}(), IStep3{VT}(),
       IStep1{VT}(), IStep2{VT}(), IStep3{VT}())
s3 = TS.Sequence{VT}(ops)
grads3_0 = VT[]
grads3 = [VT(0.0) for _ in 1:NP]
TS.set_params!(s3, params)


NP = 24
params = rand(NP) .+ 0.4
VT = Float64
ops = (Step1{VT}(), Step2{VT}(), Step3{VT}(), Step1{VT}(),
       Step1{VT}(), Step2{VT}(), Step3{VT}(), Step2{VT}(),
       Step2{VT}(), Step1{VT}(), Step3{VT}(), Step3{VT}())
s4 = TS.Sequence{VT}(ops)
grads4_0 = VT[]
grads4 = zeros(NP)
TS.set_params!(s4, params)

VT = F64Ref
ops = (Step1{VT}(), Step2{VT}(), Step3{VT}(), Step1{VT}(),
       Step1{VT}(), Step2{VT}(), Step3{VT}(), Step2{VT}(),
       Step2{VT}(), Step1{VT}(), Step3{VT}(), Step3{VT}())
s5 = TS.Sequence{VT}(ops)
grads5_0 = VT[]
grads5 = [VT(0.0) for _ in 1:NP]
TS.set_params!(s5, params)

VT = F64Ref
ops = (IStep1{VT}(), IStep2{VT}(), IStep3{VT}(), IStep1{VT}(),
       IStep1{VT}(), IStep2{VT}(), IStep3{VT}(), IStep2{VT}(),
       IStep2{VT}(), IStep1{VT}(), IStep3{VT}(), IStep3{VT}())
s6 = TS.Sequence{VT}(ops)
grads6_0 = VT[]
grads6 = [VT(0.0) for _ in 1:NP]
TS.set_params!(s6, params)

function run_compute(s, grad, n)
    for _ in 1:n
        TS.compute(s, grad)
    end
end

function run_compute!(res, s, grad, n)
    for _ in 1:n
        TS.compute!(res, s, grad)
    end
end

println("S1")
@btime run_compute($s1, $grads1_0, 1000)
@btime run_compute($s1, $grads1, 1000)
println("S2")
@btime run_compute($s2, $grads2_0, 1000)
@btime run_compute($s2, $grads2, 1000)
@btime run_compute!($(F64Ref(0)), $s2, $grads2_0, 1000)
@btime run_compute!($(F64Ref(0)), $s2, $grads2, 1000)
println("S3")
@btime run_compute($s3, $grads3_0, 1000)
@btime run_compute($s3, $grads3, 1000)
@btime run_compute!($(F64Ref(0)), $s3, $grads3_0, 1000)
@btime run_compute!($(F64Ref(0)), $s3, $grads3, 1000)
println("S4")
@btime run_compute($s4, $grads4_0, 1000)
@btime run_compute($s4, $grads4, 1000)
println("S5")
@btime run_compute($s5, $grads5_0, 1000)
@btime run_compute($s5, $grads5, 1000)
@btime run_compute!($(F64Ref(0)), $s5, $grads5_0, 1000)
@btime run_compute!($(F64Ref(0)), $s5, $grads5, 1000)
println("S6")
@btime run_compute($s6, $grads6_0, 1000)
@btime run_compute($s6, $grads6, 1000)
@btime run_compute!($(F64Ref(0)), $s6, $grads6_0, 1000)
@btime run_compute!($(F64Ref(0)), $s6, $grads6, 1000)
