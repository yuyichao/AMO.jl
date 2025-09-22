#

module TestTimeSequence

import AMO.TimeSequence as TS

using Test
using Random

mutable struct Step1{VT} <: TS.AbstractStep{VT,1}
    v1::Float64
    Step1{VT}() where VT = new()
end
function TS.set_params!(op::Step1, params)
    op.v1 = params[1]
    return
end
function TS.compute(op::Step1, grad)
    if !isempty(grad)
        grad[1] = 1
    end
    return op.v1
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
function TS.compute(op::Step2, grad)
    if !isempty(grad)
        grad[1] = op.v2
        grad[2] = op.v1
    end
    return op.v1 * op.v2
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
function TS.compute(op::Step3, grad)
    if !isempty(grad)
        grad[1] = op.v2 * op.v3
        grad[2] = op.v1 * op.v3
        grad[3] = op.v1 * op.v2
    end
    return op.v1 * op.v2 * op.v3
end

function test_scalar_sequence(s)
    NP = TS.nparams(typeof(s))
    @test TS.nparams(s) == NP

    grads = zeros(NP)
    for _ in 1:100
        params = rand(NP) .+ 0.4
        TS.set_params!(s, params)
        @test TS.compute(s, Float64[]) ≈ prod(params)
        grads .= 0
        @test TS.compute(s, grads) ≈ prod(params)
        for i in 1:NP
            @test grads[i] ≈ prod((p for (pidx, p) in enumerate(params) if pidx != i),
                                   init=1.0)
        end
    end
end

@testset "Scalar Time Sequence" begin
    ops = (Step1{Float64}(), Step2{Float64}(), Step3{Float64}(), Step1{Float64}(),
           Step1{Float64}(), Step2{Float64}(), Step3{Float64}(), Step2{Float64}(),
           Step2{Float64}(), Step1{Float64}(), Step3{Float64}(), Step3{Float64}())
    s = TS.Sequence{Float64}(ops)
    @test TS.nparams(s) == 24
    test_scalar_sequence(s)

    s1 = TS.Sequence{Float64}((Step1{Float64}(),))
    @test TS.nparams(s1) == 1
    test_scalar_sequence(s1)

    s2 = TS.Sequence{Float64}((Step2{Float64}(),))
    @test TS.nparams(s2) == 2
    test_scalar_sequence(s2)

    s3 = TS.Sequence{Float64}((Step3{Float64}(),))
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

end
