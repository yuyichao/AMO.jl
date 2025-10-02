#

module TestPauli

using Test

using AMO.Pauli: Pauli, PauliOperators, Workspace, OPToken, add!, sub!, mul!, div!

function check_consistent(op::PauliOperators{T}) where {T}
    ws = Workspace{T}()
    # Constant term must exist
    @test length(op.terms) >= 1
    last_offset = 0
    last_bits = Pauli.get_bits(op, op.terms[1])
    @test Pauli.findterm(op, Int32[]) == OPToken(1)
    @test Pauli.findterm(op, Int32[]; workspace=ws) == OPToken(1)
    @test ws.bitvec_used == 0
    @test isvalid(op, OPToken(1))
    @test !isvalid(op, OPToken(0))
    @test op[Int32[]] == op[OPToken(1)] == op.terms[1].v
    maxbit = 0
    for (i, term) in enumerate(op.terms)
        if i == 1
            # First term must be the constant term
            @test term.bits == 0
            continue
        end
        bits = Pauli.get_bits(op, term)
        # Continuity of bits buffer
        @test (term.bits >> 8) == last_offset
        last_offset += length(bits)
        @test length(bits) <= op.max_len
        # Terms ordering
        @test last_bits < bits
        last_bits = bits
        token = Pauli.findterm(op, bits)
        @test Pauli.findterm(op, bits; workspace=ws) == token
        @test ws.bitvec_used == 0
        @test isvalid(op, token)
        @test op[bits] == op[token] == term.v

        last_bitidx = 0
        for bit in bits
            # Legality of term (must be X/Y/Z)
            @test bit & 3 != 0
            bitidx = bit >> 2
            maxbit = max(maxbit, bitidx)
            # Bit index ordering
            @test bitidx > last_bitidx
            last_bitidx = bitidx
        end
    end
    # Completeness
    @test length(op.term_bits) == last_offset
    if isempty(op.terms_map_offsets)
        @test isempty(op.terms_map)
        return
    end
    maxbit = length(op.terms_map_offsets) - 1
    terms_map = [Int32[] for i in 1:maxbit]
    for (termidx, term) in enumerate(op.terms)
        for bit in Pauli.get_bits(op, term)
            push!(terms_map[bit >> 2], termidx)
        end
    end
    @test length(op.terms_map) == length(op.term_bits)
    @test op.terms_map_offsets[1] == 0
    @test op.terms_map_offsets[end] == length(op.term_bits)
    last_offset = 0
    for i in 1:maxbit
        @test op.terms_map[op.terms_map_offsets[i] + 1:op.terms_map_offsets[i + 1]] == terms_map[i]
    end
end

@testset "PauliOperators{$T}" for T in [Float32, Float64, BigFloat, ComplexF32, ComplexF64, Complex{BigFloat}]
    RT = real(T)
    POT = PauliOperators{T}
    POCT = PauliOperators{complex(T)}
    rtol = max(sqrt(eps()), sqrt(eps(RT)))

    op0 = POT()
    check_consistent(op0)
    @test sprint(show, op0) == "0.0"
    op0[Int32[]] = 1.2
    check_consistent(op0)
    @test sprint(show, op0) == "$(RT(1.2))"
    op0[OPToken(1)] = 2.3
    @test sprint(show, op0) == "$(RT(2.3))"
    op0["I"] = 3.4
    @test sprint(show, op0) == "$(RT(3.4))"
    op0["i"] = 3.5
    @test sprint(show, op0) == "$(RT(3.5))"
    @test_throws BoundsError op0[OPToken(0)]
    @test_throws BoundsError op0[[5, 9]]
    @test_throws ArgumentError op0["XY"]
    @test_throws ArgumentError op0["0"]
    @test_throws ArgumentError op0["X₀"]
    @test_throws BoundsError op0["z100"]
    @test_throws ArgumentError op0["L"]
    @test_throws BoundsError op0[((:X, 100),)]
    @test_throws ArgumentError op0[((:XY, 1),)]
    @test_throws ArgumentError op0["X₀Y₀"]
    @test_throws ArgumentError op0[[4, 9]]
    @test_throws ArgumentError op0[[5, 6]]

    @test Pauli.icomm(op0 * 2, op0) == POT()
    check_consistent(op0)

    op0′ = -op0
    @test !(op0 ≈ op0′)
    @test sprint(show, op0′) == "$(-RT(3.5))"
    op0′ = op0 * 1.2
    @test sprint(show, op0′) == "$(RT(3.5) * 1.2)"
    op0′ = 0.1 * op0
    @test sprint(show, op0′) == "$(0.1 * RT(3.5))"

    @test sprint(show, op0 / 2) == "$(RT(1.75))"
    @test sprint(show, 4 \ op0) == "$(RT(0.875))"

    op1 = POT(Dict(" X1Y2 "=>1.2, " Z3Y1X₁₀"=>-0.2))
    check_consistent(op1)
    @test sprint(show, op1) == "$(RT(1.2)) * X₁Y₂ - $(RT(0.2)) * Y₁Z₃X₁₀"
    @test sprint(show, -op1) == "$(RT(-1.2)) * X₁Y₂ + $(RT(0.2)) * Y₁Z₃X₁₀"
    @test op1[[(:x, 1), ("Y", 2)]] == T(1.2)
    @test op1[[(:y, 2), ("x", 1)]] == T(1.2)
    @test op1[[(:X, 10), ("z", 3), (:y, 1)]] == T(-0.2)
    @test op1[[(:X, 10), ("z", 3), (:y, 1)]] == T(-0.2)

    recorder = Dict{Vector{Int32},OPToken}()
    op1′ = POT(Dict("i"=>-0.3, "X1Y2"=>1.2, "Z3Y1X₁₀ "=>-0.2),
                terms_recorder=recorder)
    @test recorder == Dict(Int32[]=>OPToken(1),
                           Int32[4 + 1, 8 + 3]=>OPToken(2),
                           Int32[4 + 3, 12 + 2, 40 + 1]=>OPToken(3))
    check_consistent(op1′)
    @test sprint(show, op1′) == "$(RT(-0.3)) + $(RT(1.2)) * X₁Y₂ - $(RT(0.2)) * Y₁Z₃X₁₀"

    @test_throws ArgumentError POT(Dict("Z3Y1X₁₀Z8 "=>-0.2))
    @test_throws ArgumentError POT(Dict("Z3Y1"=>-0.2, "Y₁Z₃"=>0.2))

    @test op1 === +op1
    @test op0 != op1
    @test op0 == op0
    @test op1 == op1
    @test hash(op0) == hash(op0)
    @test hash(op1) == hash(op1)

    op2 = op0 + op1
    check_consistent(op2)
    @test sprint(show, op2) == "$(RT(3.5)) + $(RT(1.2)) * X₁Y₂ - $(RT(0.2)) * Y₁Z₃X₁₀"
    op2 = op1 + op0
    check_consistent(op2)
    @test sprint(show, op2) == "$(RT(3.5)) + $(RT(1.2)) * X₁Y₂ - $(RT(0.2)) * Y₁Z₃X₁₀"

    @test op1 + op0 == op0 + op1
    @test op1 - op0 == -(op0 - op1)

    op3 = mul!(POT(max_len=2), op2, -2)
    check_consistent(op3)
    @test sprint(show, op3) == "$(RT(-7.0)) - $(RT(2.4)) * X₁Y₂"
    op4 = mul!(POT(max_len=2), 2, op2)
    check_consistent(op4)
    @test sprint(show, op4) == "$(RT(7.0)) + $(RT(2.4)) * X₁Y₂"
    op5 = div!(POT(max_len=4), op2, 2)
    check_consistent(op5)
    @test sprint(show, op5) == "$(RT(1.75)) + $(RT(0.6)) * X₁Y₂ - $(RT(0.1)) * Y₁Z₃X₁₀"

    op6 = POT(Dict("i"=>-0.3, "X1Y2"=>1.2, "X2Z3"=>1, "Z3Y1X₁₀ "=>-0.2))
    op7 = POT(Dict("I"=>0.3, "X1Y3"=>1.3, "X2Z3"=>2, "Z3Y1X₉ "=>-0.1))
    @test !(op6 ≈ op7)
    op6_7 = op6 + op7
    check_consistent(op6_7)
    op6_7_2 = op6_7 / 2
    check_consistent(op6_7_2)
    @test op6_7 == op7 + op6
    @test op6 - op7 == -(op7 - op6)
    @test sprint(show, op6_7) == "$(RT(1.2)) * X₁Y₂ + $(RT(1.3)) * X₁Y₃ - $(RT(0.1)) * Y₁Z₃X₉ - $(RT(0.2)) * Y₁Z₃X₁₀ + $(RT(3.0)) * X₂Z₃"
    @test sprint(show, op6_7_2) == "$(RT(0.6)) * X₁Y₂ + $(RT(0.65)) * X₁Y₃ - $(RT(0.05)) * Y₁Z₃X₉ - $(RT(0.1)) * Y₁Z₃X₁₀ + $(RT(1.5)) * X₂Z₃"

    op8 = POT(Dict("X4Y2"=>1.2, "X2Z3"=>1, "Z3Y1X₁₀ "=>-0.2))
    op9 = POT(Dict("X4Y3"=>1.3, "X2Z3"=>2, "Z3Y1X₉ "=>-0.1))
    @test !(op8 ≈ op9)
    op8_9 = muladd(op8, 2, op9)
    check_consistent(op8_9)
    op8_9_2 = 2 \ op8_9
    check_consistent(op8_9_2)
    @test op8_9 == muladd(2, op8, op9)
    @test muladd(op8, -1, op9) == -muladd(op9, -1, op8)
    @test sprint(show, op8_9) == "$(RT(-0.1)) * Y₁Z₃X₉ - $(RT(0.4)) * Y₁Z₃X₁₀ + $(RT(4.0)) * X₂Z₃ + $(RT(2.4)) * Y₂X₄ + $(RT(1.3)) * Y₃X₄"
    @test sprint(show, op8_9_2) == "$(RT(-0.05)) * Y₁Z₃X₉ - $(RT(0.2)) * Y₁Z₃X₁₀ + $(RT(2.0)) * X₂Z₃ + $(RT(1.2)) * Y₂X₄ + $(RT(0.65)) * Y₃X₄"

    ic6_7 = Pauli.icomm!(POT(max_len=5), op6, op7)
    check_consistent(op6)
    check_consistent(op7)
    check_consistent(ic6_7)
    @test ic6_7 ≈ POT(Dict("X₁X₂X₃"=>2.6, "X₁Z₂Z₃"=>4.8, "Z₁Y₂Z₃X₉"=>0.24), max_len=4) rtol=rtol
    @test ic6_7 ≈ -Pauli.icomm!(POT(max_len=5), op7, op6)
    @test ic6_7 ≈ im * (mul!(POCT(max_len=5), op6, op7) - mul!(POCT(max_len=5), op7, op6))
    check_consistent(op6)
    check_consistent(op7)

    ic6_7′ = Pauli.icomm(op6, op7)
    check_consistent(ic6_7′)
    @test ic6_7′ ≈ POT(Dict("X₁X₂X₃"=>2.6, "X₁Z₂Z₃"=>4.8))
    @test ic6_7′ ≈ -Pauli.icomm(op7, op6)
    @test ic6_7′ ≈ im * (mul!(POCT(), op6, op7) - mul!(POCT(), op7, op6))

    @test Pauli.icomm(op6, op6) == POT()
    @test Pauli.icomm(op7, op7) == POT()

    if T <: Complex
        cop1 = POT(Dict("i"=>-0.3im, "X1Y2"=>1.2 + 0.1im, "Z3Y1X₁₀ "=>-0.2 - 1im))
        @test sprint(show, cop1) == "$(RT(-0.3))im + ($(RT(1.2)) + $(RT(0.1))im) * X₁Y₂ - ($(RT(0.2)) + $(RT(1.0))im) * Y₁Z₃X₁₀"
        cop2 = POT(Dict("i"=>-1 + 0.3im, "X1Y2"=>-1.2 + 0.1im, "Z3Y1X₁₀ "=>0.2 - 1im))
        @test sprint(show, cop2) == "$(RT(-1.0)) + $(RT(0.3))im - ($(RT(1.2)) - $(RT(0.1))im) * X₁Y₂ + ($(RT(0.2)) - $(RT(1.0))im) * Y₁Z₃X₁₀"

        p6_7 = mul!(POT(max_len=5), op6, op7)
        @test p6_7 ≈ POT(Dict("I"=>1.91, "X₁X₂X₃"=>-1.3im, "X₁Z₂Z₃"=>-2.4im,
                               "X₁Y₂"=>0.36, "X₁Y₃"=>-0.39, "Z₁Y₂Z₃X₉"=>-0.12im,
                               "Z₁X₃X₁₀"=>0.26, "Y₁X₂X₉"=>-0.1, "Y₁X₂X₁₀"=>-0.4,
                               "Y₁Z₃X₉"=>0.03, "Y₁Z₃X₁₀"=>-0.06, "X₂Z₃"=>-0.3,
                               "Y₂Y₃"=>1.56, "X₉X₁₀"=>0.02), max_len=4) rtol=rtol
        @test mul!(POT(max_len=5), op7, op6) ≈ p6_7 + im * ic6_7

        p6_7′ = op6 * op7
        @test p6_7′ ≈ POT(Dict("I"=>1.91, "X₁X₂X₃"=>-1.3im, "X₁Z₂Z₃"=>-2.4im,
                                 "X₁Y₂"=>0.36, "X₁Y₃"=>-0.39,
                                 "Z₁X₃X₁₀"=>0.26, "Y₁X₂X₉"=>-0.1, "Y₁X₂X₁₀"=>-0.4,
                                 "Y₁Z₃X₉"=>0.03, "Y₁Z₃X₁₀"=>-0.06, "X₂Z₃"=>-0.3,
                                 "Y₂Y₃"=>1.56, "X₉X₁₀"=>0.02), max_len=4) rtol=rtol
        @test op7 * op6 ≈ muladd(ic6_7′, im, p6_7′)

        p8_9 = mul!(POT(max_len=6), op8, op9)
        @test p8_9 ≈ POT(Dict("I"=>2, "Y₁X₂X₉"=>-0.1, "Y₁X₂X₁₀"=>-0.4,
                               "Y₁Y₂Z₃X₄X₉"=>-0.12, "Y₁X₃X₄X₁₀"=>0.26im,
                               "X₂X₃X₄"=>-1.3im, "Z₂Z₃X₄"=>-2.4im, "Y₂Y₃"=>1.56,
                               "X₉X₁₀"=>0.02), max_len=5) rtol=rtol

        p8_9′ = op8 * op9
        @test p8_9′ ≈ POT(Dict("I"=>2, "Y₁X₂X₉"=>-0.1, "Y₁X₂X₁₀"=>-0.4,
                                 "X₂X₃X₄"=>-1.3im, "Z₂Z₃X₄"=>-2.4im, "Y₂Y₃"=>1.56,
                                 "X₉X₁₀"=>0.02), max_len=4) rtol=rtol

        op8² = mul!(POT(max_len=6), op8, op8)
        @test op8² ≈ POT(Dict("I"=>2.48, "Y₁X₂X₁₀"=>-0.4,
                                "Y₁Y₂Z₃X₄X₁₀"=>-0.48), max_len=5) rtol=rtol

        op8²′ = op8 * op8
        @test op8²′ ≈ POT(Dict("I"=>2.48, "Y₁X₂X₁₀"=>-0.4)) rtol=rtol

        op9² = mul!(POT(max_len=6), op9, op9)
        @test op9² ≈ POT(Dict("I"=>5.7, "Y₁X₂X₉"=>-0.4), max_len=5) rtol=rtol
        @test op9² == op9 * op9
    end
end

end
