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
    @test length(op.terms_map) >= maxbit
    if all(isempty, op.terms_map)
        return
    end
    terms_map = [Int32[] for i in 1:length(op.terms_map)]
    for (termidx, term) in enumerate(op.terms)
        for bit in Pauli.get_bits(op, term)
            push!(terms_map[bit >> 2], termidx)
        end
    end
    for (map1, map2) in zip(op.terms_map, terms_map)
        @test map1 == map2
    end
end

@testset "PauliOperators{$T}" for T in [Float32, Float64, ComplexF32, ComplexF64]
    POT = PauliOperators{T}
    POCT = PauliOperators{complex(T)}

    op0 = POT(10)
    check_consistent(op0)
    @test sprint(show, op0) == "0.0"
    op0[Int32[]] = 1.2
    check_consistent(op0)
    @test sprint(show, op0) == "1.2"
    op0[OPToken(1)] = 2.3
    @test sprint(show, op0) == "2.3"
    op0["I"] = 3.4
    @test sprint(show, op0) == "3.4"
    op0["i"] = 3.5
    @test sprint(show, op0) == "3.5"
    @test_throws BoundsError op0[OPToken(0)]
    @test_throws BoundsError op0[[5, 9]]
    @test_throws ArgumentError op0["XY"]
    @test_throws ArgumentError op0["0"]
    @test_throws ArgumentError op0["X₀"]
    @test_throws ArgumentError op0["z100"]
    @test_throws ArgumentError op0["L"]
    @test_throws ArgumentError op0[((:X, 100),)]
    @test_throws ArgumentError op0[((:XY, 1),)]
    @test_throws ArgumentError op0["X₀Y₀"]
    @test_throws ArgumentError op0[[4, 9]]
    @test_throws ArgumentError op0[[5, 6]]

    @test Pauli.icomm(op0, op0 * 2) == POT(10)

    op0′ = -op0
    @test !(op0 ≈ op0′)
    @test sprint(show, op0′) == "-3.5"
    op0′ = op0 * 1.2
    @test sprint(show, op0′) == "$(3.5 * 1.2)"
    op0′ = 0.1 * op0
    @test sprint(show, op0′) == "$(0.1 * 3.5)"

    @test sprint(show, op0 / 2) == "1.75"
    @test sprint(show, 4 \ op0) == "0.875"

    op1 = POT(10, Dict(" X1Y2 "=>1.2, " Z3Y1X₁₀"=>-0.2))
    check_consistent(op1)
    @test sprint(show, op1) == "1.2 * X₁Y₂ - 0.2 * Y₁Z₃X₁₀"
    @test sprint(show, -op1) == "-1.2 * X₁Y₂ + 0.2 * Y₁Z₃X₁₀"
    @test op1[[(:x, 1), ("Y", 2)]] == T(1.2)
    @test op1[[(:y, 2), ("x", 1)]] == T(1.2)
    @test op1[[(:X, 10), ("z", 3), (:y, 1)]] == T(-0.2)
    @test op1[[(:X, 10), ("z", 3), (:y, 1)]] == T(-0.2)

    recorder = Dict{Vector{Int32},OPToken}()
    op1′ = POT(10, Dict("i"=>-0.3, "X1Y2"=>1.2, "Z3Y1X₁₀ "=>-0.2),
                            terms_recorder=recorder)
    @test recorder == Dict(Int32[]=>OPToken(1),
                           Int32[4 + 1, 8 + 3]=>OPToken(2),
                           Int32[4 + 3, 12 + 2, 40 + 1]=>OPToken(3))
    check_consistent(op1′)
    @test sprint(show, op1′) == "-0.3 + 1.2 * X₁Y₂ - 0.2 * Y₁Z₃X₁₀"

    @test_throws ArgumentError POT(10, Dict("Z3Y1X₁₀Z8 "=>-0.2))
    @test_throws ArgumentError POT(10, Dict("Z3Y1X20 "=>-0.2))
    @test_throws ArgumentError POT(10, Dict("Z3Y1"=>-0.2, "Y₁Z₃"=>0.2))

    @test op1 === +op1
    @test op0 != op1
    @test op0 == op0
    @test op1 == op1
    @test hash(op0) == hash(op0)
    @test hash(op1) == hash(op1)

    op2 = op0 + op1
    check_consistent(op2)
    @test sprint(show, op2) == "3.5 + 1.2 * X₁Y₂ - 0.2 * Y₁Z₃X₁₀"
    op2 = op1 + op0
    check_consistent(op2)
    @test sprint(show, op2) == "3.5 + 1.2 * X₁Y₂ - 0.2 * Y₁Z₃X₁₀"

    @test op1 + op0 == op0 + op1
    @test op1 - op0 == -(op0 - op1)

    @test_throws ArgumentError mul!(POT(3), POT(3), POT(4))
    @test_throws ArgumentError mul!(POT(3), POT(4), POT(3))
    @test_throws ArgumentError mul!(POT(3), POT(4), 0.1)
    @test_throws ArgumentError mul!(POT(3), 0.1, POT(4))

    op3 = mul!(POT(10, max_len=2), op2, -2)
    check_consistent(op3)
    @test sprint(show, op3) == "-7.0 - 2.4 * X₁Y₂"
    op4 = mul!(POT(10, max_len=2), 2, op2)
    check_consistent(op4)
    @test sprint(show, op4) == "7.0 + 2.4 * X₁Y₂"
    op5 = div!(POT(10, max_len=4), op2, 2)
    check_consistent(op5)
    @test sprint(show, op5) == "1.75 + 0.6 * X₁Y₂ - 0.1 * Y₁Z₃X₁₀"

    op6 = POT(10, Dict("i"=>-0.3, "X1Y2"=>1.2, "X2Z3"=>1, "Z3Y1X₁₀ "=>-0.2))
    op7 = POT(10, Dict("I"=>0.3, "X1Y3"=>1.3, "X2Z3"=>2, "Z3Y1X₉ "=>-0.1))
    @test !(op6 ≈ op7)
    op6_7 = op6 + op7
    check_consistent(op6_7)
    op6_7_2 = op6_7 / 2
    check_consistent(op6_7_2)
    @test op6_7 == op7 + op6
    @test op6 - op7 == -(op7 - op6)
    @test sprint(show, op6_7) == "1.2 * X₁Y₂ + 1.3 * X₁Y₃ - 0.1 * Y₁Z₃X₉ - 0.2 * Y₁Z₃X₁₀ + 3.0 * X₂Z₃"
    @test sprint(show, op6_7_2) == "0.6 * X₁Y₂ + 0.65 * X₁Y₃ - 0.05 * Y₁Z₃X₉ - 0.1 * Y₁Z₃X₁₀ + 1.5 * X₂Z₃"

    op8 = POT(10, Dict("X4Y2"=>1.2, "X2Z3"=>1, "Z3Y1X₁₀ "=>-0.2))
    op9 = POT(10, Dict("X4Y3"=>1.3, "X2Z3"=>2, "Z3Y1X₉ "=>-0.1))
    @test !(op8 ≈ op9)
    op8_9 = muladd(op8, 2, op9)
    check_consistent(op8_9)
    op8_9_2 = 2 \ op8_9
    check_consistent(op8_9_2)
    @test op8_9 == muladd(2, op8, op9)
    @test muladd(op8, -1, op9) == -muladd(op9, -1, op8)
    @test sprint(show, op8_9) == "-0.1 * Y₁Z₃X₉ - 0.4 * Y₁Z₃X₁₀ + 4.0 * X₂Z₃ + 2.4 * Y₂X₄ + 1.3 * Y₃X₄"
    @test sprint(show, op8_9_2) == "-0.05 * Y₁Z₃X₉ - 0.2 * Y₁Z₃X₁₀ + 2.0 * X₂Z₃ + 1.2 * Y₂X₄ + 0.65 * Y₃X₄"

    ic6_7 = Pauli.icomm!(POT(10, max_len=5), op6, op7)
    check_consistent(ic6_7)
    @test ic6_7 ≈ POT(10, Dict("X₁X₂X₃"=>2.6, "X₁Z₂Z₃"=>4.8, "Z₁Y₂Z₃X₉"=>0.24), max_len=4)
    @test ic6_7 ≈ -Pauli.icomm!(POT(10, max_len=5), op7, op6)
    @test ic6_7 ≈ im * (mul!(POCT(10, max_len=5), op6, op7) - mul!(POCT(10, max_len=5), op7, op6))

    ic6_7′ = Pauli.icomm(op6, op7)
    check_consistent(ic6_7′)
    @test ic6_7′ ≈ POT(10, Dict("X₁X₂X₃"=>2.6, "X₁Z₂Z₃"=>4.8))
    @test ic6_7′ ≈ -Pauli.icomm(op7, op6)
    @test ic6_7′ ≈ im * (mul!(POCT(10), op6, op7) - mul!(POCT(10), op7, op6))

    @test Pauli.icomm(op6, op6) == POT(10)
    @test Pauli.icomm(op7, op7) == POT(10)

    if T <: Complex
        cop1 = POT(10, Dict("i"=>-0.3im, "X1Y2"=>1.2 + 0.1im, "Z3Y1X₁₀ "=>-0.2 - 1im))
        @test sprint(show, cop1) == "-0.3im + (1.2 + 0.1im) * X₁Y₂ - (0.2 + 1.0im) * Y₁Z₃X₁₀"
        cop2 = POT(10, Dict("i"=>-1 + 0.3im, "X1Y2"=>-1.2 + 0.1im,
                            "Z3Y1X₁₀ "=>0.2 - 1im))
        @test sprint(show, cop2) == "-1.0 + 0.3im - (1.2 - 0.1im) * X₁Y₂ + (0.2 - 1.0im) * Y₁Z₃X₁₀"

        p6_7 = mul!(POT(10, max_len=5), op6, op7)
        @test p6_7 ≈ POT(10, Dict("I"=>1.91, "X₁X₂X₃"=>-1.3im, "X₁Z₂Z₃"=>-2.4im,
                                   "X₁Y₂"=>0.36, "X₁Y₃"=>-0.39, "Z₁Y₂Z₃X₉"=>-0.12im,
                                   "Z₁X₃X₁₀"=>0.26, "Y₁X₂X₉"=>-0.1, "Y₁X₂X₁₀"=>-0.4,
                                   "Y₁Z₃X₉"=>0.03, "Y₁Z₃X₁₀"=>-0.06, "X₂Z₃"=>-0.3,
                                   "Y₂Y₃"=>1.56, "X₉X₁₀"=>0.02), max_len=4)
        @test mul!(POT(10, max_len=5), op7, op6) ≈ p6_7 + im * ic6_7

        p6_7′ = op6 * op7
        @test p6_7′ ≈ POT(10, Dict("I"=>1.91, "X₁X₂X₃"=>-1.3im, "X₁Z₂Z₃"=>-2.4im,
                                   "X₁Y₂"=>0.36, "X₁Y₃"=>-0.39,
                                     "Z₁X₃X₁₀"=>0.26, "Y₁X₂X₉"=>-0.1, "Y₁X₂X₁₀"=>-0.4,
                                   "Y₁Z₃X₉"=>0.03, "Y₁Z₃X₁₀"=>-0.06, "X₂Z₃"=>-0.3,
                                   "Y₂Y₃"=>1.56, "X₉X₁₀"=>0.02), max_len=4)
        @test op7 * op6 ≈ muladd(ic6_7′, im, p6_7′)

        p8_9 = mul!(POT(10, max_len=6), op8, op9)
        @test p8_9 ≈ POT(10, Dict("I"=>2, "Y₁X₂X₉"=>-0.1, "Y₁X₂X₁₀"=>-0.4,
                                   "Y₁Y₂Z₃X₄X₉"=>-0.12, "Y₁X₃X₄X₁₀"=>0.26im,
                                   "X₂X₃X₄"=>-1.3im, "Z₂Z₃X₄"=>-2.4im, "Y₂Y₃"=>1.56,
                                   "X₉X₁₀"=>0.02), max_len=5)

        p8_9′ = op8 * op9
        @test p8_9′ ≈ POT(10, Dict("I"=>2, "Y₁X₂X₉"=>-0.1, "Y₁X₂X₁₀"=>-0.4,
                                     "X₂X₃X₄"=>-1.3im, "Z₂Z₃X₄"=>-2.4im, "Y₂Y₃"=>1.56,
                                     "X₉X₁₀"=>0.02), max_len=4)

        op8² = mul!(POT(10, max_len=6), op8, op8)
        @test op8² ≈ POT(10, Dict("I"=>2.48, "Y₁X₂X₁₀"=>-0.4,
                                    "Y₁Y₂Z₃X₄X₁₀"=>-0.48), max_len=5)

        op8²′ = op8 * op8
        @test op8²′ ≈ POT(10, Dict("I"=>2.48, "Y₁X₂X₁₀"=>-0.4))

        op9² = mul!(POT(10, max_len=6), op9, op9)
        @test op9² ≈ POT(10, Dict("I"=>5.7, "Y₁X₂X₉"=>-0.4), max_len=5)
        @test op9² == op9 * op9
    end
end

end
