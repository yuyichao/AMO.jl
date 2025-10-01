#

module Pauli

using Base.Order

import LinearAlgebra: mul!
using Static

using ..Utils: ThreadObjectPool

public add!, sub!, mul!, div!, icomm, icomm!, Workspace, PauliOperators, OPToken

mutable struct Workspace{T}
    const bitvec_cache::Vector{Vector{Int}}
    bitvec_used::Int
    const termidx_map::Dict{Vector{Int},Int}
    const terms::Vector{Tuple{Ptr{Nothing},T}}
    const visited::Vector{Bool}
    function Workspace{T}() where {T}
        return new{T}(Vector{Int}[], 0, Dict{Vector{Int},Int}(),
                      Tuple{Ptr{Nothing},T}[], Bool[])
    end
end

@generated function _workspace_pool(::Type{T}) where T
    pool = ThreadObjectPool(Workspace{T})
    atexit(()->empty!(pool))
    return pool
end

@inline function with_workspace(@specialize(cb), ::Type{T}, workspace=nothing) where T
    if workspace !== nothing
        res = @inline cb(workspace)
        reset!(workspace)
        return res
    end
    pool = _workspace_pool(T)
    workspace = get(pool)
    try
        return @inline cb(workspace)
    finally
        reset!(workspace)
        put!(pool, workspace)
    end
end

@inline function accumulate_term!(workspace::Workspace, bits::Vector{Int}, v)
    h = workspace.termidx_map
    index, sh = Base.ht_keyindex2_shorthash!(h, bits)
    found = index > 0
    @inbounds if found
        termidx = h.vals[index]
        term = workspace.terms[termidx]
        workspace.terms[termidx] = (term[1], term[2] + v)
    else
        push!(workspace.terms, (pointer_from_objref(bits), v))
        Base._setindex!(h, length(workspace.terms), bits, -index, sh)
    end
    return !found
end

function reset!(workspace::Workspace)
    workspace.bitvec_used = 0
    empty!(workspace.terms)
    empty!(workspace.termidx_map)
    return
end

function alloc_intvec(workspace::Workspace)
    idx = workspace.bitvec_used + 1
    workspace.bitvec_used = idx
    if idx <= length(workspace.bitvec_cache)
        res = @inbounds workspace.bitvec_cache[idx]
        empty!(res)
        return res
    end
    res = Int[]
    push!(workspace.bitvec_cache, res)
    return res
end

@inline function mul_bits(workspace::Workspace, bits1, bits2, max_len,
                          ::Val{anticommute}=Val(false)) where {anticommute}
    phase = 0
    nbits1 = length(bits1)
    nbits2 = length(bits2)
    idx1 = 1
    idx2 = 1
    res = alloc_intvec(workspace)
    @inbounds if idx1 <= nbits1 && idx2 <= nbits2
        bit1 = bits1[idx1]
        bit2 = bits2[idx2]
        while true
            bitidx1 = bit1 >> 2
            bitidx2 = bit2 >> 2

            if bitidx1 < bitidx2
                push!(res, bit1)
                idx1 += 1
                if idx1 > nbits1
                    break
                end
                bit1 = bits1[idx1]
            elseif bitidx1 > bitidx2
                push!(res, bit2)
                idx2 += 1
                if idx2 > nbits2
                    break
                end
                bit2 = bits2[idx2]
            else
                xz1 = bit1 & 3
                xz2 = bit2 & 3
                xz = xz1 ⊻ xz2
                if xz != 0
                    push!(res, (bitidx1 << 2) | xz)
                    if (xz1 & xz2 & 1) != 0 # XY or YX
                        phase += (xz2 - xz1) >> 1
                    else
                        phase += xz1 - xz2
                    end
                end
                idx1 += 1
                idx2 += 1
                if idx1 > nbits1 || idx2 > nbits2
                    break
                end
                bit1 = bits1[idx1]
                bit2 = bits2[idx2]
            end
        end
    end
    phase = phase & 3
    if (anticommute && (phase & 1) == 0) || (length(res) > max_len)
        workspace.bitvec_used -= 1
        return nothing, phase
    end
    @inbounds for i in idx1:nbits1
        push!(res, bits1[i])
    end
    @inbounds for i in idx2:nbits2
        push!(res, bits2[i])
    end
    if length(res) > max_len
        workspace.bitvec_used -= 1
        return nothing, phase
    end
    return res, phase
end

@eval primitive type OPToken $(sizeof(Int) * 8) end
OPToken(v::Int) = reinterpret(OPToken, v)
Base.Int(v::OPToken) = reinterpret(Int, v)

struct Term{T}
    # coefficient
    v::T
    # an range in the `term_bits` array representing the pauli string for this term
    bits::Int # nbits:8 + offset:24/56
end

Base.:(==)(t1::Term, t2::Term) = t1.v == t2.v && t1.bits == t2.bits
Base.hash(t::Term, h::UInt) = hash(t.v, hash(t.bits, hash(:Term, h)))

mutable struct PauliOperators{T}
    const terms::Vector{Term{T}}
    # Array storing the bit pattern for each term
    # The last two bits of each number represents which pauli matrix (1: X, 2: Y, 3: Z)
    # is present for the term, the upper bits of the number represent the bit index.
    const term_bits::Vector{Int}
    # Lazily populated arrays to identify which term contains the corresponding qubit
    const terms_map::Memory{Vector{Int}}
    const max_len::Int
    function PauliOperators{T}(nbits; max_len=3) where {T}
        terms_map = Memory{Vector{Int}}(undef, nbits)
        @inbounds for i in 1:nbits
            terms_map[i] = Int[]
        end
        return new{T}([Term{T}(0, 0)], Int[], terms_map, max_len)
    end
end

Base.isvalid(op::PauliOperators, idx::OPToken) = checkbounds(Bool, op.terms, Int(idx))

Base.@propagate_inbounds Base.getindex(op::PauliOperators, idx::OPToken) =
    op.terms[Int(idx)].v

Base.@propagate_inbounds function Base.setindex!(op::PauliOperators{T}, v,
                                                 _idx::OPToken) where T
    idx = Int(_idx)
    @boundscheck checkbounds(op.terms, idx)
    term = @inbounds op.terms[idx]
    @inbounds op.terms[idx] = Term{T}(v, term.bits)
    return
end

Base.@propagate_inbounds Base.getindex(op::PauliOperators, bits; workspace=nothing) =
    getindex(op, findterm(op, bits, workspace=workspace))
Base.@propagate_inbounds Base.setindex!(op::PauliOperators, v, bits; workspace=nothing) =
    setindex!(op, v, findterm(op, bits, workspace=workspace))

@inline function get_bits(op::PauliOperators, term::Term)
    nbits = term.bits & 255
    offset = term.bits >> 8
    return @inbounds @view(op.term_bits[offset + 1:offset + nbits])
end

function _parse_bits!(res, bits)
    for b in bits
        if b isa Integer
            push!(res, b)
            continue
        end
        typ, bitidx = b
        if typ == "x" || typ == "X" || typ == :x || typ == :X
            typ = 1
        elseif typ == "y" || typ == "Y" || typ == :y || typ == :Y
            typ = 3
        elseif typ == "z" || typ == "Z" || typ == :z || typ == :Z
            typ = 2
        else
            throw(ArgumentError("Unknown operator type $typ"))
        end
        push!(res, bitidx * 4 + typ)
    end
end

function _add_bit(res, idx, typ, bitstr)
    if typ != 0
        if idx == 0
            throw(ArgumentError("Missing qubit number in Pauli string: $bitstr"))
        end
        push!(res, idx * 4 + typ)
    end
end

function _parse_bits!(res, bitstr::AbstractString)
    bitstr = strip(bitstr)
    if bitstr == "I" || bitstr == "i"
        return
    end
    typ = 0
    idx = 0
    for c in bitstr
        new_typ = 0
        if c == 'x' || c == 'X'
            new_typ = 1
        elseif c == 'y' || c == 'Y'
            new_typ = 3
        elseif c == 'z' || c == 'Z'
            new_typ = 2
        elseif '0' <= c <= '9'
            idx = idx * 10 + (c - '0')
        elseif '₀' <= c <= '₉'
            idx = idx * 10 + (c - '₀')
        else
            throw(ArgumentError("Invalid Pauli string $bitstr"))
        end
        if new_typ == 0
            # Number
            if typ == 0
                throw(ArgumentError("Pauli string must start with x/y/z: $bitstr"))
            end
            continue
        end
        _add_bit(res, idx, typ, bitstr)
        typ = new_typ
        idx = 0
    end
    _add_bit(res, idx, typ, bitstr)
end

function parse_bits!(bits, input, nbits)
    _parse_bits!(bits, input)
    last_bitidx = 0
    sort!(bits)
    for bit in bits
        bitidx = bit >> 2
        typ = bit & 3
        if bitidx > nbits
            throw(ArgumentError("Bit index out of range"))
        elseif bitidx <= last_bitidx
            throw(ArgumentError("Duplicated bit index"))
        elseif typ == 0
            throw(ArgumentError("Invalid Pauli operator type: $typ"))
        end
        last_bitidx = bitidx
    end
    return bits
end

function _findterm(op::PauliOperators, bits::AbstractVector{Int})
    lb = 1
    ub = length(op.terms)
    while ub >= lb
        mid = ub
        c = cmp(bits, get_bits(op, @inbounds(op.terms[mid])))
        if c == 0
            return OPToken(mid)
        elseif c > 0
            lb = mid + 1
        else
            ub = mid - 1
        end
    end
    return OPToken(0)
end

function findterm(op::PauliOperators{T}, bits; workspace=nothing) where T
    if isempty(bits)
        return OPToken(1)
    end
    return with_workspace(T, workspace) do workspace
        return _findterm(op, parse_bits!(alloc_intvec(workspace),
                                         bits, length(op.terms_map)))
    end
end

_record_term(::Nothing, bits, idx) = nothing
function _record_term(v, bits, idx)
    v[copy(bits)] = OPToken(idx)
    return
end

@inline ptr_to_intvec(ptr) = ccall(:jl_value_ptr, Ref{Vector{Int}}, (Ptr{Nothing},), ptr)

function _build_op!(out::PauliOperators{T}, workspace::Workspace, keep_zero=false,
                    recorder=nothing) where {T}
    empty!(out)
    sort!(workspace.terms, by=@inline term->ptr_to_intvec(term[1]))
    for (ptr, v) in workspace.terms
        bits = ptr_to_intvec(ptr)
        nbits = length(bits)
        if nbits == 0
            _record_term(recorder, bits, 1)
            @inbounds out.terms[1] = Term{T}(v, out.terms[1].bits)
            continue
        elseif !keep_zero && v == 0
            _record_term(recorder, bits, 0)
            continue
        end
        _add_term!(out, v, bits)
        _record_term(recorder, bits, length(out.terms))
    end
    reset!(workspace)
    return out
end

function PauliOperators{T}(nbits, terms; max_len=3, workspace=nothing,
                           terms_recorder=nothing) where {T}
    op = PauliOperators{T}(nbits, max_len=max_len)
    return with_workspace(T, workspace) do workspace
        for (bits, v) in terms
            bits = parse_bits!(alloc_intvec(workspace), bits, nbits)
            if length(bits) > max_len
                throw(ArgumentError("Term too long"))
            end
            if !accumulate_term!(workspace, bits, v)
                throw(ArgumentError("Duplicated terms"))
            end
        end
        return _build_op!(op, workspace, true, terms_recorder)
    end
end

_empty(op::PauliOperators{T1}, ::Type{T2}) where {T1,T2} =
    PauliOperators{promote_type(T1, T2)}(length(op.terms_map), max_len=op.max_len)
_empty(op1::PauliOperators{T1}, op2::PauliOperators{T2}) where {T1,T2} =
    PauliOperators{promote_type(T1, T2)}(length(op1.terms_map),
                                         max_len=max(op1.max_len, op2.max_len))
_empty(op1::PauliOperators{T1}, op2::PauliOperators{T2}, ::Type{T3}) where {T1,T2,T3} =
    PauliOperators{promote_type(T1, T2, T3)}(length(op1.terms_map),
                                             max_len=max(op1.max_len, op2.max_len))

Base.:(==)(op1::PauliOperators, op2::PauliOperators) =
    op1.terms == op2.terms && op1.term_bits == op2.term_bits
Base.hash(op::PauliOperators, h::UInt) =
    hash(op.terms, hash(op.term_bits, hash(:PauliOperators, h)))

function show_bits(io::IO, bits)
    for bit in bits
        typ = bit & 3
        bitidx = bit >> 2
        if typ == 1
            print(io, "X")
        elseif typ == 2
            print(io, "Z")
        else
            print(io, "Y")
        end
        print(io, replace(string(bitidx),
                          "0"=>"₀", "1"=>"₁", "2"=>"₂", "3"=>"₃", "4"=>"₄",
                          "5"=>"₅", "6"=>"₆", "7"=>"₇", "8"=>"₈", "9"=>"₉"))
    end
end

function show_sign(io::IO, v, isfirst)
    if v < 0
        if isfirst
            print(io, "-")
        else
            print(io, " - ")
        end
        return -1
    elseif !isfirst
        print(io, " + ")
    end
    return 1
end

function show_coefficient(io::IO, v, isfirst, need_paren=true)
    rv = real(v)
    iv = imag(v)
    if rv == 0 && iv != 0
        print(io, flipsign(iv, show_sign(io, iv, isfirst)), "im")
        return
    end
    s = show_sign(io, rv, isfirst)
    rv = flipsign(rv, s)
    if iv == 0
        print(io, rv)
        return
    end
    if need_paren
        print(io, "(")
        iv = flipsign(iv, s)
    end
    print(io, rv)
    print(io, flipsign(iv, show_sign(io, iv, false)), need_paren ? "im)" : "im")
    return
end

function Base.show(io::IO, op::PauliOperators)
    had_term = false
    for term in op.terms
        bits = get_bits(op, term)
        if length(bits) == 0
            if term.v != 0 || isempty(op.term_bits)
                had_term = true
                show_coefficient(io, term.v, true, false)
            end
            continue
        end
        v = term.v
        show_coefficient(io, term.v, !had_term)
        had_term = true
        print(io, " * ")
        show_bits(io, bits)
    end
end

function Base.empty!(op::PauliOperators{T}) where T
    resize!(op.terms, 1)
    @inbounds op.terms[1] = Term{T}(0, op.terms[1].bits)
    empty!(op.term_bits)
    for bit_term in op.terms_map
        empty!(bit_term)
    end
    return op
end

function _ensure_terms_map(op::PauliOperators{T}) where T
    if isempty(op.term_bits)
        return
    end
    # Assume that the map is either empty or fully filled
    @inbounds if !isempty(op.terms_map[op.term_bits[1] >> 2])
        return
    end
    @inbounds for (termidx, term) in enumerate(op.terms)
        nbits = term.bits & 255
        offset = term.bits >> 8
        @inbounds for i in offset + 1:offset + nbits
            push!(op.terms_map[op.term_bits[i] >> 2], termidx)
        end
    end
end

# Internal function for operator construction only.
@inline function _add_term!(op::PauliOperators{T}, v, bits) where T
    nbits = length(bits)
    old_bitlen = length(op.term_bits)
    resize!(op.term_bits, old_bitlen + nbits)
    @inbounds @simd for i in 1:nbits
        op.term_bits[old_bitlen + i] = bits[i]
    end
    push!(op.terms, Term{T}(v, (old_bitlen << 8) | nbits))
end

@inline function _copy_vec!(tgt, src)
    n = length(src)
    resize!(tgt, n)
    @inbounds @simd for i in 1:n
        tgt[i] = src[i]
    end
    return tgt
end

@inline function check_nbits(tgt::PauliOperators, src::PauliOperators)
    nbits = length(tgt.terms_map)
    if length(src.terms_map) > nbits
        throw(ArgumentError("Destination bit count too small"))
    end
    return nbits
end
@inline function check_nbits(nbits::Integer, src::PauliOperators)
    if length(src.terms_map) > nbits
        throw(ArgumentError("Destination bit count too small"))
    end
    return nbits
end

function mul!(out::PauliOperators{T}, op::PauliOperators, scale::Number) where T
    nbits = check_nbits(out, op)
    max_len = out.max_len
    if max_len < op.max_len
        empty!(out)
        @inbounds out.terms[1] = Term{T}(op.terms[1].v * scale, out.terms[1].bits)
        nterms = length(op.terms)
        @inbounds for i in 2:length(op.terms)
            term = op.terms[i]
            bits = get_bits(op, term)
            if length(bits) <= max_len
                _add_term!(out, term.v * scale, bits)
            end
        end
    else
        @inbounds for i in 1:nbits
            _copy_vec!(out.terms_map[i], op.terms_map[i])
        end
        _copy_vec!(out.term_bits, op.term_bits)
        nterms = length(op.terms)
        resize!(out.terms, nterms)
        @inbounds @simd for i in 1:nterms
            term = op.terms[i]
            out.terms[i] = Term{T}(term.v * scale, term.bits)
        end
    end
    return out
end
@inline function mul!(out::PauliOperators, scale::Number, op::PauliOperators)
    return mul!(out, op, scale)
end
@inline function div!(out::PauliOperators, op::PauliOperators, scale::Number)
    return mul!(out, op, 1 / scale)
end

@inline function foreach_nzterms(cb, A::PauliOperators, B::PauliOperators)
    ntermsa = length(A.terms)
    ntermsb = length(B.terms)
    idxa = 2
    idxb = 2
    @inbounds if idxa <= ntermsa && idxb <= ntermsb
        terma = A.terms[idxa]
        termb = B.terms[idxb]
        while true
            bitsa = get_bits(A, terma)
            bitsb = get_bits(B, termb)

            c = cmp(bitsa, bitsb)
            if c < 0
                @inline cb(bitsa, terma.v, static(false))
                idxa += 1
                if idxa > ntermsa
                    break
                end
                terma = A.terms[idxa]
            elseif c > 0
                @inline cb(bitsb, static(false), termb.v)
                idxb += 1
                if idxb > ntermsb
                    break
                end
                termb = B.terms[idxb]
            else
                @inline cb(bitsa, terma.v, termb.v)
                idxa += 1
                idxb += 1
                if idxa > ntermsa || idxb > ntermsb
                    break
                end
                terma = A.terms[idxa]
                termb = B.terms[idxb]
            end
        end
    end
    @inbounds for i in idxa:ntermsa
        terma = A.terms[i]
        @inline cb(get_bits(A, terma), terma.v, static(false))
    end
    @inbounds for i in idxb:ntermsb
        termb = B.terms[i]
        @inline cb(get_bits(B, termb), static(false), termb.v)
    end
end

function add!(out::PauliOperators{T}, A::PauliOperators, ca::Number,
              B::PauliOperators, cb::Number) where T
    check_nbits(check_nbits(out, A), B)
    empty!(out)
    @inbounds out.terms[1] = Term{T}(A.terms[1].v * ca + B.terms[1].v * cb, out.terms[1].bits)
    max_len = out.max_len
    foreach_nzterms(A::PauliOperators, B::PauliOperators) do bits, va, vb
        if length(bits) <= max_len
            _add_term!(out, va * ca + vb * cb, bits)
        end
    end
    return out
end
@inline function add!(out::PauliOperators, A::PauliOperators, B::PauliOperators)
    return add!(out, A, static(1), B, static(1))
end
@inline function sub!(out::PauliOperators, A::PauliOperators, B::PauliOperators)
    return add!(out, A, static(1), B, static(-1))
end

function apply_phase(v, phase)
    rv = real(v)
    iv = imag(v)
    if phase == 0
        return v
    elseif phase == 1
        return complex(-iv, rv)
    elseif phase == 2
        return complex(-rv, -iv)
    else
        return complex(iv, -rv)
    end
end

@inline function mul!(out::PauliOperators{T}, A::PauliOperators,
                      B::PauliOperators; workspace=nothing) where T
    check_nbits(check_nbits(out, A), B)
    with_workspace(T, workspace) do workspace
        max_len = out.max_len
        @inbounds for terma in A.terms
            bitsa = get_bits(A, terma)
            for termb in B.terms
                bits, phase = mul_bits(workspace, bitsa, get_bits(B, termb), max_len)
                if bits === nothing
                    continue
                end
                if !accumulate_term!(workspace, bits,
                                     apply_phase(terma.v * termb.v, phase))
                    workspace.bitvec_used -= 1
                end
            end
        end
        return _build_op!(out, workspace)
    end
end

@inline function check_visited(visited, curbitidx, bits)
    @inbounds for bit in bits
        bitidx = bit >> 2
        if bitidx >= curbitidx
            break
        end
        if visited[bitidx]
            return true
        end
    end
    return false
end

# out = i[A, B]
@inline function icomm!(out::PauliOperators{T}, A::PauliOperators,
                        B::PauliOperators; workspace=nothing) where T
    nbits = check_nbits(check_nbits(out, A), B)
    _ensure_terms_map(B)
    with_workspace(T, workspace) do workspace
        max_len = out.max_len
        visited = workspace.visited
        resize!(visited, nbits)
        @inbounds for terma in A.terms
            bitsa = get_bits(A, terma)
            fill!(visited, false)
            for bita in bitsa
                bitidxa = bita >> 2
                visited[bitidxa] = true
                for idxb in B.terms_map[bitidxa]
                    termb = B.terms[idxb]
                    bitsb = get_bits(B, termb)
                    if check_visited(visited, bitidxa, bitsb)
                        continue
                    end
                    bits, phase = mul_bits(workspace, bitsa, bitsb, max_len, Val(true))
                    if bits === nothing
                        continue
                    end
                    v = terma.v * termb.v * 2
                    if (phase & 2) == 0
                        v = -v
                    end
                    if !accumulate_term!(workspace, bits, v)
                        workspace.bitvec_used -= 1
                    end
                end
            end
        end
        return _build_op!(out, workspace)
    end
end

Base.:(+)(A::PauliOperators) = A
Base.:(-)(A::PauliOperators) = static(-1) * A
Base.:(+)(A::PauliOperators, B::PauliOperators) = add!(_empty(A, B), A, B)
Base.:(-)(A::PauliOperators, B::PauliOperators) = sub!(_empty(A, B), A, B)
function Base.muladd(A::PauliOperators, c::Number, B::PauliOperators)
    return add!(_empty(A, B, typeof(c)), A, c, B, static(1))
end
function Base.muladd(c::Number, A::PauliOperators, B::PauliOperators)
    return add!(_empty(A, B, typeof(c)), A, c, B, static(1))
end
Base.:(*)(A::PauliOperators, c::Number) = mul!(_empty(A, typeof(c)), A, c)
Base.:(*)(c::Number, A::PauliOperators) = mul!(_empty(A, typeof(c)), A, c)
Base.:(/)(A::PauliOperators, c::Number) = mul!(_empty(A, typeof(c)), A, 1 / c)
Base.:(\)(c::Number, A::PauliOperators) = mul!(_empty(A, typeof(c)), A, 1 / c)
Base.:(*)(A::PauliOperators, B::PauliOperators) = mul!(_empty(A, B), A, B)
icomm(A::PauliOperators, B::PauliOperators; workspace=nothing) =
    icomm!(_empty(A, B), A, B; workspace=workspace)

function Base.isapprox(A::PauliOperators, B::PauliOperators; kws...)
    @inbounds if !isapprox(A.terms[1].v, B.terms[1].v; kws...)
        return false
    end
    max_len = min(A.max_len, B.max_len)
    res = Ref(true)
    foreach_nzterms(A::PauliOperators, B::PauliOperators) do bits, va, vb
        res[] &= isapprox(va, vb, kws...)
    end
    return res[]
end

end
