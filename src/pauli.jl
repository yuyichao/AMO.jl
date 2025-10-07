#

module Pauli

using Base.Order
using Base.Sort

import LinearAlgebra: mul!
using Static

using ..Utils: ThreadObjectPool

public add!, sub!, mul!, div!, icomm, icomm!, Workspace, PauliOperators, OPToken

struct PVector{T} <: AbstractVector{T}
    ptr::Ptr{T}
    len::Int
    PVector(a::AbstractVector{T}) where T = new{T}(pointer(a), length(a))
    PVector(a::AbstractVector{T}, rng::UnitRange) where T =
        new{T}(pointer(a, first(rng)), length(rng))
end

@inline Base.size(a::PVector) = (a.len,)
@inline Base.strides(::PVector) = (1,)
@inline Base.unsafe_convert(::Type{Ptr{T}}, a::PVector) where T = Ptr{T}(a.ptr)
@inline Base.elsize(::Type{PVector{T}}) where T = sizeof(T)
@inline Base.getindex(a::PVector{T}, i) where T =
    Base.pointerref(a.ptr, Int(i), Base.datatype_alignment(T))
@inline Base.getindex(a::PVector, rng::UnitRange) = PVector(a, rng)
@inline Base.setindex!(a::PVector{T}, v, i) where T =
    Base.pointerset(a.ptr, T(v), Int(i), Base.datatype_alignment(T))

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
    const term_bits::Vector{Int32}
    # Lazily populated arrays to identify which term contains the corresponding qubit
    const terms_map_offsets::Vector{Int32}
    const terms_map::Vector{Int32}
    max_len::UInt8
    function PauliOperators{T}(; max_len=3) where T
        return new{T}([Term{T}(0, 0)], Int32[], Int32[], Int32[], max_len)
    end
end

mutable struct Workspace{T}
    const bitvec_cache::Vector{Vector{Int32}}
    bitvec_used::Int
    const termidx_map::Dict{Vector{Int32},Int32}
    const terms::Vector{Tuple{Ptr{Nothing},T}}
    const visited::Vector{Bool}
    const op_cache::Vector{PauliOperators{T}}
    op_used::Int
    function Workspace{T}() where T
        return new{T}(Vector{Int32}[], 0, Dict{Vector{Int32},Int32}(),
                      Tuple{Ptr{Nothing},T}[], Bool[], PauliOperators{T}[], 0)
    end
end

@generated function _workspace_pool(::Type{T}) where T
    pool = ThreadObjectPool(Workspace{T})
    atexit(()->empty!(pool))
    return pool
end

const RST_BITVEC = static(1)
const RST_TERMS = static(2)
const RST_OP = static(4)
const RST_ALL = RST_BITVEC | RST_TERMS | RST_OP

@inline function with_workspace(@specialize(cb), ::Type{T}, workspace, rst_flag) where T
    if workspace !== nothing
        res = @inline cb(workspace)
        reset!(workspace, rst_flag)
        return res
    end
    pool = _workspace_pool(T)
    workspace = get(pool)
    try
        return @inline cb(workspace)
    finally
        reset!(workspace, RST_ALL)
        put!(pool, workspace)
    end
end

@inline function accumulate_term!(workspace::Workspace, bits::Vector{Int32}, v)
    h = workspace.termidx_map
    index, sh = Base.ht_keyindex2_shorthash!(h, bits)
    found = index > 0
    @inbounds if found
        termidx = Int(h.vals[index])
        term = workspace.terms[termidx]
        workspace.terms[termidx] = (term[1], term[2] + v)
    else
        push!(workspace.terms, (pointer_from_objref(bits), v))
        Base._setindex!(h, length(workspace.terms) % Int32, bits, -index, sh)
    end
    return !found
end

@inline function reset!(workspace::Workspace, rst_flag)
    if (rst_flag & RST_BITVEC) != 0
        workspace.bitvec_used = 0
    end
    if (rst_flag & RST_TERMS) != 0
        empty!(workspace.terms)
        empty!(workspace.termidx_map)
    end
    if (rst_flag & RST_OP) != 0
        workspace.op_used = 0
    end
    return
end

@inline function load_ptrarray(array::Vector{T}, i) where T
    ptr = Base.pointerref(Ptr{Ptr{Nothing}}(pointer(array, i)), 1, sizeof(C_NULL))
    return ccall(:jl_value_ptr, Ref{T}, (Ptr{Nothing},), ptr)
end

@noinline function _alloc_intvec(workspace::Workspace)
    res = Int32[]
    push!(workspace.bitvec_cache, res)
    return res
end

@inline function alloc_intvec(workspace::Workspace)
    idx = workspace.bitvec_used + 1
    workspace.bitvec_used = idx
    if idx <= length(workspace.bitvec_cache)
        res = load_ptrarray(workspace.bitvec_cache, idx)
        empty!(res)
        return res
    end
    return _alloc_intvec(workspace)
end

@noinline function _alloc_op(workspace::Workspace{T}, max_len::UInt8) where T
    res = PauliOperators{T}(max_len=max_len)
    push!(workspace.op_cache, res)
    return res
end

@inline function alloc_op(workspace::Workspace, max_len::UInt8)
    idx = workspace.op_used + 1
    workspace.op_used = idx
    if idx <= length(workspace.op_cache)
        res = load_ptrarray(workspace.op_cache, idx)
        empty!(res)
        res.max_len = max_len
        return res
    end
    return _alloc_op(workspace, max_len)
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

@eval primitive type OPToken 32 end
OPToken(v::Int) = reinterpret(OPToken, v % Int32)
Base.Int(v::OPToken) = reinterpret(Int32, v) % Int

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

@inline function get_bits(term_bits::PVector, term::Term)
    nbits = term.bits & 255
    offset = term.bits >> 8
    return term_bits[offset + 1:offset + nbits]
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

function parse_bits!(bits, input)
    _parse_bits!(bits, input)
    last_bitidx = 0
    sort!(bits)
    for bit in bits
        bitidx = bit >> 2
        typ = bit & 3
        if bitidx <= last_bitidx
            throw(ArgumentError("Duplicated bit index"))
        elseif typ == 0
            throw(ArgumentError("Invalid Pauli operator type: $typ"))
        end
        last_bitidx = bitidx
    end
    return bits
end

function _findterm(op::PauliOperators, bits::Vector{Int32})
    GC.@preserve op bits begin
        term_bits = PVector(op.term_bits)
        terms = PVector(op.terms)
        bits = PVector(bits)
        lb = 1
        ub = length(terms)
        while ub >= lb
            mid = ub
            c = cmp(bits, get_bits(term_bits, terms[mid]))
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
end

function findterm(op::PauliOperators{T}, bits; workspace=nothing) where T
    if isempty(bits)
        return OPToken(1)
    end
    return with_workspace(T, workspace, RST_BITVEC) do workspace
        return _findterm(op, parse_bits!(alloc_intvec(workspace), bits))
    end
end

_record_term(::Nothing, bits, idx) = nothing
function _record_term(v, bits, idx)
    v[copy(bits)] = OPToken(idx)
    return
end

@inline ptr_to_intvec(ptr) = ccall(:jl_value_ptr, Ref{Vector{Int32}}, (Ptr{Nothing},), ptr)

@inline function _build_op!(out::PauliOperators{T}, workspace::Workspace, keep_zero=false,
                            recorder=nothing) where T
    empty!(out)
    GC.@preserve workspace begin
        ws_terms = workspace.terms
        if isbitstype(T)
            ws_terms = PVector(ws_terms)
        end
        @inline sort!(ws_terms, by=(@inline term->ptr_to_intvec(term[1])),
                      alg=Sort.QuickSort)
        for (ptr, v) in ws_terms
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
    end
    return out
end

function PauliOperators{T}(terms; max_len=3, workspace=nothing,
                           terms_recorder=nothing) where T
    op = PauliOperators{T}(max_len=max_len)
    return with_workspace(T, workspace, RST_BITVEC | RST_TERMS) do workspace
        for (bits, v) in terms
            bits = parse_bits!(alloc_intvec(workspace), bits)
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
    PauliOperators{promote_type(T1, T2)}(max_len=op.max_len)
_empty(op1::PauliOperators{T1}, op2::PauliOperators{T2}) where {T1,T2} =
    PauliOperators{promote_type(T1, T2)}(max_len=max(op1.max_len, op2.max_len))
_empty(op1::PauliOperators{T1}, op2::PauliOperators{T2}, ::Type{T3}) where {T1,T2,T3} =
    PauliOperators{promote_type(T1, T2, T3)}(max_len=max(op1.max_len, op2.max_len))

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

@inline function Base.empty!(op::PauliOperators{T}) where T
    resize!(op.terms, 1)
    @inbounds op.terms[1] = Term{T}(zero(T), op.terms[1].bits)
    empty!(op.term_bits)
    empty!(op.terms_map)
    empty!(op.terms_map_offsets)
    return op
end

function _populate_terms_map(op::PauliOperators)
    GC.@preserve op begin
        terms = PVector(op.terms)
        term_bits = PVector(op.term_bits)
        resize!(op.terms_map, length(term_bits))
        terms_map = PVector(op.terms_map)

        _offsets = op.terms_map_offsets
        maxbit = 0
        @inbounds for bit in term_bits
            bitidx = bit >> 2
            if bitidx > maxbit
                resize!(_offsets, bitidx + 1)
                for i in maxbit + 1:bitidx - 1
                    _offsets[i + 1] = Int32(0)
                end
                _offsets[bitidx + 1] = Int32(1)
                maxbit = bitidx
            else
                _offsets[bitidx + 1] += Int32(1)
            end
        end
        offsets = PVector(_offsets)
        total_bits::Int32 = 0
        offsets[1] = 0
        for i in 1:maxbit
            cnt = offsets[i + 1]
            offsets[i + 1] = total_bits
            total_bits += cnt
        end
        for (termidx, term) in enumerate(terms)
            for bit in get_bits(term_bits, term)
                bitidx = bit >> 2
                mapidx = offsets[bitidx + 1] + 1
                offsets[bitidx + 1] = mapidx
                terms_map[mapidx] = termidx
            end
        end
    end
end

@inline function _ensure_terms_map(op::PauliOperators)
    # Assume that the map is either empty or fully filled
    if !isempty(op.terms_map)
        return
    end
    _populate_terms_map(op)
end

# Internal function for operator construction only.
@inline function _add_term!(op::PauliOperators{T}, v, bits) where T
    nbits = length(bits)
    term_bits = op.term_bits
    old_bitlen = length(term_bits)
    resize!(term_bits, old_bitlen + nbits)
    @inbounds @simd for i in 1:nbits
        term_bits[old_bitlen + i] = bits[i]
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

function mul!(out::PauliOperators{T}, op::PauliOperators, scale::Number) where T
    max_len = out.max_len
    op_terms = PVector(op.terms)
    nterms = length(op_terms)
    GC.@preserve out op if max_len < op.max_len
        empty!(out)
        @inbounds out.terms[1] = Term{T}(op_terms[1].v * scale, out.terms[1].bits)
        term_bits = PVector(op.term_bits)
        @inbounds for i in 2:length(op_terms)
            term = op_terms[i]
            bits = get_bits(term_bits, term)
            if length(bits) <= max_len
                _add_term!(out, term.v * scale, bits)
            end
        end
    else
        _copy_vec!(out.terms_map_offsets, op.terms_map_offsets)
        _copy_vec!(out.terms_map, op.terms_map)
        _copy_vec!(out.term_bits, op.term_bits)
        resize!(out.terms, nterms)
        @inbounds @simd for i in 1:nterms
            term = op_terms[i]
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
    GC.@preserve A B begin
        termsa = PVector(A.terms)
        termsb = PVector(B.terms)
        term_bitsa = PVector(A.term_bits)
        term_bitsb = PVector(B.term_bits)
        ntermsa = length(termsa)
        ntermsb = length(termsb)
        idxa = 2
        idxb = 2
        @inbounds if idxa <= ntermsa && idxb <= ntermsb
            terma = termsa[idxa]
            termb = termsb[idxb]
            while true
                bitsa = get_bits(term_bitsa, terma)
                bitsb = get_bits(term_bitsb, termb)

                c = cmp(bitsa, bitsb)
                if c < 0
                    @inline cb(bitsa, terma.v, static(false))
                    idxa += 1
                    if idxa > ntermsa
                        break
                    end
                    terma = termsa[idxa]
                elseif c > 0
                    @inline cb(bitsb, static(false), termb.v)
                    idxb += 1
                    if idxb > ntermsb
                        break
                    end
                    termb = termsb[idxb]
                else
                    @inline cb(bitsa, terma.v, termb.v)
                    idxa += 1
                    idxb += 1
                    if idxa > ntermsa || idxb > ntermsb
                        break
                    end
                    terma = termsa[idxa]
                    termb = termsb[idxb]
                end
            end
        end
        @inbounds for i in idxa:ntermsa
            terma = termsa[i]
            @inline cb(get_bits(term_bitsa, terma), terma.v, static(false))
        end
        @inbounds for i in idxb:ntermsb
            termb = termsb[i]
            @inline cb(get_bits(term_bitsb, termb), static(false), termb.v)
        end
    end
end

function add!(out::PauliOperators{T}, A::PauliOperators, ca::Number,
              B::PauliOperators, cb::Number) where T
    empty!(out)
    @inbounds out.terms[1] = Term{T}(A.terms[1].v * ca + B.terms[1].v * cb,
                                     out.terms[1].bits)
    max_len = out.max_len
    foreach_nzterms(A::PauliOperators, B::PauliOperators) do bits, va, vb
        if length(bits) <= max_len
            v = va * ca + vb * cb
            if v != 0
                _add_term!(out, v, bits)
            end
        end
    end
    return out
end
@inline function add!(out::PauliOperators, A::PauliOperators, B::PauliOperators)
    return add!(out, A, static(true), B, static(true))
end
@inline function sub!(out::PauliOperators, A::PauliOperators, B::PauliOperators)
    return add!(out, A, static(true), B, static(-1))
end

function apply_phase(v, phase)
    rv = real(v)
    iv = imag(v)
    if phase == 0
        return complex(v)
    elseif phase == 1
        return complex(-iv, rv)
    elseif phase == 2
        return complex(-rv, -iv)
    else
        return complex(iv, -rv)
    end
end

function mul!(out::PauliOperators{T}, A::PauliOperators,
              B::PauliOperators; workspace=nothing) where T
    with_workspace(T, workspace, RST_BITVEC | RST_TERMS) do workspace
        max_len = out.max_len
        termsa = PVector(A.terms)
        termsb = PVector(B.terms)
        term_bitsa = PVector(A.term_bits)
        term_bitsb = PVector(B.term_bits)
        GC.@preserve A B @inbounds for terma in termsa
            bitsa = get_bits(term_bitsa, terma)
            for termb in termsb
                bits, phase = mul_bits(workspace, bitsa,
                                       get_bits(term_bitsb, termb), max_len)
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
function icomm!(out::PauliOperators{T}, A::PauliOperators,
                B::PauliOperators; workspace=nothing) where T
    if isempty(A.term_bits) || isempty(B.term_bits)
        empty!(out)
        return out
    end
    _ensure_terms_map(B)
    with_workspace(T, workspace, RST_BITVEC | RST_TERMS) do workspace
        max_len = out.max_len
        map_offsetsb = PVector(B.terms_map_offsets)
        maxbitb = length(map_offsetsb) - 1
        resize!(workspace.visited, maxbitb)

        visited = PVector(workspace.visited)
        termsa = PVector(A.terms)
        termsb = PVector(B.terms)
        term_bitsa = PVector(A.term_bits)
        term_bitsb = PVector(B.term_bits)
        terms_mapb = PVector(B.terms_map)

        GC.@preserve workspace A B @inbounds for terma in termsa
            bitsa = get_bits(term_bitsa, terma)
            fill!(visited, false)
            for bita in bitsa
                bitidxa = bita >> 2
                if bitidxa > maxbitb
                    break
                end
                visited[bitidxa] = true
                for mapidx in map_offsetsb[bitidxa] + 1:map_offsetsb[bitidxa + 1]
                    termb = termsb[terms_mapb[mapidx]]
                    bitsb = get_bits(term_bitsb, termb)
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

# log(exp(iA)exp(iB))/i
function ibch!(out::PauliOperators{T}, A::PauliOperators, B::PauliOperators;
               workspace=nothing, max_order=3) where T
    if max_order > 4
        throw(ArgumentError("max order for bch is 4"))
    end
    # 1st order:
    #   A + B
    # 2nd order:
    #   1/2 * i[A, B]
    # 3rd order:
    #   1/12 * (i[i[A, B], B] - i[i[A, B], A])
    # 4th order:
    #   1/24 * i[i[i[A, B], A], B]
    with_workspace(T, workspace, RST_OP) do workspace
        # live: A, B
        # dead: out
        if max_order <= 1
            return add!(out, A, B)
        end
        max_len = max(A.max_len, B.max_len)
        rmax_len = min(out.max_len, max_len)
        s1 = add!(alloc_op(workspace, rmax_len), A, B)
        # live: A, B, s1(rmax_len)
        # dead: out
        AB = icomm!(alloc_op(workspace, max_len), A, B, workspace=workspace)
        # live: A, B, s1(rmax_len), AB(max_len)
        # dead: out
        if max_order == 2
            return add!(out, s1, static(true), AB, T(0.5))
        end
        s2 = add!(alloc_op(workspace, rmax_len), s1, static(true), AB, T(0.5))
        # live: A, B, AB(max_len), s2(rmax_len)
        # dead: out, s1(rmax_len)
        ABB = icomm!(s1, AB, B, workspace=workspace)
        # live: A, B, AB(max_len), s2(rmax_len), ABB(rmax_len)
        # dead: out
        ABA = icomm!(alloc_op(workspace, max_order > 3 ? max_len : rmax_len),
                     AB, A, workspace=workspace)
        # live: A, B, s2(rmax_len), ABB(rmax_len), ABA(max_len*)
        # dead: out, AB(max_len)
        AB.max_len = rmax_len
        ABB_ABA = sub!(AB, ABB, ABA)
        if max_order == 3
            # live: A, B, s2(rmax_len), ABA(rmax_len), ABB_ABA(rmax_len)
            # dead: out, ABB(rmax_len)
            return add!(out, s2, static(true), ABB_ABA, 1 / T(12))
        end
        # live: A, B, s2(rmax_len), ABA(max_len*), ABB_ABA(rmax_len)
        # dead: out, ABB(rmax_len)
        s3 = add!(ABB, s2, static(true), ABB_ABA, 1 / T(12))
        # live: A, B, ABA(max_len), s3(rmax_len)
        # dead: out, s2(rmax_len), ABB_ABA(rmax_len)
        ABAB = icomm!(s2, ABA, B, workspace=workspace)
        # live: A, B, s3(rmax_len), ABAB(rmax_len)
        # dead: out, ABB_ABA(rmax_len), ABA(max_len)
        return add!(out, s3, static(true), ABAB, -1 / T(24))
    end
end

Base.:(+)(A::PauliOperators) = A
Base.:(-)(A::PauliOperators) = static(-1) * A
Base.:(+)(A::PauliOperators, B::PauliOperators) = add!(_empty(A, B), A, B)
Base.:(-)(A::PauliOperators, B::PauliOperators) = sub!(_empty(A, B), A, B)
function Base.muladd(A::PauliOperators, c::Number, B::PauliOperators)
    return add!(_empty(A, B, typeof(c)), A, c, B, static(true))
end
function Base.muladd(c::Number, A::PauliOperators, B::PauliOperators)
    return add!(_empty(A, B, typeof(c)), A, c, B, static(true))
end
Base.:(*)(A::PauliOperators, c::Number) = mul!(_empty(A, typeof(c)), A, c)
Base.:(*)(c::Number, A::PauliOperators) = mul!(_empty(A, typeof(c)), A, c)
Base.:(/)(A::PauliOperators, c::Number) = mul!(_empty(A, typeof(c)), A, 1 / c)
Base.:(\)(c::Number, A::PauliOperators) = mul!(_empty(A, typeof(c)), A, 1 / c)
Base.:(*)(A::PauliOperators, B::PauliOperators) = mul!(_empty(A, B), A, B)
icomm(A::PauliOperators, B::PauliOperators; workspace=nothing) =
    icomm!(_empty(A, B), A, B; workspace=workspace)
ibch(A::PauliOperators, B::PauliOperators; workspace=nothing, max_order=3) =
    ibch!(_empty(A, B), A, B; workspace=workspace, max_order=max_order)
Base.copy!(A::PauliOperators, B::PauliOperators) = mul!(A, B, static(true))
Base.complex(A::PauliOperators) = copy!(_empty(A, Complex{Bool}), A)

function Base.isapprox(A::PauliOperators, B::PauliOperators; kws...)
    @inbounds if !isapprox(A.terms[1].v, B.terms[1].v; kws...)
        return false
    end
    max_len = min(A.max_len, B.max_len)
    res = Ref(true)
    foreach_nzterms(A::PauliOperators, B::PauliOperators) do bits, va, vb
        res[] &= isapprox(va, vb; kws...)
    end
    return res[]
end

end
