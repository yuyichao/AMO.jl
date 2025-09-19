#

module Rand

using Random

_nbits(::Type{Bool}) = 1
_nbits(::Type{T}) where T = sizeof(T) * 8

# Copied from randsubseq! from Base
@inline function __rand_setbits(r::AbstractRNG, ::Type{T}, L) where T
    n = sizeof(T) * 8
    S = zero(T)
    # Skip through A, in order, from each element i to the next element i+s
    # included in S. The probability that the next included element is
    # s==k (k > 0) is (1-p)^(k-1) * p, and hence the probability (CDF) that
    # s is in {1,...,k} is 1-(1-p)^k = F(k).   Thus, we can draw the skip s
    # from this probability distribution via the discrete inverse-transform
    # method: s = ceil(F^{-1}(u)) where u = rand(), which is simply
    # s = ceil(log(rand()) / log1p(-p)).
    # -log(rand()) is an exponential variate, so can use randexp().
    i = 0
    while true
        s = randexp(r) * L
        s >= n - i && return S # compare before ceil to avoid overflow
        i += unsafe_trunc(Int, ceil(s))
        S |= one(T) << (i - 1)
    end
    # [This algorithm is similar in spirit to, but much simpler than,
    #  the one by Vitter for a related problem in "Faster methods for
    #  random sampling," Comm. ACM Magazine 7, 703-718 (1984).]
end

@inline function _rand_setbits(r::AbstractRNG, ::Type{T}, p::Real) where T
    L = -1 / log1p(-p) # L > 0
    return __rand_setbits(r, T, L)
end

mutable struct RandSetBits{T<:Union{Bool,Base.BitInteger},R<:AbstractRNG,P<:Real}
    state::UInt128
    ele_left::Int
    # if p_L >= 0, it is the original probability
    # if p_L < 0, it's 1 / log1p(-p) used for computing the amount to jump ahead.
    const p_L::P
    const rng::R
    function RandSetBits{T}(r::R, p::P) where {T<:Union{Bool,Base.BitInteger},R<:AbstractRNG,P<:Real}
        0 <= p <= 1 || throw(ArgumentError("probability $p not in [0,1]"))
        if 0 < p <= 0.17 # empirical threshold for trivial O(n) algorithm to be better
            p = 1 / log1p(-p) # L > 0
        end
        return new{T,R,P}(0, 0, p, r)
    end
    RandSetBits{T}(p::Real) where T<:Union{Bool,Base.BitInteger} =
        RandSetBits{T}(Random.default_rng(), p)
end

function _new_state(sb::RandSetBits)
    Te = UInt128
    n = sizeof(Te) * 8
    S = zero(Te)
    r = sb.rng
    p = sb.p_L
    @inline if p >= 0
        if p == 0
            return S
        elseif p == 1
            return ~S
        end
        for i = 1:n
            S |= Te(rand(r) < p) << (i - 1)
        end
    else
        S = __rand_setbits(r, Te, -p)
    end
    return S
end

@inline function Random.rand(sb::RandSetBits{T}) where T
    Te = UInt128
    @inline if sizeof(T) == sizeof(Te)
        return _new_state(sb) % T
    end
    n = sizeof(Te) * 8
    state = sb.state
    if sb.ele_left == 0
        sb.ele_left = n ÷ _nbits(T)
        state = _new_state(sb)
    end
    sb.ele_left -= 1
    res = state % T
    sb.state = state >> _nbits(T)
    return res
end

function rand_setbits(r::AbstractRNG, ::Type{T}, p::Real) where {T <: Union{Bool,Base.BitInteger}}
    0 <= p <= 1 || throw(ArgumentError("probability $p not in [0,1]"))
    S = zero(T)
    p == 1 && return ~S
    p == 0 && return S
    @inline if T === Bool
        return rand(r) < p
    end
    n = sizeof(T) * 8
    @inline if p > 0.17 # empirical threshold for trivial O(n) algorithm to be better
        for i = 1:n
            S |= T(rand(r) < p) << (i - 1)
        end
        return S
    else
        return _rand_setbits(r, T, p)
    end
end
@inline rand_setbits(::Type{T}, p::Real) where T =
    rand_setbits(Random.default_rng(), T, p)


mutable struct RandDepol{T<:Union{Bool,Base.BitInteger},R<:AbstractRNG,P<:Real}
    state_x::UInt128
    state_z::UInt128
    ele_left::Int
    err_sel_left::Int
    err_sel_state::UInt64
    # if p_L >= 0, it is the original probability
    # if p_L < 0, it's 1 / log1p(-p) used for computing the amount to jump ahead.
    const p_L::P
    const rng::R

    function RandDepol{T}(r::R, p::P) where {T<:Union{Bool,Base.BitInteger},R<:AbstractRNG,P<:Real}
        0 <= p <= 1 || throw(ArgumentError("probability $p not in [0,1]"))
        if 0 < p <= 0.17 # empirical threshold for trivial O(n) algorithm to be better
            p = 1 / log1p(-p) # L > 0
        end
        return new{T,R,P}(0, 0, 0, 0, 0, p, r)
    end
    RandDepol{T}(p::Real) where T<:Union{Bool,Base.BitInteger} =
        RandDepol{T}(Random.default_rng(), p)
end

@inline function _rand_err_sel(rd::RandDepol)
    @inline if rd.err_sel_left == 0
        rd.err_sel_state = rand(rd.rng, zero(UInt64):UInt64(UInt64(3)^36 - 1))
        rd.err_sel_left = 36
    end
    rd.err_sel_state, res = divrem(rd.err_sel_state, 3)
    rd.err_sel_left -= 1
    return res % Int
end

function _new_state(rd::RandDepol)
    Te = UInt128
    n = sizeof(Te) * 8
    S1 = zero(Te)
    S2 = zero(Te)
    r = rd.rng
    p = rd.p_L
    @inline if p >= 0
        if p == 0
            return S1, S2
        end
        for i = 1:n
            v = rand(r)
            xerr = v <= p * 2 / 3
            zerr = p / 3 < v <= p
            S1 |= Te(xerr) << (i - 1)
            S2 |= Te(zerr) << (i - 1)
        end
    else
        i = 0
        while true
            s = randexp(r) * -p
            s >= n - i && return S1, S2 # compare before ceil to avoid overflow
            i += unsafe_trunc(Int, ceil(s))
            err_type = _rand_err_sel(rd)
            S1 |= Te(err_type <= 1) << (i - 1)
            S2 |= Te(err_type >= 1) << (i - 1)
        end
    end
    return S1, S2
end

@inline function Random.rand(rd::RandDepol{T}) where T
    Te = UInt128
    @inline if sizeof(T) == sizeof(Te)
        x, z = _new_state(rd)
        return x % T, z % T
    end
    n = sizeof(Te) * 8
    state_x = rd.state_x
    state_z = rd.state_z
    if rd.ele_left == 0
        rd.ele_left = n ÷ _nbits(T)
        state_x, state_z = _new_state(rd)
    end
    rd.ele_left -= 1
    res_x = state_x % T
    res_z = state_z % T
    rd.state_x = state_x >> _nbits(T)
    rd.state_z = state_z >> _nbits(T)
    return res_x, res_z
end


mutable struct Rand2QDepol{T<:Union{Bool,Base.BitInteger},R<:AbstractRNG,P<:Real}
    state_x1::UInt128
    state_z1::UInt128
    state_x2::UInt128
    state_z2::UInt128
    ele_left::Int
    err_sel_left::Int
    err_sel_state::UInt64
    # if p_L >= 0, it is the original probability
    # if p_L < 0, it's 1 / log1p(-p) used for computing the amount to jump ahead.
    const p_L::P
    const rng::R

    function Rand2QDepol{T}(r::R, p::P) where {T<:Union{Bool,Base.BitInteger},R<:AbstractRNG,P<:Real}
        0 <= p <= 1 || throw(ArgumentError("probability $p not in [0,1]"))
        if 0 < p <= 0.17 # empirical threshold for trivial O(n) algorithm to be better
            p = 1 / log1p(-p) # L > 0
        end
        return new{T,R,P}(0, 0, 0, 0, 0, 0, 0, p, r)
    end
    Rand2QDepol{T}(p::Real) where T<:Union{Bool,Base.BitInteger} =
        Rand2QDepol{T}(Random.default_rng(), p)
end

@inline function _rand_err_sel(rd::Rand2QDepol)
    @inline if rd.err_sel_left == 0
        rd.err_sel_state = rand(rd.rng, zero(UInt64):UInt64(UInt64(15)^14 - 1))
        rd.err_sel_left = 14
    end
    rd.err_sel_state, res = divrem(rd.err_sel_state, 15)
    rd.err_sel_left -= 1
    return res % Int
end

function _new_state(rd::Rand2QDepol)
    Te = UInt128
    n = sizeof(Te) * 8
    S1 = zero(Te)
    S2 = zero(Te)
    S3 = zero(Te)
    S4 = zero(Te)
    r = rd.rng
    p = rd.p_L
    @inline if p >= 0
        if p == 0
            return S1, S2, S3, S4
        end
        for i = 1:n
            v = rand(r)
            if v >= p
                continue
            end
            v = unsafe_trunc(Int, floor(v * (15 / p))) + 1
            S1 |= Te(v & 1) << (i - 1)
            S2 |= Te((v >> 1) & 1) << (i - 1)
            S3 |= Te((v >> 2) & 1) << (i - 1)
            S4 |= Te(v >> 3) << (i - 1)
        end
    else
        i = 0
        while true
            s = randexp(r) * -p
            s >= n - i && return S1, S2, S3, S4 # compare before ceil to avoid overflow
            i += unsafe_trunc(Int, ceil(s))
            v = _rand_err_sel(rd) + 1
            S1 |= Te(v & 1) << (i - 1)
            S2 |= Te((v >> 1) & 1) << (i - 1)
            S3 |= Te((v >> 2) & 1) << (i - 1)
            S4 |= Te(v >> 3) << (i - 1)
        end
    end
    return S1, S2, S3, S4
end

@inline function Random.rand(rd::Rand2QDepol{T}) where T
    Te = UInt128
    @inline if sizeof(T) == sizeof(Te)
        x1, z1, x2, z2 = _new_state(rd)
        return x1 % T, z1 % T, x2 % T, z2 % T
    end
    n = sizeof(Te) * 8
    state_x1 = rd.state_x1
    state_z1 = rd.state_z1
    state_x2 = rd.state_x2
    state_z2 = rd.state_z2
    if rd.ele_left == 0
        rd.ele_left = n ÷ _nbits(T)
        state_x1, state_z1, state_x2, state_z2 = _new_state(rd)
    end
    rd.ele_left -= 1
    res_x1 = state_x1 % T
    res_z1 = state_z1 % T
    res_x2 = state_x2 % T
    res_z2 = state_z2 % T
    rd.state_x1 = state_x1 >> _nbits(T)
    rd.state_z1 = state_z1 >> _nbits(T)
    rd.state_x2 = state_x2 >> _nbits(T)
    rd.state_z2 = state_z2 >> _nbits(T)
    return res_x1, res_z1, res_x2, res_z2
end

end
