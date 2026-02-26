#

module Atomic

using HalfIntegers
using RationalRoots
using WignerSymbols

public g_sum, g_s, g_l

"""
    g_sum(J, J1, g1, J2, g2)

Compute the g-factor for angular momentum `J` that is created from `J1` with g-factor `g1`
and `J2` with g-factor `g2`.
"""
function g_sum(J, J1, g1, J2, g2)
    return (g1 * (J * (J + 1) + J1 * (J1 + 1) - J2 * (J2 + 1)) +
        g2 * (J * (J + 1) + J2 * (J2 + 1) - J1 * (J1 + 1))) / (2 * J * (J + 1))
end

"""
    g_s

g-factor for electron spin
"""
const g_s = 2.00231930436092


"""
    g_l = 1.0

g-factor for electron orbit
"""
const g_l = 1.0

_sqrt(::Type{RationalRoot{T}}, v) where T = signedroot(RationalRoot{T}, v)
_sqrt(::Type{T}, v) where T = sqrt(T(v))

"""
    couple_reduced_element(::Type{T}=RationalRoot{BigInt}, J1′, J2′, J0, J1, J2, k=1)

Compute the ratio `R` between the reduced matrix elements for `(J1,J2)` and `(J1′,J2′)`.
i.e.,

    ⟨J1′||T^k||J2′⟩ = R ⟨J1||T^k||J2⟩

The definition of the reduced matrix element follows the Wigner-Eckart theorem
with Clebsh-Gordan coefficient:

    ⟨J1,m_J1|T^k_q|J2,m_J2⟩=⟨J1||T^k||J2⟩⟨J1,m_J1|J2,m_J2;k,q⟩

`J1′` and `J2′` are created from another angular momentum `J0` coupling to `J1` and `J2`
respectively. `J0` is assumed to be not coupled to the tensor operator `T`.
"""
function couple_reduced_element(::Type{T}, J1′, J2′, J0, J1, J2, k=1) where T<:Real
    r = _sqrt(T, (2 * J1 + 1) * (2 * J2′ + 1)) * wigner6j(T, J1, J2, k, J2′, J1′, J0)
    return isodd(Int(J0 + J1 + J2′ + k)) ? -r : r
end

couple_reduced_element(J1′, J2′, J0, J1, J2, k=1) =
    couple_reduced_element(RationalRoot{BigInt}, J1′, J2′, J0, J1, J2, k)

"""
    hyperfine(F; I, J, A, B, C=0)

Compute hyperfine state energy for angular momentum `F`
coupled from electron angular momentum `J` and nuclear angular momentum `I`
with hyperfine structure constant `A`, `B` and `C`.
"""
function hyperfine(F; I, J, A, B, C=0)
    T = float(promote_type(typeof(F), typeof(I), typeof(J),
                           typeof(A), typeof(B), typeof(C)))

    @fastmath begin
        F2 = F * (F + 1)
        I2 = I * (I + 1)
        J2 = J * (J + 1)
        K = F2 - (I2 + J2)
        res::T = A * K / T(2)
        if I > 0.5 && J > 0.5
            IJ2 = I2 * J2
            D2 = I * (2 * I - 1) * J * (2 * J - 1)
            Q = 0.375 * K * (K + 1) - 0.5 * IJ2
            res += B * Q / D2
            if I > 1 && J > 1
                O = 5 * K^2 * (K * 0.25 + 1) + K * (I2 + J2 + 3 - 3 * IJ2) - 5 * IJ2
                res += C * O / (D2 * (I - 1) * (J - 1))
            end
        end
    end
    return res
end

_double(j)::Int = twice(j)

struct SpinManifoldIter{N}
    dJs::NTuple{N,Int}
    global _double_spin_iter(dJs::NTuple{N,Int}) where N = new{N}(dJs)
end

SpinManifoldIter(Js...) = _double_spin_iter(_double.(Js))

_start((dJ0,)::Tuple{Int}) = (dJ0,)
_start((dJ0, dJ1, dJs...),) = (dJ0, _start((abs(dJ1 - dJ0), dJs...))...)

_next((dJ0,)::Tuple{Int}, (state,)::Tuple{Int}) = false, state
function _next((dJ0, dJ1, dJs...), (st0, st1, states...))
    valid, next_states = _next((st1, dJs...), (st1, states...))
    if valid
        return true, (st0, next_states...)
    elseif st1 >= dJ0 + dJ1
        return false, (st0, next_states...)
    end
    return true, (dJ0, _start((st1 + 2, dJs...))...)
end

_length(::Tuple{Int}) = 1
_length((dJ0, dJ1)::Tuple{Int,Int}) = min(dJ0, dJ1) + 1
@inline function _length((dJ0, dJ1, dJs...),)
    res = 0
    dJ = abs(dJ0 - dJ1)
    while dJ <= dJ0 + dJ1
        res += _length((dJ, dJs...))
        dJ += 2
    end
    return res
end

function Base.show(io::IO, iter::SpinManifoldIter)
    print(io, "Atomic.SpinManifoldIter(")
    isfirst = true
    for dj in iter.dJs
        if isfirst
            isfirst = false
        else
            print(io, ", ")
        end
        show(io, half(dj))
    end
    print(io, ")")
end

Base.iterate(iter::SpinManifoldIter{0}, state=nothing) = nothing

function Base.iterate(iter::SpinManifoldIter, state=nothing)
    if state === nothing
        state = _start(iter.dJs)
    else
        valid, state = _next(iter.dJs, state)
        if !valid
            return
        end
    end
    return half.(state), state
end

Base.length(iter::SpinManifoldIter{0}) = 0
Base.length(iter::SpinManifoldIter) = _length(sort(iter.dJs))

Base.eltype(::Type{SpinManifoldIter{N}}) where N = NTuple{N,Half{Int}}

end
