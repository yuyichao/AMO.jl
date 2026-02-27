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
Base.length(iter::SpinManifoldIter) = _length(iter.dJs)

Base.eltype(::Type{SpinManifoldIter{N}}) where N = NTuple{N,Half{Int}}

_double_spin(S) = S isa Number ? (_double(S),) : _double.(S)

function _dipole_size(dL1, dL2, dS)
    if abs(dL1 - dL2) > 2 || abs(dL1 - dL2) == 1
        throw(ArgumentError("Angular momentum $(half(dL1)) and $(half(dL2)) not coupled via dipole"))
    end
    nS = prod((ds + 1 for ds in dS), init=1)
    return (dL1 + 1) * nS, (dL2 + 1) * nS
end

function _dipole_couple_matrix!(M, dL1, dL2, Ωs, dS)
    fill!(M, 0)
    T = eltype(M)
    RT = real(T)
    nspin = length(dS)

    offset1 = 0
    for (i1, j1s) in enumerate(_double_spin_iter((dL1, dS...)))
        jend1 = j1s[end]
        njend1 = twice(jend1) + 1

        offset2 = 0
        for (i2, j2s) in enumerate(_double_spin_iter((dL2, dS...)))
            jend2 = j2s[end]
            njend2 = twice(jend2) + 1

            forbidden = false
            for (j1, j2) in zip(j1s, j2s)
                if abs(j1 - j2) > 1
                    forbidden = true
                    break
                end
            end

            if forbidden
                offset2 += njend2
                continue
            end

            scale = one(RT)
            for si in 1:nspin
                scale *= couple_reduced_element(RT, j1s[si + 1], j2s[si + 1],
                                                half(dS[si]), j1s[si], j2s[si], 1)
            end

            for mj1 in -jend1:jend1
                idx1 = twice(mj1 + jend1) ÷ 2 + 1 + offset1
                @inbounds for Ωk in -1:1
                    mj2 = mj1 + Ωk
                    if !(-jend2 <= mj2 <= jend2)
                        continue
                    end
                    Ω = Ωs[Ωk + 2]
                    idx2 = twice(mj2 + jend2) ÷ 2 + 1 + offset2
                    M[idx1, idx2] = Ω * (scale * clebschgordan(RT, jend2, mj2, 1, -Ωk,
                                                                jend1, mj1))
                end
            end

            offset2 += njend2
        end
        offset1 += njend1
    end
end

function dipole_couple_matrix!(M, L1, L2, Ωs; S=())
    dL1 = _double(L1)
    dL2 = _double(L2)
    dS = _double_spin(S)

    sz = size(M)
    sz_exp = _dipole_size(dL1, dL2, dS)

    if sz != sz_exp
        throw(DimensionError("Size mismatch: expected $(sz_exp) got $(sz)"))
    end

    _dipole_couple_matrix!(M, dL1, dL2, Ωs, dS)
    return M
end

function dipole_couple_matrix(::Type{T}, L1, L2, Ωs; S=()) where T
    dL1 = _double(L1)
    dL2 = _double(L2)
    dS = _double_spin(S)

    M = Array{T}(undef, _dipole_size(dL1, dL2, dS))
    _dipole_couple_matrix!(M, dL1, dL2, Ωs, dS)
    return M
end

dipole_couple_matrix(L1, L2, Ωs; S=()) = dipole_couple_matrix(ComplexF64, L1, L2, Ωs; S=S)

end
