#

module Trap

using ..Math: assoc_laguerre
using SpecialFunctions

const c = 299792458
const h = 6.62607015e-34
const ħ = h / 2π

@generated function _check_one_kwarg(kws)
    names = fieldnames(kws)
    found_idx = 0
    for (i, type) in enumerate(fieldtypes(kws))
        if type === Nothing
            continue
        end
        if found_idx == 0
            found_idx = i
            continue
        end
        msg = "Argument conflict: only one of $(join(names, ", ")) may be provided"
        return :(throw(ArgumentError($msg)))
    end
    if found_idx == 0
        msg = "Argument missing: one of $(join(names, ", ")) must be provided"
        return :(throw(ArgumentError($msg)))
    end
    return QuoteNode(names[found_idx])
end

@inline function η(m; fm=nothing, ωm=nothing, kp=nothing, λp=nothing, νp=nothing)
    name = _check_one_kwarg((;fm, ωm))
    if name === :fm
        ωm = float(typeof(fm))(2π * fm)
    else
        @assert name === :ωm
    end
    name = _check_one_kwarg((;λp, kp, νp))
    if name === :λp
        kp = float(typeof(λp))(2π / λp)
    elseif name === :νp
        kp = float(typeof(νp))(2π / c * νp)
    else
        @assert name === :kp
    end
    T = float(promote_type(typeof(m), typeof(ωm), typeof(kp)))
    return sqrt(T(ħ) / 2 / m / ωm) * kp
end

function (_sideband(n1::Integer, n2::Integer, η::T)::T) where {T<:AbstractFloat}
    if n1 < 0 || n2 < 0
        return 0
    elseif η == 0
        return n1 == n2
    end
    # Ref http://journals.aps.org/pra/pdf/10.1103/PhysRevA.20.1521
    # Δn ≡ |n1 - n2|
    # n₋ ≡ min(n1, n2)
    # n₊ ≡ max(n1, n2)
    #   ⟨n1|exp(ikx)|n2⟩
    # = ⟨n1|exp(iη(a + a†))|n2⟩
    # = exp(-η^2 / 2) η^Δn √(γ(n₋ + 1) / γ(n₊ + 1)) L^Δn_n₋(η^2)
    # = exp(-η^2 / 2 + Δn log(η) + lγ(n₋ + 1) / 2 - lγ(n₊ + 1) / 2) L^Δn_n₋(η^2)
    η²::T = η * η
    @fastmath if n1 == n2
        lpre = η² * T(-0.5)
        lag = assoc_laguerre(η², n1, 0)
    else
        n₋ = min(n1, n2)
        n₊ = max(n1, n2)
        Δn = abs(n1 - n2)
        lpre = (-η² + lgamma(T(n₋ + 1)) - lgamma(T(n₊ + 1))) / 2 + log(η) * Δn
        lag = assoc_laguerre(η², n₋, Δn)
    end
    return @fastmath lag * exp(lpre)
end
@inline _sideband(n1::Integer, n2::Integer, η) = _sideband(n1, n2, float(η))

@inline function sideband_phase(s, Δn, phase)
    if !phase
        return s
    end
    Δn = Δn & 3
    if Δn == 0
        return complex(s, 0)
    elseif Δn == 1
        return complex(0, s)
    elseif Δn == 2
        return complex(-s, 0)
    else
        return complex(0, -s)
    end
end

@inline sideband(n1, n2, η; phase=true) =
    sideband_phase(_sideband(n1, n2, η), abs(n1 - n2), phase)

struct SidebandIter{T<:AbstractFloat,Phase}
    Δn::Int
    l_1::T
    a::T
    s0::T
    function SidebandIter(Δn, η; phase=true)
        Δn = abs(Δn)
        η = float(η)
        T = typeof(η)
        η² = η^2
        return new{T,phase}(Δn, 1 + Δn - η², Δn - 1 - η²,
                            exp((-η² - lgamma(T(Δn + 1))) / 2 + log(η) * Δn))
    end
end

Base.IteratorSize(::Type{<:SidebandIter}) = Base.IsInfinite()
Base.eltype(::Type{SidebandIter{T,Phase}}) where {T,Phase} = Phase ? Complex{T} : T

Base.iterate(iter::SidebandIter{T,Phase}) where {T,Phase} =
    (sideband_phase(iter.s0, iter.Δn, Phase), (1, iter.s0, T(1), iter.l_1))

function Base.iterate(iter::SidebandIter{T,Phase}, (n, s′, l_n′, l_n)) where {T,Phase}
    Δn = iter.Δn
    b = Δn - 1
    a = iter.a
    s = s′ * sqrt(n / (n + Δn))
    l1 = muladd(a, l_n, -b * l_n′)
    l2 = muladd(2, l_n, -l_n′)
    return sideband_phase(s * l_n, Δn, Phase), (n + 1, s, l_n, l1 / T(n + 1) + l2)
end

end
