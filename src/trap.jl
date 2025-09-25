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
        lpre = (-η² + logabsgamma(T(n₋ + 1))[1] - logabsgamma(T(n₊ + 1))[1]) / 2 + log(η) * Δn
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
    @inline function SidebandIter(Δn, η; phase=true)
        Δn = abs(Δn)
        η = float(η)
        T = typeof(η)
        η² = η^2
        return new{T,phase}(Δn, 1 + Δn - η², Δn - 1 - η²,
                            exp((-η² - logabsgamma(T(Δn + 1))[1]) / 2 + log(η) * Δn))
    end
end

Base.IteratorSize(::Type{<:SidebandIter}) = Base.IsInfinite()
Base.eltype(::Type{SidebandIter{T,Phase}}) where {T,Phase} = Phase ? Complex{T} : T

@inline Base.iterate(iter::SidebandIter{T,Phase}) where {T,Phase} =
    (sideband_phase(iter.s0, iter.Δn, Phase), (1, iter.s0, T(1), iter.l_1))

@inline function Base.iterate(iter::SidebandIter{T,Phase},
                              (n, s′, l_n′, l_n)) where {T,Phase}
    Δn = iter.Δn
    b = Δn - 1
    a = iter.a
    s = s′ * @fastmath sqrt(T(n) / T(n + Δn))
    l1 = muladd(a, l_n, -b * l_n′)
    l2 = muladd(2, l_n, -l_n′)
    return sideband_phase(s * l_n, Δn, Phase), (n + 1, s, l_n, l1 / T(n + 1) + l2)
end

function thermal_population(nbar, n)
    p0 = 1 / (nbar + 1)
    α = nbar * p0
    return p0 * α^n
end

struct ThermalPopulationIter{T<:AbstractFloat}
    p0::T
    α::T
    @inline function ThermalPopulationIter(nbar)
        nbar = float(nbar)
        T = typeof(nbar)
        p0 = 1 / (nbar + 1)
        α = nbar * p0
        return new{T}(p0, α)
    end
end

Base.IteratorSize(::Type{<:ThermalPopulationIter}) = Base.IsInfinite()
Base.eltype(::Type{ThermalPopulationIter{T}}) where T = T

@inline Base.iterate(iter::ThermalPopulationIter, p=iter.p0) = (p, p * iter.α)

@generated function _thermal_sideband(nbars::NTuple{N}, Δns::NTuple{N}, ηs::NTuple{N,T},
                                      t, thresh) where {N,T}
    vs_iter = Symbol[]
    vs_p0 = Symbol[]
    vs_α = Symbol[]
    vs_Δn = Symbol[]
    vs_nmax = Symbol[]

    body = quote
        thresh /= N
        lthresh = log(thresh)
    end
    for I in 1:N
        @gensym v_iter v_p0 v_α v_Δn v_nmax v_nbar v_η
        push!(vs_iter, v_iter)
        push!(vs_p0, v_p0)
        push!(vs_α, v_α)
        push!(vs_Δn, v_Δn)
        push!(vs_nmax, v_nmax)
        push!(body.args, quote
                  $v_nbar = nbars[$I]
                  $v_Δn = Δns[$I]
                  $v_η = ηs[$I]
                  $v_iter = @inline(SidebandIter($v_Δn, $v_η; phase=false))
                  $v_p0 = T(1 / ($v_nbar + 1))
                  $v_α = T($v_nbar * $v_p0)
                  $v_nmax = max(ceil(Int, lthresh / log($v_α)), 1)
                  if $v_Δn < 0
                      $v_p0 *= $v_α^(-$v_Δn)
                      $v_nmax = max(0, $v_nmax + $v_Δn)
                  end
              end)
    end
    function gen_body(v_t, I)
        @gensym s i n t t0 Ω p
        return quote
            $s = zero(T)
            $t0 = $v_t
            $p = $(vs_p0[I])
            for ($i, $Ω) in enumerate($(vs_iter[I]))
                $n = $i - 1
                $t = $t0 * $Ω
                $s = muladd($p, $(I == N ? :(sin($t)^2) : gen_body(t, I + 1)), $s)
                if $n == $(vs_nmax[I])
                    break
                end
                $p *= $(vs_α[I])
            end
            $s
        end
    end
    push!(body.args, gen_body(:(T(t)), 1))
    return body
end

@inline function thermal_sideband(nbars::NTuple{N,Any}, Δns::NTuple{N,Any},
                                  ηs::NTuple{N,Any}, t; thresh=1e-3) where {N}
    return _thermal_sideband(@inline(promote(float.(nbars)...)),
                             @inline(promote(Δns...)),
                             @inline(promote(float.(ηs)...)), t, thresh)
end

@inline thermal_sideband(nbar::Real, Δn::Integer, η::Real, t; kws...) =
    thermal_sideband((nbar,), (Δn,), (η,), t; kws...)

end
