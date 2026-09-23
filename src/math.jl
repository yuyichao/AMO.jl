#

module Math

using LinearAlgebra: BlasInt, BlasFloat
using LinearAlgebra.BLAS: @blasfunc, libblastrampoline
using LinearAlgebra.LAPACK: chklapackerror
using SpecialFunctions

export assoc_laguerre

public Imaginary, HermEigenWorkspace, heevd!

# LGPLv3 implementation from libstdc++

# function poly_laguerre_large_n(n::Integer, α, x::Tp) where Tp
#     a::Tp = -n
#     b::Tp = α + 1
#     η::Tp = 2b - 4a
#     cos²th = x / η
#     sin²th = 1 - cos²th
#     costh = @fastmath sqrt(cos²th)
#     th = @fastmath acos(costh)
#     pre_h = (Tp(π / 2)^2) * η * η * cos²th * sin²th
#     lg_b = logabsgamma(Tp(n + b))[1]
#     lnfact = logabsgamma(Tp(n + 1))[1]
#     pre_term1 = @fastmath Tp(0.5) * (1 - b) * log(Tp(0.25) * x * η)
#     pre_term2 = @fastmath Tp(0.25) * log(pre_h)
#     lnpre = lg_b - lnfact + Tp(0.5) * x + pre_term1 - pre_term2

#     th2 = 2 * th
#     sin2th = @fastmath 2 * costh * sqrt(sin²th)

#     # From libstdc++
#     ser_term1 = 0 # @fastmath sinpi(a)
#     # This might be off by a minus sign or sth like that.
#     # Evaluating at `10000001, 10, 1.2` gives the wrong result
#     ser_term2 = @fastmath sin(Tp(0.25) * η * (th2 - sin2th) + Tp(π / 4))

#     ser::Tp = ser_term1 + ser_term2
#     return @fastmath exp(lnpre) * ser
# end

# function poly_laguerre_hyperg(n::Integer, α, x::Tp) where Tp
#     b::Tp = Tp(α) + 1
#     mx = -x
#     tc_sgn::Tp = x < 0 ? 1 : ((n % 2 == 1) ? -1 : 1)
#     # Get |x|^n/n!
#     tc::Tp = 1
#     ax = abs(x)
#     for k in 1:n
#         tc *= ax / k
#     end
#     term::Tp = tc * tc_sgn
#     _sum::Tp = term
#     for k in (n - 1):-1:0
#         term *= ((b + Tp(k)) / Tp(n - k)) * Tp(k + 1) / mx
#         _sum += term
#     end
#     return _sum
# end

function poly_laguerre_recursion(n::Integer, α, x::Tp) where Tp
    # Compute l_0.
    l_0::Tp = 1

    # Compute l_1^alpha.
    l_1::Tp = -x + 1 + α

    # Compute l_n^alpha by recursion on n.
    l_n′::Tp = l_0
    l_n::Tp = l_1
    b::Tp = α - 1
    a::Tp = b - x
    for nn in 2:n
        l1 = muladd(a, l_n, -b * l_n′)
        l2 = muladd(2, l_n, -l_n′)
        l_n, l_n′ = l1 / Tp(nn) + l2, l_n
    end
    return l_n
end

function (genlaguerre(n::Integer, α, x::Tp)::Tp) where Tp<:AbstractFloat
    if n < 0
        return 0
    elseif n == 0
        return 1
    elseif isnan(α) || isnan(x)
        # Return NaN on NaN input.
        return NaN
    elseif n == 1
        return Tp(1) + Tp(α) - x
    elseif x == 0
        prod::Tp = α + 1
        for k in 2:n
            prod *= Tp(α + k) / Tp(k)
        end
        return prod
    # elseif n > 10000000 && α > -1 && abs(x) < 2 * (α + 1) + 4n
    #     return poly_laguerre_large_n(n, α, x)
    # elseif α >= 0 || (x > 0 && α < -(n + 1))
    else
        return poly_laguerre_recursion(n, α, x)
    # else
    #     return poly_laguerre_hyperg(n, α, x)
    end
end
assoc_laguerre(x, n::Integer, α=0.0) = genlaguerre(n, α, float(x))

"""
    Imaginary(v)

A purely imaginary number `v * im` with real `v`. Multiplication with a real number,
a complex number or another `Imaginary` only performs the necessary real
operations, and `exp` of an `Imaginary` is computed with `cis`.
This lets the compiler exploit the vanishing real part, e.g. for the `-im * t`
factor in a time evolution operator.
"""
struct Imaginary{T<:Real} <: Number
    v::T
end
@inline Base.real(a::Imaginary) = zero(a.v)
@inline Base.imag(a::Imaginary) = a.v
@inline Base.iszero(a::Imaginary) = iszero(a.v)
@inline Base.zero(a::Imaginary) = Imaginary(zero(a.v))
@inline Base.zero(::Type{Imaginary{T}}) where T = Imaginary(zero(T))
@inline Base.:-(a::Imaginary) = Imaginary(-a.v)
@inline Base.:*(a::Imaginary, b::Real) = Imaginary(a.v * b)
@inline Base.:*(b::Real, a::Imaginary) = Imaginary(b * a.v)
# Avoid ambiguity with the `Bool * Number` methods in Base
@inline Base.:*(a::Imaginary, b::Bool) = Imaginary(a.v * b)
@inline Base.:*(b::Bool, a::Imaginary) = Imaginary(b * a.v)
@inline Base.:*(a::Imaginary, b::Imaginary) = -(a.v * b.v)
@inline Base.:*(a::Imaginary, z::Complex) = Complex(-(a.v * imag(z)), a.v * real(z))
@inline Base.:*(z::Complex, a::Imaginary) = Complex(-(imag(z) * a.v), real(z) * a.v)
@inline Base.:/(a::Imaginary, b::Real) = Imaginary(a.v / b)
@inline Base.exp(a::Imaginary) = cis(a.v)
Base.Complex{T}(a::Imaginary) where T<:Real = Complex{T}(zero(T), a.v)
Base.Complex(a::Imaginary) = Complex(zero(a.v), a.v)
Base.promote_rule(::Type{Imaginary{T}}, ::Type{Imaginary{S}}) where {T,S} =
    Imaginary{promote_type(T, S)}
Base.promote_rule(::Type{Imaginary{T}}, ::Type{S}) where {T,S<:Real} =
    Complex{promote_type(T, S)}
Base.promote_rule(::Type{Imaginary{T}}, ::Type{Complex{S}}) where {T,S} =
    Complex{promote_type(T, S)}
Base.show(io::IO, a::Imaginary) = print(io, "Imaginary(", a.v, ")")


"""
    HermEigenWorkspace{T}(n)

Preallocated workspace for the LAPACK Hermitian eigensolver `zheevd`/`cheevd`
(divide and conquer) for `n × n` matrices with element type `T`
(`ComplexF64` or `ComplexF32`), so that [`heevd!`](@ref) does not allocate.
"""
struct HermEigenWorkspace{T<:BlasFloat,RT<:Real}
    V::Matrix{T}      # input (upper triangle), overwritten with the eigenvectors
    λ::Vector{RT}
    work::Vector{T}
    rwork::Vector{RT}
    iwork::Vector{BlasInt}
end

for (fname, elty, relty) in ((:zheevd_, :ComplexF64, :Float64), (:cheevd_, :ComplexF32, :Float32))
    @eval begin
        function _heevd_ccall!(V::Matrix{$elty}, λ::Vector{$relty}, work::Vector{$elty},
                               rwork::Vector{$relty}, iwork::Vector{BlasInt},
                               lwork::Integer, lrwork::Integer, liwork::Integer)
            n = size(V, 1)
            info = Ref{BlasInt}()
            ccall((@blasfunc($fname), libblastrampoline), Cvoid,
                  (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ref{BlasInt},
                   Ptr{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Clong, Clong),
                  'V', 'U', n, V, max(1, n), λ, work, lwork, rwork, lrwork, iwork, liwork,
                  info, 1, 1)
            chklapackerror(info[])
            return
        end
        function HermEigenWorkspace{$elty}(n::Integer)
            V = Matrix{$elty}(undef, n, n)
            λ = Vector{$relty}(undef, n)
            # Workspace query
            work = Vector{$elty}(undef, 1)
            rwork = Vector{$relty}(undef, 1)
            iwork = Vector{BlasInt}(undef, 1)
            _heevd_ccall!(V, λ, work, rwork, iwork, -1, -1, -1)
            resize!(work, max(BlasInt(real(work[1])), 1))
            resize!(rwork, max(BlasInt(rwork[1]), 1))
            resize!(iwork, max(iwork[1], 1))
            return HermEigenWorkspace{$elty,$relty}(V, λ, work, rwork, iwork)
        end
    end
end

"""
    heevd!(ws::HermEigenWorkspace, A::AbstractMatrix)

Eigendecomposition of the Hermitian matrix `A` (only the upper triangle is accessed)
using the workspace `ws`. Returns the eigenvalues (ascending) and the matrix of
eigenvectors, both aliasing the workspace and therefore overwritten by the next call.
"""
function heevd!(ws::HermEigenWorkspace, A::AbstractMatrix)
    V = ws.V
    copyto!(V, A)
    _heevd_ccall!(V, ws.λ, ws.work, ws.rwork, ws.iwork, length(ws.work), length(ws.rwork),
                  length(ws.iwork))
    return ws.λ, V
end

end
