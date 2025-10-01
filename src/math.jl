#

module Math

using SpecialFunctions

export assoc_laguerre

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

end
