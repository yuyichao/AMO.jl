#

module Atomic

using RationalRoots
using WignerSymbols

public g_sum, g_s, g_l

function g_sum(J, J1, g1, J2, g2)
    return (g1 * (J * (J + 1) + J1 * (J1 + 1) - J2 * (J2 + 1)) +
        g2 * (J * (J + 1) + J2 * (J2 + 1) - J1 * (J1 + 1))) / (2 * J * (J + 1))
end

const g_s = 2.00231930436092
const g_l = 1.0

_sqrt(::Type{RationalRoot{T}}, v) where T = signedroot(RationalRoot{T}, v)
_sqrt(::Type{T}, v) where T = sqrt(T(v))

function couple_reduced_element(::Type{T}, J1′, J2′, J0, J1, J2, k=1) where T<:Real
    r = _sqrt(T, (2 * J1 + 1) * (2 * J2′ + 1)) * wigner6j(T, J1, J2, k, J2′, J1′, J0)
    return isodd(Int(J0 + J1 + J2′ + k)) ? -r : r
end

couple_reduced_element(J1′, J2′, J0, J1, J2, k=1) =
    couple_reduced_element(RationalRoot{BigInt}, J1′, J2′, J0, J1, J2, k)

end
