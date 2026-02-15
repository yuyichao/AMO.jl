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

end
