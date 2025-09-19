#

module Atomic

function g_sum(J, J1, g1, J2, g2)
    return (g1 * (J * (J + 1) + J1 * (J1 + 1) - J2 * (J2 + 1)) +
        g2 * (J * (J + 1) + J2 * (J2 + 1) - J1 * (J1 + 1))) / (2 * J * (J + 1))
end

const g_s = 2.00231930436092
const g_l = 1.0

end
