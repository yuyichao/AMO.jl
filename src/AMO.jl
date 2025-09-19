module AMO

include("math.jl")

include("rand.jl")
using .Rand: RandSetBits, rand_setbits, RandDepol, Rand2QDepol

include("atomic.jl")
using .Atomic: g_sum, g_s, g_l

include("precompile.jl")

end
