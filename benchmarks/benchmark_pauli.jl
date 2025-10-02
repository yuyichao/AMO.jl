#

import AMO.Pauli as P
using BenchmarkTools

ws = P.Workspace{Float64}()
cws = P.Workspace{ComplexF64}()

cop = P.PauliOperators{ComplexF64}()
cop′ = P.PauliOperators{ComplexF64}()
cop1 = P.PauliOperators{ComplexF64}((("I", 0.2), ("x1z2z3", 1.2), ("x2z1y3", 0.2), ("x2x1y5", 0.1), ("y2x3z9", -0.5), ("x3x2x10", 0.3), ("z3x2y1", 0.1)))
cop2 = P.PauliOperators{ComplexF64}((("I", 2im), ("x1z2y3", 2.2), ("x2x1y3", 0.1), ("z2x1y5", -0.1im), ("x2x3y10", 0.5), ("x3x4x10", -0.3im), ("z9x3z2", -0.1)))

op = P.PauliOperators{Float64}()
op′ = P.PauliOperators{Float64}()
op1 = P.PauliOperators{Float64}((("I", 0.2), ("x1z2z3", 1.2), ("x2z1y3", 0.2), ("x2x1y5", 0.1), ("y2x3z9", -0.5), ("x3x2x10", 0.3), ("z3x2y1", 0.1)))
op2 = P.PauliOperators{Float64}((("I", 2), ("x1z2y3", 2.2), ("x2x1y3", 0.1), ("z2x1y5", -0.1), ("x2x3y10", 0.5), ("x3x4x10", -0.3), ("z9x3z2", -0.1)))

function add_comm!(op, op′, op1, op2; workspace=nothing)
    P.add!(op′, op1, op2)
    P.icomm!(op, op1, op′, workspace=workspace)
end

println("Add")
@btime P.add!($(op), $(op1), $(op2))
@btime P.add!($(cop), $(cop1), $(cop2))

println("Mul (scalar)")
@btime P.mul!($(op), $(op1), $(0.2))
@btime P.mul!($(cop), $(cop1), $(0.3))

println("Mul")
@btime P.mul!($(cop), $(cop1), $(cop2), workspace=$(cws))
@btime P.mul!($(cop), $(cop1), $(cop2))

println("Comm")
@btime P.icomm!($(op), $(op1), $(op2), workspace=$(ws))
@btime P.icomm!($(cop), $(cop1), $(cop2), workspace=$(cws))
@btime P.icomm!($(op), $(op1), $(op2))
@btime P.icomm!($(cop), $(cop1), $(cop2))

println("Add + Comm")
@btime add_comm!($(op), $(op′), $(op1), $(op2), workspace=$(ws))
@btime add_comm!($(cop), $(cop′), $(cop1), $(cop2), workspace=$(cws))
@btime add_comm!($(op), $(op′), $(op1), $(op2))
@btime add_comm!($(cop), $(cop′), $(cop1), $(cop2))
