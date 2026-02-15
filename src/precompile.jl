#

using Base.Iterators

let
    precompile(Math.genlaguerre, (Int, Float32, Float32))
    precompile(Math.genlaguerre, (Int, Float64, Float64))
    precompile(Math.genlaguerre, (Int, Float64, Int))
    precompile(Math.genlaguerre, (Int, Float32, Int))
    for T in (Bool, Int8, UInt8, Int16, UInt16, Int32, UInt32,
              Int64, UInt64, Int128, UInt128)
        precompile(rand, (typeof(RandSetBits{T}(0.0)),))
        precompile(rand, (typeof(RandDepol{T}(0.0)),))
        precompile(rand, (typeof(Rand2QDepol{T}(0.0)),))
        precompile(rand_setbits, (T, Float64))
    end
    for J in [Int, Float64]
        for J1 in [Int, Float64]
            for J2 in [Int, Float64]
                precompile(g_sum, (J, J1, Float64, J2, Float64))
                precompile(g_sum, (J, J1, Int, J2, Float64))
                precompile(g_sum, (J, J1, Float64, J2, Int))
                precompile(g_sum, (J, J1, Int, J2, Int))
            end
        end
    end
    for kw1 in ((;fm=1.0), (;fm=1), (;ωm=1.0), (;ωm=1))
        for kw2 in ((;kp=1.0), (;kp=1), (;λp=1.0), (;λp=1), (;νp=1.0), (;νp=1))
            Trap.η(1.0; kw1..., kw2...)
            Trap.η(1; kw1..., kw2...)
        end
    end
    for η in Number[0.1, 0]
        Trap.sideband(1, 2, η)
        Trap.sideband(1, 2, η; phase=true)
        Trap.sideband(1, 2, η; phase=false)
        collect(Iterators.take(Trap.SidebandIter(1, η), 3))
        collect(Iterators.take(Trap.SidebandIter(1, η; phase=true), 3))
        collect(Iterators.take(Trap.SidebandIter(1, η; phase=false), 3))
    end
    for nbar in Number[0.1, 1]
        Trap.thermal_population(nbar, 1)
        collect(Iterators.take(Trap.ThermalPopulationIter(nbar), 3))
    end
    for η in Number[0.1, 0]
        for nbar in Number[0.01, 0]
            Trap.thermal_sideband(nbar, 0, η, 0.1)
            Trap.thermal_sideband(nbar, 0, η, 0.1; thresh=1e-1)
            Trap.thermal_sideband((nbar,), (0,), (η,), 0.1)
            Trap.thermal_sideband((nbar,), (0,), (η,), 0.1; thresh=1e-1)
            Trap.thermal_sideband((nbar, nbar), (0, 0), (η, η), 0.1)
            Trap.thermal_sideband((nbar, nbar), (0, 0), (η, η), 0.1; thresh=1e-1)
            Trap.thermal_sideband((nbar, nbar, nbar), (0, 0, 0), (η, η, η), 0.1)
            Trap.thermal_sideband((nbar, nbar, nbar), (0, 0, 0), (η, η, η), 0.1;
                                  thresh=1e-1)
            Trap.thermal_sideband((nbar, nbar, nbar, nbar), (0, 0, 0, 0),
                                  (η, η, η, η), 0.1)
            Trap.thermal_sideband((nbar, nbar, nbar, nbar), (0, 0, 0, 0),
                                  (η, η, η, η), 0.1; thresh=1e-1)
        end
    end

    precompile(Int, (Pauli.OPToken,))
    precompile(Pauli.OPToken, (Int,))
    for T in [Float32, Float64, ComplexF32, ComplexF64]
        ws = Pauli.Workspace{T}()
        PO = Pauli.PauliOperators{T}
        op = PO(["X1"=>0.1])
        PO(["X1"=>0.1]; max_len=1, workspace=ws,
           terms_recorder=Dict{Vector{Int},Pauli.OPToken}())
        PO(max_len=1)
        empty!(PO())
        precompile(complex, (PO,))
        precompile(isvalid, (PO, Pauli.OPToken))
        precompile(getindex, (PO, Pauli.OPToken))
        precompile(setindex!, (PO, T, Pauli.OPToken))
        op["X1"] = 0.2
        op["X1"]
        +op
        -op
        op == op
        op != op
        hash(op)
        show(IOBuffer(), op)
        op * real(T)(0.1)
        real(T)(0.1) * op
        op / real(T)(0.1)
        real(T)(0.1) \ op
        op * complex(T)(0.1)
        complex(T)(0.1) * op
        op / complex(T)(0.1)
        complex(T)(0.1) \ op
        op + op
        op - op
        muladd(op, real(T)(0.1), op)
        muladd(op, complex(T)(0.1), op)
        muladd(real(T)(0.1), op, op)
        muladd(complex(T)(0.1), op, op)
        op * op
        Pauli.mul!(PO(), op, op)
        Pauli.mul!(PO(), op, op, workspace=ws)
        Pauli.icomm(op, op)
        Pauli.icomm(op, op, workspace=ws)
        Pauli.icomm!(PO(), op, op)
        Pauli.icomm!(PO(), op, op, workspace=ws)
        Pauli.ibch(op, op)
        Pauli.ibch(op, op, max_order=1)
        Pauli.ibch(op, op, workspace=ws)
        Pauli.ibch(op, op, workspace=ws, max_order=1)
    end

    for F in Any[0.5, 1]
        for J in Any[0.5, 1]
            for I in Any[0.5, 1]
                for A in Any[0, 0.5]
                    for B in Any[0, 0.5]
                        Atomic.hyperfine(F; I=I, J=I, A=A, B=B)
                        for C in Any[0, 0.5]
                            Atomic.hyperfine(F; I=I, J=I, A=A, B=B, C=C)
                        end
                    end
                end
            end
        end
    end
end
