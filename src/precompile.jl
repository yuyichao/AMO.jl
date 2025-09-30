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
end
