#

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
