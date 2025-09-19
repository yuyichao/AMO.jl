#

module TestRand

using AMO: RandSetBits, rand_setbits, RandDepol, Rand2QDepol

using Test

count1(v::Bool) = Int(v)
count1(v) = count_ones(v)


function _test_rand_setbits_p(cb, ::Type{T}, p) where T
    nbits = T === Bool ? 1 : (sizeof(T) * 8)
    ntotal = 10240000
    niter = ntotal ÷ nbits
    nones = 0
    for _ in 1:niter
        nones += count1(cb())
    end
    nexpect = ntotal * p
    @test nones ≈ nexpect atol=16 * sqrt(nexpect)
end

function test_rand_setbits_p(::Type{T}, p) where T
    _test_rand_setbits_p(()->rand_setbits(T, p), T, p)
    gen = RandSetBits{T}(p)
    _test_rand_setbits_p(()->rand(gen), T, p)
end

function test_rand_setbits(::Type{T}) where T
    @test rand_setbits(T, 0) === zero(T)
    @test rand_setbits(T, 1) === ~zero(T)
    @test rand(RandSetBits{T}(0)) === zero(T)
    @test rand(RandSetBits{T}(1)) === ~zero(T)
    @test_throws ArgumentError rand_setbits(T, -0.1)
    @test_throws ArgumentError rand_setbits(T, 1.1)
    @test_throws ArgumentError RandSetBits{T}(-0.1)
    @test_throws ArgumentError RandSetBits{T}(1.1)

    test_rand_setbits_p(T, 0.1)
    test_rand_setbits_p(T, 0.3)
end

@testset "RandSetBits [$T]" for T in (Bool, Int8, UInt8, Int16, UInt16, Int32, UInt32,
                                      Int64, UInt64, Int128, UInt128)
    test_rand_setbits(T)
end

function test_rand_depol_p(::Type{T}, p) where T
    nbits = T === Bool ? 1 : (sizeof(T) * 8)
    ntotal = 10240000
    niter = ntotal ÷ nbits
    nx = 0
    ny = 0
    nz = 0
    gen = RandDepol{T}(p)
    for _ in 1:niter
        x, z = rand(gen)
        nx += count1(x & ~z)
        ny += count1(x & z)
        nz += count1(~x & z)
    end
    nexpect = ntotal * p / 3
    @test nx ≈ nexpect atol=16 * sqrt(nexpect)
    @test ny ≈ nexpect atol=16 * sqrt(nexpect)
    @test nz ≈ nexpect atol=16 * sqrt(nexpect)
end

function test_rand_depol(::Type{T}) where T
    @test rand(RandDepol{T}(0)) === (zero(T), zero(T))
    @test_throws ArgumentError RandDepol{T}(-0.1)
    @test_throws ArgumentError RandDepol{T}(1.1)

    test_rand_depol_p(T, 0.1)
    test_rand_depol_p(T, 0.3)
    test_rand_depol_p(T, 1.0)
end

@testset "RandDepol [$T]" for T in (Bool, Int8, UInt8, Int16, UInt16, Int32, UInt32,
                                    Int64, UInt64, Int128, UInt128)
    test_rand_depol(T)
end

function test_rand_2qdepol_p(::Type{T}, p) where T
    nbits = T === Bool ? 1 : (sizeof(T) * 8)
    ntotal = 20480000
    niter = ntotal ÷ nbits

    mask0 = zero(T)
    mask1 = ~mask0
    get_mask(v) = v == 0 ? mask1 : mask0

    counts = zeros(Int, 15)
    gen = Rand2QDepol{T}(p)
    for _ in 1:niter
        x1, z1, x2, z2 = rand(gen)
        for i in 1:15
            counts[i] += count1(((get_mask(i & 1) ⊻ x1) & (get_mask(i & 2) ⊻ z1) &
                (get_mask(i & 4) ⊻ x2) & (get_mask(i & 8) ⊻ z2)))
        end
    end
    nexpect = ntotal * p / 15
    for i in 1:15
        @test counts[i] ≈ nexpect atol=16 * sqrt(nexpect)
    end
end

function test_rand_2qdepol(::Type{T}) where T
    @test rand(Rand2QDepol{T}(0)) === (zero(T), zero(T), zero(T), zero(T))
    @test_throws ArgumentError Rand2QDepol{T}(-0.1)
    @test_throws ArgumentError Rand2QDepol{T}(1.1)

    test_rand_2qdepol_p(T, 0.1)
    test_rand_2qdepol_p(T, 0.3)
    test_rand_2qdepol_p(T, 1.0)
end

@testset "Rand2QDepol [$T]" for T in (Bool, Int8, UInt8, Int16, UInt16, Int32, UInt32,
                                      Int64, UInt64, Int128, UInt128)
    test_rand_2qdepol(T)
end

end
