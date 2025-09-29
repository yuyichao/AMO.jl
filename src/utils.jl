#

module Utils

using Base.Threads
using Base.Iterators

export ThreadObjectPool, eachobj

mutable struct ObjSlot{T}
    @atomic value::Union{T,Nothing}
end

mutable struct ThreadObjectPool{T,CB}
    const lock::ReentrantLock
    const cb::CB
    # The array is used for lock-less fast path and needs to be accessed atomically
    # The array itself will never by mutated (only the ObjSlot{} objects in it will)
    # So access of the array member/size does not need to be atomic.
    # Access of the Atomic variables stored in the array should all be atomic exchanges
    # so that we never have duplicated/missing references to any objects.
    @atomic array::Vector{ObjSlot{T}}
    const extra::Vector{T} # Protected by lock
    function ThreadObjectPool(cb::CB) where CB
        obj = cb()
        T = typeof(obj)
        array = [ObjSlot{T}(obj)]
        return new{T,CB}(ReentrantLock(), cb, array, T[])
    end
end

function Base.get(pool::ThreadObjectPool{T}) where T
    array = @atomic :acquire pool.array
    id = Threads.threadid()
    if id <= length(array)
        obj = @atomicswap(:acquire_release, @inbounds(array[id]).value = nothing)
        if obj !== nothing
            return obj::T
        end
    end
    @lock pool.lock begin
        # Reload, in case someone else changed it
        # No atomicity needed since the thread that may have changed it must
        # have done so with the lock held.
        array = pool.array
        oldlen = length(array)
        if id > oldlen
            array = [i <= oldlen ? @inbounds(array[i]) : ObjSlot{T}(nothing)
                     for i in 1:Threads.maxthreadid()]
            @atomic :release pool.array = array
        end
        if !isempty(pool.extra)
            return pop!(pool.extra)
        end
        return pool.cb()::T
    end
end

function Base.put!(pool::ThreadObjectPool{T}, obj::T) where T
    array = @atomic :acquire pool.array
    id = Threads.threadid()
    if id <= length(array)
        obj = @atomicswap(:acquire_release, @inbounds(array[id]).value = obj)
        if obj === nothing
            return
        end
    end
    @lock pool.lock begin
        push!(pool.extra, obj)
    end
    return
end

function eachobj(pool::ThreadObjectPool{T}) where T
    normal_vals = ((@atomic :unordered s.value) for s in (@atomic :unordered pool.array))
    return Iterators.flatten(((v for v in normal_vals if v !== nothing), pool.extra))
end

end
