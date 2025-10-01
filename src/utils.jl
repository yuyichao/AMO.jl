#

module Utils

using Base.Threads
using Base.Iterators

export ThreadObjectPool, ObjectPool, eachobj

mutable struct ObjSlot{T}
    @atomic value::Union{T,Nothing}
end

_typeof(v) = typeof(v)
_typeof(::Type{T}) where T = Type{T}

mutable struct ThreadObjectPool{T,CB}
    const lock::ReentrantLock
    const cb::CB
    # The array is used for lock-less fast path and needs to be accessed atomically
    # The array itself will never by mutated (only the ObjSlot{} objects in it will)
    # So access of the array member/size does not need to be atomic.
    # Access of the Atomic variables stored in the array should all be atomic exchanges
    # so that we never have duplicated/missing references to any objects.
    @atomic array::Memory{ObjSlot{T}}
    const extra::Vector{T} # Protected by lock
    function ThreadObjectPool(cb)
        obj = cb()
        T = typeof(obj)
        array = Memory{ObjSlot{T}}(undef, 1)
        @inbounds array[1] = ObjSlot{T}(obj)
        return new{T,_typeof(cb)}(ReentrantLock(), cb, array, T[])
    end
end

@inline function _get_slot(array::Memory{ObjSlot{T}}, i) where T
    ptr = Base.pointerref(Ptr{Ptr{Nothing}}(pointer(array, i)), 1, sizeof(C_NULL))
    return ccall(:jl_value_ptr, Ref{ObjSlot{T}}, (Ptr{Nothing},), ptr)
end

function _get_slow(pool::ThreadObjectPool{T}) where T
    @lock pool.lock begin
        if !isempty(pool.extra)
            return pop!(pool.extra)
        end
        return pool.cb()::T
    end
end

function _put_slow(pool::ThreadObjectPool{T}, obj::T, id) where T
    @lock pool.lock begin
        # Reload, in case someone else changed it
        # No atomicity needed since the thread that may have changed it must
        # have done so with the lock held.
        array = @atomic :unordered pool.array
        oldlen = length(array)
        if id > oldlen
            nt = Threads.maxthreadid()
            new_array = Memory{ObjSlot{T}}(undef, nt)
            @inbounds for i in 1:nt
                if i <= oldlen
                    new_array[i] = _get_slot(array, i)
                elseif i == id
                    new_array[i] = ObjSlot{T}(obj)
                else
                    new_array[i] = ObjSlot{T}(nothing)
                end
            end
            @atomic :release pool.array = new_array
        else
            push!(pool.extra, obj)
        end
    end
    return
end

@inline function Base.get(pool::ThreadObjectPool{T}) where T
    array = @atomic :acquire pool.array
    id = Threads.threadid()
    if id <= length(array)
        obj = @atomicswap(:acquire_release, _get_slot(array, id).value = nothing)
        if obj !== nothing
            return obj::T
        end
    end
    return _get_slow(pool)
end

@inline function Base.put!(pool::ThreadObjectPool{T}, obj::T) where T
    array = @atomic :acquire pool.array
    id = Threads.threadid()
    if id <= length(array)
        obj = @atomicswap(:acquire_release, _get_slot(array, id).value = obj)
        if obj === nothing
            return
        end
    end
    _put_slow(pool, obj, id)
end

function Base.empty!(pool::ThreadObjectPool{T}) where T
    array = @atomic :unordered pool.array
    ele1 = _get_slot(array, 1)
    @atomic :unordered ele1.value = nothing
    new_array = Memory{ObjSlot{T}}(undef, 1)
    @inbounds new_array[1] = ele1
    @atomic :release pool.array = new_array
    empty!(pool.extra)
    return
end

function eachobj(pool::ThreadObjectPool{T}) where T
    normal_vals = ((@atomic :unordered s.value) for s in (@atomic :unordered pool.array))
    return Iterators.flatten(((v::T for v in normal_vals if v !== nothing), pool.extra))
end

mutable struct ObjectPool{T,CB}
    const lock::ReentrantLock
    const cb::CB
    @atomic value::Union{T,Nothing}
    const extra::Vector{T} # Protected by lock
    function ObjectPool(cb)
        obj = cb()
        T = typeof(obj)
        return new{T,_typeof(cb)}(ReentrantLock(), cb, obj, T[])
    end
end

function _get_slow(pool::ObjectPool{T}) where T
    @lock pool.lock begin
        if !isempty(pool.extra)
            return pop!(pool.extra)
        end
        return pool.cb()::T
    end
end

function _put_slow(pool::ObjectPool{T}, obj::T) where T
    @lock pool.lock begin
        push!(pool.extra, obj)
    end
    return
end

@inline function Base.get(pool::ObjectPool{T}) where T
    obj = @atomicswap(:acquire_release, pool.value = nothing)
    if obj !== nothing
        return obj::T
    end
    return _get_slow(pool)
end

function Base.put!(pool::ObjectPool{T}, obj::T) where T
    obj = @atomicswap(:acquire_release, pool.value = obj)
    if obj === nothing
        return
    end
    _put_slow(pool, obj)
end

function Base.empty!(pool::ObjectPool)
    @atomic :unordered pool.value = nothing
    empty!(pool.extra)
    return
end

function eachobj(pool::ObjectPool{T}) where T
    normal_vals = (@atomic(:unordered, pool.value),)
    return Iterators.flatten(((v::T for v in normal_vals if v !== nothing), pool.extra))
end

end
