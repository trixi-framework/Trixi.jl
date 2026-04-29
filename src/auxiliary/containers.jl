# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Abstract base type - all containers that want to use these features must inherit from it
abstract type AbstractContainer end

# Generic functions for which concrete containers must add implementations
function invalidate! end
function raw_copy! end
function move_connectivity! end
function delete_connectivity! end
function reset_data_structures! end

# Auxiliary copy function to copy data between containers
function copy_data!(target::AbstractArray, source::AbstractArray,
                    first::Int, last::Int, destination::Int, block_size::Int = 1)
    count = last - first + 1
    if destination <= first || destination > last
        # In this case it is safe to copy forward (left-to-right) without overwriting data
        for i in 0:(count - 1), j in 1:block_size
            target[block_size * (destination + i - 1) + j] = source[block_size * (first + i - 1) + j]
        end
    else
        # In this case we need to copy backward (right-to-left) to prevent overwriting data
        for i in (count - 1):-1:0, j in 1:block_size
            target[block_size * (destination + i - 1) + j] = source[block_size * (first + i - 1) + j]
        end
    end

    return target
end

# Inquire about capacity and size
capacity(c::AbstractContainer) = c.capacity
Base.length(c::AbstractContainer) = c.length
Base.size(c::AbstractContainer) = (length(c),)

"""
    resize!(c::AbstractContainer, new_length) -> AbstractContainer

Resize `c` to contain `new_length` elements. If `new_length` is smaller than the current container
length, the first `new_length` elements will be retained. If `new_length` is
larger, the new elements are invalidated.
"""
function Base.resize!(c::AbstractContainer, new_length)
    @assert new_length>=zero(new_length) "New length must be >= 0"
    @assert new_length<=capacity(c) "New length would exceed capacity"

    # If new length is greater than current length, append to container.
    # If new length is less than current length, shrink container.
    # If new length is equal to current length, do nothing.
    if new_length > length(c)
        # First, invalidate range (to be sure that no sensible values are accidentally left there)
        invalidate!(c, length(c) + 1, new_length)

        # Then, set new container length
        c.length = new_length
    elseif new_length < length(c)
        # Rely on remove&shift to do The Right Thing (`remove_shift!` also updates the length)
        remove_shift!(c, new_length + 1, length(c))
    end

    return c
end

# Copy data range from source to target container.
#
# Calls `raw_copy` internally, which must be implemented for each concrete type
# inheriting from AbstractContainer.
# TODO: Shall we extend Base.copyto! ?
function Trixi.copy!(target::AbstractContainer, source::AbstractContainer,
                     first::Int, last::Int, destination::Int)
    @assert 1<=first<=length(source) "First cell out of range"
    @assert 1<=last<=length(source) "Last cell out of range"
    @assert 1<=destination<=length(target) "Destination out of range"
    @assert destination + (last - first)<=length(target) "Target range out of bounds"

    # Return if copy would be a no-op
    if last < first || (source === target && first == destination)
        return target
    end

    raw_copy!(target, source, first, last, destination)

    return target
end

# Convenience method to copy a single element
function Trixi.copy!(target::AbstractContainer, source::AbstractContainer, from::Int,
                     destination::Int)
    return Trixi.copy!(target, source, from, from, destination)
end

# Convenience method for copies within a single container
function Trixi.copy!(c::AbstractContainer, first::Int, last::Int, destination::Int)
    return Trixi.copy!(c, c, first, last, destination)
end

# Convenience method for copying a single element within a single container
function Trixi.copy!(c::AbstractContainer, from::Int, destination::Int)
    return Trixi.copy!(c, c, from, from, destination)
end

# Move elements in a way that preserves connectivity.
function move!(c::AbstractContainer, first::Int, last::Int, destination::Int)
    @assert 1<=first<=length(c) "First cell $first out of range"
    @assert 1<=last<=length(c) "Last cell $last out of range"
    @assert 1<=destination<=length(c) "Destination $destination out of range"
    @assert destination + (last - first)<=length(c) "Target range out of bounds"

    # Return if move would be a no-op
    if last < first || first == destination
        return c
    end

    # Copy cells to new location
    raw_copy!(c, first, last, destination)

    # Move connectivity
    move_connectivity!(c, first, last, destination)

    # Invalidate original cell locations (unless they already contain new data due to overlap)
    # 1) If end of destination range is within original range, shift first_invalid to the right
    count = last - first + 1
    first_invalid = (first <= destination + count - 1 <= last) ? destination + count :
                    first
    # 2) If beginning of destination range is within original range, shift last_invalid to the left
    last_invalid = (first <= destination <= last) ? destination - 1 : last
    # 3) Invalidate range
    invalidate!(c, first_invalid, last_invalid)

    return c
end
function move!(c::AbstractContainer, from::Int, destination::Int)
    return move!(c, from, from, destination)
end

# Default implementation for moving a single element
function move_connectivity!(c::AbstractContainer, from::Int, destination::Int)
    return move_connectivity!(c, from, from, destination)
end

# Default implementation for invalidating a single element
function invalidate!(c::AbstractContainer, id::Int)
    return invalidate!(c, id, id)
end

# Insert blank elements in container, shifting the following elements back.
#
# After a call to insert!, the range `position:position + count - 1` will be available for use.
# TODO: Shall we extend Base.insert! ?
function insert!(c::AbstractContainer, position::Int, count::Int)
    @assert 1<=position<=length(c)+1 "Insert position out of range"
    @assert count>=0 "Count must be non-negative"
    @assert count + length(c)<=capacity(c) "New length would exceed capacity"

    # Return if insertation would be a no-op
    if count == 0
        return c
    end

    # Append and return if insertion is beyond last current element
    if position == length(c) + 1
        resize!(c, length(c) + count)
        return c
    end

    # Increase length
    c.length += count

    # Move original cells that currently occupy the insertion region, unless
    # insert position is one beyond previous length
    if position <= length(c) - count
        move!(c, position, length(c) - count, position + count)
    end

    return c
end

# Remove cells and shift existing cells forward to close the gap
function remove_shift!(c::AbstractContainer, first::Int, last::Int)
    @assert 1<=first<=length(c) "First cell out of range"
    @assert 1<=last<=length(c) "Last cell out of range"

    # Return if removal would be a no-op
    if last < first
        return c
    end

    # Delete connectivity of cells to be removed
    delete_connectivity!(c, first, last)

    if last == length(c)
        # If everything up to the last cell is removed, no shifting is required
        invalidate!(c, first, last)
    else
        # Otherwise, the corresponding cells are moved forward
        move!(c, last + 1, length(c), first)
    end

    # Reduce length
    count = last - first + 1
    c.length -= count

    return c
end
remove_shift!(c::AbstractContainer, id::Int) = remove_shift!(c, id, id)

# Invalidate all elements and set length to zero.
function clear!(c::AbstractContainer)
    invalidate!(c)
    c.length = 0

    return c
end

# Helpful overloads for `raw_copy`
function raw_copy!(c::AbstractContainer, first::Int, last::Int, destination::Int)
    return raw_copy!(c, c, first, last, destination)
end
function raw_copy!(target::AbstractContainer, source::AbstractContainer, from::Int,
                   destination::Int)
    return raw_copy!(target, source, from, from, destination)
end
function raw_copy!(c::AbstractContainer, from::Int, destination::Int)
    return raw_copy!(c, c, from, from, destination)
end

# Trixi storage types must implement these two Adapt.jl methods
function Adapt.adapt_structure(to, c::AbstractContainer)
    error("Interface: Must implement Adapt.adapt_structure(to, ::$(typeof(c)))")
    return nothing
end

function Adapt.parent_type(C::Type{<:AbstractContainer})
    error("Interface: Must implement Adapt.parent_type(::Type{$C}")
    return nothing
end

function Adapt.unwrap_type(C::Type{<:AbstractContainer})
    return Adapt.unwrap_type(Adapt.parent_type(C))
end

# TODO: Upstream to Adapt
"""
    storage_type(x)

Return the storage type of `x`, which is a concrete array type, such as `Array`, `CuArray`, or `ROCArray`.
"""
function storage_type(x)
    return storage_type(typeof(x))
end

function storage_type(T::Type)
    error("Interface: Must implement storage_type(::Type{$T}")
    return nothing
end

function storage_type(::Type{<:Array})
    return Array
end

function storage_type(C::Type{<:AbstractContainer})
    return storage_type(Adapt.unwrap_type(C))
end

# backend handling
"""
    trixi_backend(x)

Return the computational backend for `x`, which is either a KernelAbstractions backend or `nothing`.
If the backend is `nothing`, the default multi-threaded CPU backend is used.
"""
function trixi_backend(x)
    if (_PREFERENCE_THREADING === :polyester && LoopVectorization.check_args(x)) ||
       (_PREFERENCE_THREADING !== :kernelabstractions &&
        get_backend(x) isa KernelAbstractions.CPU)
        return nothing
    end
    return get_backend(x)
end

# TODO: After https://github.com/SciML/RecursiveArrayTools.jl/pull/455 we need to investigate the right way to handle StaticArray as uEltype for MultiDG.
function trixi_backend(x::VectorOfArray)
    u = parent(x)
    # FIXME(vchuravy): This is a workaround because KA.get_backend is ambivalent of where a SArray is residing.
    if eltype(u) <: StaticArrays.StaticArray
        return nothing
    end
    if length(u) == 0
        error("VectorOfArray is empty, cannot determine backend.")
    end
    # Use the backend of the first element in the parent array
    return get_backend(u[1])
end

# For some storage backends like CUDA.jl, empty arrays do seem to simply be
# null pointers which can cause `unsafe_wrap` to fail when calling
# Adapt.adapt (ArgumentError, see
# https://github.com/JuliaGPU/CUDA.jl/blob/v5.4.2/src/array.jl#L212-L229).
# To circumvent this, on length zero arrays this allocates
# a separate empty array instead of wrapping.
# However, since zero length arrays are not used in calculations,
# it should be okay if the underlying storage vectors and wrapped arrays
# are not the same as long as they are properly wrapped when `resize!`d etc.
function unsafe_wrap_or_alloc(to, vector, size)
    if length(vector) == 0
        return similar(vector, size)
    else
        return unsafe_wrap(to, pointer(vector), size)
    end
end

struct TrixiAdaptor{Storage, RealT} end

"""
    trixi_adapt(Storage, RealT, x)

Adapt `x` to the storage type `Storage` and real type `RealT`.
"""
function trixi_adapt(Storage, RealT, x)
    return adapt(TrixiAdaptor{Storage, RealT}(), x)
end

# Custom rules
# 1. handling of StaticArrays
function Adapt.adapt_storage(::TrixiAdaptor{<:Any, RealT},
                             x::StaticArrays.StaticArray) where {RealT}
    return StaticArrays.similar_type(x, RealT)(x)
end

# 2. Handling of Arrays
function Adapt.adapt_storage(::TrixiAdaptor{Storage, RealT},
                             x::AbstractArray{T}) where {Storage, RealT,
                                                         T <: AbstractFloat}
    return adapt(Storage{RealT}, x)
end

function Adapt.adapt_storage(::TrixiAdaptor{Storage, RealT},
                             x::AbstractArray{T}) where {Storage, RealT,
                                                         T <: StaticArrays.StaticArray}
    return adapt(Storage{StaticArrays.similar_type(T, RealT)}, x)
end

# Our threaded cache contains MArray, it is unlikely that we would want to adapt those
function Adapt.adapt_storage(::TrixiAdaptor{Storage, RealT},
                             x::Array{T}) where {Storage, RealT,
                                                 T <: StaticArrays.MArray}
    return adapt(Array{StaticArrays.similar_type(T, RealT)}, x)
end

function Adapt.adapt_storage(::TrixiAdaptor{Storage, RealT},
                             x::AbstractArray) where {Storage, RealT}
    return adapt(Storage, x)
end

# 3. TODO: Should we have a fallback? But that would imply implementing things for NamedTuple again

function unsafe_wrap_or_alloc(::TrixiAdaptor{Storage}, vec, size) where {Storage}
    return unsafe_wrap_or_alloc(Storage, vec, size)
end
end # @muladd
