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

function storage_type(::Type{<:FixedSizeArrayDims{T, N, D, Mem}}) where {T, N, D, Mem}
    return storage_type(Mem)
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

"""
    trixi_backend_info!(setup, backend)

Add information about the computational backend to `setup`, which is a vector of key-value pairs.
Used by Trixi.jl's summary callback to print information about the simulation setup.
"""
function trixi_backend_info!(_, ::Nothing)
    return nothing
end

function trixi_backend_info!(setup, ::KernelAbstractions.CPU)
    push!(setup, "Backend" => "KernelAbstractions CPU")
    return nothing
end

function trixi_backend_info!(setup, ::KernelAbstractions.Backend)
    push!(setup, "Backend" => "Unknown KernelAbstractions backend")
    return nothing
end

"""
    trixi_device_memory_use(backend)

Return the device memory use in bytes for the given `backend`, or `nothing`.
Reports the current active device and memory pool.
"""
function trixi_device_memory_use(backend::Union{Nothing, KernelAbstractions.Backend})
    return nothing
end

# Wrap the internal (`resize!`able) storage `vector` of a container as a
# multi-dimensional array of size `dims` without copying. Only one-dimensional arrays
# can be `resize!`d in Julia, so containers keep their data in `vector`s and wrap them
# (at creation, after every `resize!`, and in `Adapt.adapt_structure`). `wrap_array`
# wraps the solution vector `u_ode` in the same way. All callers adapt or resize `vector`
# to the target storage type before wrapping it, so we can dispatch on its type.
# The wrapped array must be re-created after every `resize!` of `vector`.
#
# For CPU `Vector`s, this returns a `VectorBackedArray`, an immutable `FixedSizeArray`
# backed by `vector`. Previously, we used `unsafe_wrap(Array, pointer(vector), dims)`
# here. In contrast, a `VectorBackedArray`
# - keeps a reference to `vector`, so the memory stays rooted and temporaries do not
#   need to be `GC.@preserve`d,
# - is safe with respect to the garbage collector also for non-`isbits` element types
#   (e.g., `BigFloat`s), since stores trigger the necessary write barriers,
# - allows `resize!`ing `vector` also on Julia v1.10, where `reshape(vector, dims)` marks
#   `vector` as shared (`ERROR: cannot resize array with shared data`),
# - has dimensions that are plain values. Since Julia v1.11, `Array` is a mutable struct,
#   so its size and data pointer are loaded from memory and LLVM often cannot hoist these
#   loads out of inner loops.
# It is a `DenseArray` that can be passed to external libraries such as MPI (via the
# MPI.jl package extension of FixedSizeArrays.jl) and HDF5.
#
# For all other vectors, we use `reshape`. For GPU arrays such as `CuArray` or `ROCArray`,
# the result shares the (reference-counted) buffer of `vector`. This also covers device
# arrays used inside kernels (e.g., `CuDeviceArray`) and CPU views that are never
# `resize!`d, e.g., columns of a matrix in `linear_structure` or `jacobian_fd` and parts
# of `u_ode` in coupled semidiscretizations.
const VectorBackedArray{T, N} = FixedSizeArray{T, N, Vector{T}}

@inline function wrap_storage(vector::Vector, dims)
    return FixedSizeArrays.wrap(FixedSizeArray, vector, dims)
end
@inline wrap_storage(vector::AbstractVector, dims) = reshape(vector, dims)

struct TrixiAdaptor{Storage, RealT} end

"""
    trixi_adapt(Storage, RealT, x)

Adapt `x` to the storage type `Storage` and real type `RealT`.

To adapt a semidiscretization, use the method
`trixi_adapt(semi; storage_type, real_type, tuning)` with keyword
arguments instead.
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

# 3. Equations and equations of state: delegate to Base.similar
function Adapt.adapt_structure(::TrixiAdaptor{<:Any, NewRealT},
                               equations::AbstractEquations) where {NewRealT}
    return similar(equations, NewRealT)
end

# 4. TODO: Should we have a fallback? But that would imply implementing things for NamedTuple again

end # @muladd
