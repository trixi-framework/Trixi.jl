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

function storage_type(::Type{<:FixedSizeArray{T, N, Mem}}) where {T, N, Mem}
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
# multi-dimensional array of size `dims` without copying. This is used for the
# containers (initial creation, `resize!`, and `Adapt.adapt_structure`). All callers
# adapt or resize `vector` to the target storage type before wrapping it, so we can
# dispatch on the type of `vector` itself.
#
# For CPU `Vector`s, this returns a `VectorBackedArray` (see `wrap_vector`), which keeps
# a reference to `vector`. This is safe with respect to the garbage collector for all
# element types (including non-`isbits` ones) and allows `resize!`ing `vector`
# afterwards (followed by wrapping it again), also on Julia v1.10.
#
# For all other vectors, we use `reshape`. For GPU arrays such as `CuArray` or `ROCArray`,
# the result shares the (reference-counted) buffer of `vector`. This also covers device
# arrays used inside kernels (e.g., `CuDeviceArray`) and CPU views.
@inline wrap_storage(vector::Vector, dims) = wrap_vector(vector, dims)
@inline wrap_storage(vector::AbstractVector, dims) = reshape(vector, dims)

struct TrixiAdaptor{Storage, RealT} end

"""
    trixi_adapt(Storage, RealT, x)

Adapt `x` to the storage type `Storage` and real type `RealT`.

To adapt a semidiscretization, use the method
`trixi_adapt(semi; storage_type, real_type, flux_differencing_kernel)` with keyword
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

# An N-dimensional view of a `resize!`able backing `Vector`, used instead of
# `unsafe_wrap(Array, pointer(v), dims)` in containers that are resized by AMR.
# Since Julia v1.11, `Array` is a mutable struct, so its size and data pointer are
# loaded from memory and LLVM often cannot hoist these loads out of inner loops.
# A `FixedSizeArray` is immutable, so its dimensions are plain values. It holds a
# reference to `v`, so the memory stays rooted (no `GC.@preserve` needed) and stores
# of non-isbits elements get the necessary write barriers.
# The view must be re-created after every `resize!` of `v`.
const VectorBackedArray{T, N} = FixedSizeArray{T, N, Vector{T}}

@inline function wrap_vector(v::Vector, dims)
    return FixedSizeArrays.wrap(FixedSizeArray, v, dims)
end

# Product of the static sizes `S`, computed at compile time. `prod(S.parameters)` iterates
# over a `Core.SimpleVector` at runtime, which dispatches dynamically and allocates.
@generated static_prod(::Type{S}) where {S <: Tuple} = prod(S.parameters)

# An `N`-dimensional array backed by the vector `data` (column-major, without copying),
# whose sizes `S` (a `Tuple` type) of all but the last dimension are known at compile
# time, while the size `nlast` of the last dimension is only known at runtime.
# This is the typical layout of the solution and its time derivative in Trixi.jl
# (`[variable, node..., element]`), where the number of variables and nodes is fixed
# but the number of elements changes, e.g., with AMR. Knowing the strides at compile
# time simplifies the index computations in performance-critical kernels.
# The array keeps a reference to `data`, so the memory stays alive. It works with any
# `AbstractVector` storage, e.g., GPU arrays, and is converted to a device array by Adapt.jl.
# Polyester.jl converts it to a `PtrArray` with static sizes via the StaticArrayInterface.jl
# traits defined below. See `static_view`.
#
# The elements are accessed via `ref`, which is the `MemoryRef` of `data` if `data` is a
# `Vector` (on Julia v1.11 and newer) and `data` itself otherwise. Since Julia v1.11,
# `Vector` is a mutable struct whose data pointer can change (`resize!`), so it must be
# loaded from the `Vector` object for each access. LLVM can hoist this load out of an
# inner loop only if the loop is guaranteed to run, since it does not know that the
# `Vector` object is dereferenceable. Thus, the load stays, e.g., in front of the inner
# loops of the flux differencing kernels, which only run for some nodes (`ii > i`).
# This costs about 2-3% in the volume integral of `elixir_euler_ec.jl` in 2D.
# The `MemoryRef` is an immutable part of this array (like the pointer of the `PtrArray`s
# that `wrap_array` returned when Trixi.jl used Polyester.jl), so the data pointer is
# loaded only once. It also keeps the memory alive. The `MemoryRef` is not updated by
# `resize!(data, ...)`, so (as the size `nlast`) the view must be re-created afterward.
struct StaticSizeArray{T, N, S <: Tuple, P <: AbstractVector{T}, R} <: DenseArray{T, N}
    data::P
    ref::R
    nlast::Int

    function StaticSizeArray{S}(data::AbstractVector{T}, nlast::Integer) where {S, T}
        N = length(S.parameters) + 1
        if length(data) != static_prod(S) * nlast
            throw(DimensionMismatch("length of the data $(length(data)) does not match the size $((S.parameters..., nlast))"))
        end
        ref = storage_ref(data)
        return new{T, N, S, typeof(data), typeof(ref)}(data, ref, nlast)
    end
end

# See the comment above `StaticSizeArray`
@inline storage_ref(data::AbstractVector) = data
Base.@propagate_inbounds storage_getindex(data::AbstractVector, i::Int) = data[i]
Base.@propagate_inbounds function storage_setindex!(data::AbstractVector, x, i::Int)
    data[i] = x
    return nothing
end
@static if VERSION >= v"1.11"
    @inline storage_ref(data::Vector) = data.ref
    # Bounds are checked by the callers, i.e., `getindex` and `setindex!` of
    # `StaticSizeArray` (as in `getindex(::Array, ::Int)` for the `MemoryRef` of an `Array`)
    @inline function storage_getindex(ref::MemoryRef, i::Int)
        return Core.memoryrefget(Core.memoryrefnew(ref, i, false), :not_atomic, false)
    end
    @inline function storage_setindex!(ref::MemoryRef{T}, x, i::Int) where {T}
        Core.memoryrefset!(Core.memoryrefnew(ref, i, false), convert(T, x),
                           :not_atomic, false)
        return nothing
    end
end

@inline Base.size(A::StaticSizeArray{T, N, S}) where {T, N, S} = (S.parameters..., A.nlast)
@inline Base.parent(A::StaticSizeArray) = A.data
# Linear indexing goes straight to the storage (e.g., for broadcasts over the whole array)
Base.IndexStyle(::Type{<:StaticSizeArray}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(A::StaticSizeArray, i::Int)
    @boundscheck checkbounds(A, i)
    return @inbounds storage_getindex(A.ref, i)
end
Base.@propagate_inbounds function Base.setindex!(A::StaticSizeArray, x, i::Int)
    @boundscheck checkbounds(A, i)
    @inbounds storage_setindex!(A.ref, x, i)
    return A
end

# Linear index in Horner form, e.g., `i + n1 * ((j - 1) + n2 * (k - 1))` in 3D, as in
# StrideArraysCore.jl
@generated function linear_index(::StaticSizeArray{T, N, S},
                                 I::Vararg{Int, N}) where {T, N, S}
    dims = S.parameters
    ex = :(I[$N] - 1)
    for d in (N - 1):-1:2
        ex = :((I[$d] - 1) + $(dims[d]) * $ex)
    end
    return N == 1 ? :(I[1]) : :(I[1] + $(dims[1]) * $ex)
end

Base.@propagate_inbounds function Base.getindex(A::StaticSizeArray{T, N},
                                                I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(A, I...)
    return @inbounds storage_getindex(A.ref, linear_index(A, I...))
end

Base.@propagate_inbounds function Base.setindex!(A::StaticSizeArray{T, N}, x,
                                                 I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(A, I...)
    @inbounds storage_setindex!(A.ref, x, linear_index(A, I...))
    return A
end

# Strided array interface (for `pointer`, LoopVectorization.jl, and Polyester.jl)
@generated function Base.strides(::StaticSizeArray{T, N, S}) where {T, N, S}
    strides = Any[1]
    for d in 2:N
        push!(strides, strides[end] * S.parameters[d - 1])
    end
    return Expr(:tuple, strides...)
end
Base.elsize(::Type{<:StaticSizeArray{T}}) where {T} = Base.elsize(Vector{T})
function Base.unsafe_convert(::Type{Ptr{T}}, A::StaticSizeArray{T}) where {T}
    return Base.unsafe_convert(Ptr{T}, A.data)
end

# Static sizes and strides for StaticArrayInterface.jl
function StaticArrayInterface.static_size(A::StaticSizeArray{T, N, S}) where {T, N, S}
    return (map(StaticInt, Tuple(S.parameters))..., A.nlast)
end
@generated function StaticArrayInterface.static_strides(::StaticSizeArray{T, N,
                                                                          S}) where {T, N,
                                                                                     S}
    strides = Any[1]
    for d in 2:N
        push!(strides, strides[end] * S.parameters[d - 1])
    end
    return Expr(:tuple, (:(StaticInt($s)) for s in strides)...)
end

function Adapt.adapt_structure(to, A::StaticSizeArray{T, N, S}) where {T, N, S}
    return StaticSizeArray{S}(adapt(to, A.data), A.nlast)
end

KernelAbstractions.get_backend(A::StaticSizeArray) = get_backend(A.data)

"""
    static_view(A, equations, dg)

Return a view of the solution-like array `A` with dimensions
`[variable, node..., element]` whose sizes of all but the last dimension (i.e.,
`nvariables(equations)` and `nnodes(dg)`) are known at compile time, without copying.
Returns `A` itself for array types without such a view. Requires that `nnodes(dg)` is
known at compile time (e.g., `DGSEM`). `wrap_array` returns such a view for `DGSEM`.
"""
@inline function static_view(A::VectorBackedArray, equations, dg)
    S = static_sizes(Val(nvariables(equations)), Val(nnodes(dg)), Val(ndims(A) - 2))
    return StaticSizeArray{S}(parent(A), size(A, ndims(A)))
end
# `Tuple{nvariables, nnodes, ...}` for `ndims_nodes` node dimensions. A generated function
# so that the type is inferred also on Julia v1.10, which does not infer
# `Tuple{nvariables(equations), ntuple(_ -> nnodes(dg), ndims(A) - 2)...}`; `wrap_array`
# returns this view and must be type stable.
@generated function static_sizes(::Val{NVARS}, ::Val{NNODES},
                                 ::Val{NDIMS_NODES}) where {NVARS, NNODES, NDIMS_NODES}
    return Tuple{NVARS, ntuple(_ -> NNODES, NDIMS_NODES)...}
end
@inline static_view(A, equations, dg) = A

# Array types accepted by the hand-written `@turbo` kernels (e.g., specializations of
# `flux_differencing_kernel!`). We restrict the element types to those supported natively
# by LoopVectorization.jl (e.g., no `ForwardDiff.Dual`s), mirroring
# `LoopVectorization.check_args`.
# With Polyester.jl as threading backend, these are the arrays returned by `wrap_array`:
# a `StaticSizeArray` (`DGSEM`) or `VectorBackedArray` for CPU `Vector`s, and a reshaped
# contiguous view of an `Array`, e.g., a column of a matrix in `linear_structure` or
# `jacobian_fd`, or a part of `u_ode` in coupled semidiscretizations. Threading via
# Polyester.jl may convert them to `PtrArray`s. With other threading backends, the
# `@turbo` kernels are not selected for the arrays returned by `wrap_array` (see the
# "Heisenbug" note in `@threaded`).
@static if _PREFERENCE_THREADING === :polyester
    const TurboArray = Union{PtrArray{<:LoopVectorization.NativeTypes},
                             VectorBackedArray{<:LoopVectorization.NativeTypes},
                             StaticSizeArray{<:LoopVectorization.NativeTypes, N, S,
                                             <:Vector} where {N, S},
                             Base.ReshapedArray{<:LoopVectorization.NativeTypes, N,
                                                <:Base.FastContiguousSubArray{<:Any, 1,
                                                                              <:Array}} where {N}}
else
    const TurboArray = PtrArray{<:LoopVectorization.NativeTypes}
end
