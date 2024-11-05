# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Composite type that represents a NDIMS-dimensional tree (serial version).
#
# Implements everything required for AbstractContainer.
#
# Note: The way the data structures are set up and the way most algorithms
# work, it is *always* assumed that
#   a) we have a balanced tree (= at most one level difference between
#                                 neighboring cells, or 2:1 rule)
#   b) we may not have all children (= some children may not exist)
#   c) the tree is stored depth-first
#
# However, the way the refinement/coarsening algorithms are currently
# implemented, we only have fully refined cells. That is, a cell either has 2^NDIMS children or
# no children at all (= leaf cell). This restriction is also assumed at
# multiple positions in the refinement/coarsening algorithms.
#
# An exception to the 2:1 rule exists for the low-level `refine_unbalanced!`
# function, which is required for implementing level-wise refinement in a sane
# way. Also, depth-first ordering *might* not by guaranteed during
# refinement/coarsening operations.
mutable struct SerialTree{NDIMS, RealT <: Real} <: AbstractTree{NDIMS}
    parent_ids::Vector{Int}
    child_ids::Matrix{Int}
    neighbor_ids::Matrix{Int}
    levels::Vector{Int}
    coordinates::Matrix{RealT}
    original_cell_ids::Vector{Int}

    capacity::Int
    length::Int
    dummy::Int

    center_level_0::SVector{NDIMS, RealT}
    length_level_0::RealT
    periodicity::NTuple{NDIMS, Bool}

    function SerialTree{NDIMS, RealT}(capacity::Integer) where {NDIMS, RealT <: Real}
        # Verify that NDIMS is an integer
        @assert NDIMS isa Integer

        # Create instance
        t = new()

        # Initialize fields with defaults
        # Note: length as capacity + 1 is to use `capacity + 1` as temporary storage for swap operations
        t.parent_ids = fill(typemin(Int), capacity + 1)
        t.child_ids = fill(typemin(Int), 2^NDIMS, capacity + 1)
        t.neighbor_ids = fill(typemin(Int), 2 * NDIMS, capacity + 1)
        t.levels = fill(typemin(Int), capacity + 1)
        t.coordinates = fill(convert(RealT, NaN), NDIMS, capacity + 1) # `NaN` is of type Float64
        t.original_cell_ids = fill(typemin(Int), capacity + 1)

        t.capacity = capacity
        t.length = 0
        t.dummy = capacity + 1

        t.center_level_0 = SVector(ntuple(_ -> convert(RealT, NaN), NDIMS))
        t.length_level_0 = convert(RealT, NaN)

        return t
    end
end

# Constructor for passing the dimension as an argument. Default datatype: Float64
SerialTree(::Val{NDIMS}, args...) where {NDIMS} = SerialTree{NDIMS, Float64}(args...)

# Create and initialize tree
function SerialTree{NDIMS, RealT}(capacity::Int, center::AbstractArray{RealT},
                                  length::RealT,
                                  periodicity = true) where {NDIMS, RealT <: Real}
    # Create instance
    t = SerialTree{NDIMS, RealT}(capacity)

    # Initialize root cell
    init!(t, center, length, periodicity)

    return t
end
function SerialTree{NDIMS}(capacity::Int, center::AbstractArray{RealT},
                           length::RealT,
                           periodicity = true) where {NDIMS, RealT <: Real}
    t = SerialTree{NDIMS, RealT}(capacity, center, length, periodicity)
end

# Constructors accepting a single number as center (as opposed to an array) for 1D
function SerialTree{1, RealT}(cap::Int, center::RealT, len::RealT,
                              periodicity = true) where {RealT <: Real}
    SerialTree{1, RealT}(cap, [center], len, periodicity)
end
function SerialTree{1}(cap::Int, center::RealT, len::RealT,
                       periodicity = true) where {RealT <: Real}
    SerialTree{1, RealT}(cap, [center], len, periodicity)
end

# Clear tree with deleting data structures, store center and length, and create root cell
function init!(t::SerialTree, center::AbstractArray{RealT}, length::RealT,
               periodicity = true) where {RealT}
    clear!(t)

    # Set domain information
    t.center_level_0 = center
    t.length_level_0 = length

    # Create root cell
    t.length += 1
    t.parent_ids[1] = 0
    t.child_ids[:, 1] .= 0
    t.levels[1] = 0
    set_cell_coordinates!(t, t.center_level_0, 1)
    t.original_cell_ids[1] = 0

    # Set neighbor ids: for each periodic direction, the level-0 cell is its own neighbor
    if all(periodicity)
        # Also catches case where periodicity = true
        t.neighbor_ids[:, 1] .= 1
        t.periodicity = ntuple(x -> true, ndims(t))
    elseif !any(periodicity)
        # Also catches case where periodicity = false
        t.neighbor_ids[:, 1] .= 0
        t.periodicity = ntuple(x -> false, ndims(t))
    else
        # Default case if periodicity is an iterable
        for dimension in 1:ndims(t)
            if periodicity[dimension]
                t.neighbor_ids[2 * dimension - 1, 1] = 1
                t.neighbor_ids[2 * dimension - 0, 1] = 1
            else
                t.neighbor_ids[2 * dimension - 1, 1] = 0
                t.neighbor_ids[2 * dimension - 0, 1] = 0
            end
        end

        t.periodicity = Tuple(periodicity)
    end
end

# Convenience output for debugging
function Base.show(io::IO, ::MIME"text/plain", t::SerialTree)
    @nospecialize t # reduce precompilation time

    l = t.length
    println(io, '*'^20)
    println(io, "t.parent_ids[1:l] = $(t.parent_ids[1:l])")
    println(io, "transpose(t.child_ids[:, 1:l]) = $(transpose(t.child_ids[:, 1:l]))")
    println(io,
            "transpose(t.neighbor_ids[:, 1:l]) = $(transpose(t.neighbor_ids[:, 1:l]))")
    println(io, "t.levels[1:l] = $(t.levels[1:l])")
    println(io,
            "transpose(t.coordinates[:, 1:l]) = $(transpose(t.coordinates[:, 1:l]))")
    println(io, "t.original_cell_ids[1:l] = $(t.original_cell_ids[1:l])")
    println(io, "t.capacity = $(t.capacity)")
    println(io, "t.length = $(t.length)")
    println(io, "t.dummy = $(t.dummy)")
    println(io, "t.center_level_0 = $(t.center_level_0)")
    println(io, "t.length_level_0 = $(t.length_level_0)")
    println(io, '*'^20)
end

# Set information for child cell `child_id` based on parent cell `cell_id` (except neighbors)
function init_child!(t::SerialTree, cell_id, child, child_id)
    t.parent_ids[child_id] = cell_id
    t.child_ids[child, cell_id] = child_id
    t.child_ids[:, child_id] .= 0
    t.levels[child_id] = t.levels[cell_id] + 1
    set_cell_coordinates!(t,
                          child_coordinates(t, cell_coordinates(t, cell_id),
                                            length_at_cell(t, cell_id), child),
                          child_id)
    t.original_cell_ids[child_id] = 0

    return nothing
end

# Reset range of cells to values that are prone to cause errors as soon as they are used.
#
# Rationale: If an invalid cell is accidentally used, we want to know it as soon as possible.
function invalidate!(t::SerialTree{NDIMS, RealT},
                     first::Int, last::Int) where {NDIMS, RealT <: Real}
    @assert first > 0
    @assert last <= t.capacity + 1

    # Integer values are set to smallest negative value, floating point values to NaN
    t.parent_ids[first:last] .= typemin(Int)
    t.child_ids[:, first:last] .= typemin(Int)
    t.neighbor_ids[:, first:last] .= typemin(Int)
    t.levels[first:last] .= typemin(Int)
    t.coordinates[:, first:last] .= convert(RealT, NaN) # `NaN` is of type Float64
    t.original_cell_ids[first:last] .= typemin(Int)

    return nothing
end

# Raw copy operation for ranges of cells.
#
# This method is used by the higher-level copy operations for AbstractContainer
function raw_copy!(target::SerialTree, source::SerialTree, first::Int, last::Int,
                   destination::Int)
    copy_data!(target.parent_ids, source.parent_ids, first, last, destination)
    copy_data!(target.child_ids, source.child_ids, first, last, destination,
               n_children_per_cell(target))
    copy_data!(target.neighbor_ids, source.neighbor_ids, first, last,
               destination, n_directions(target))
    copy_data!(target.levels, source.levels, first, last, destination)
    copy_data!(target.coordinates, source.coordinates, first, last, destination,
               ndims(target))
    copy_data!(target.original_cell_ids, source.original_cell_ids, first, last,
               destination)
end

# Reset data structures by recreating all internal storage containers and invalidating all elements
function reset_data_structures!(t::SerialTree{NDIMS, RealT}) where {NDIMS, RealT <:
                                                                           Real}
    t.parent_ids = Vector{Int}(undef, t.capacity + 1)
    t.child_ids = Matrix{Int}(undef, 2^NDIMS, t.capacity + 1)
    t.neighbor_ids = Matrix{Int}(undef, 2 * NDIMS, t.capacity + 1)
    t.levels = Vector{Int}(undef, t.capacity + 1)
    t.coordinates = Matrix{RealT}(undef, NDIMS, t.capacity + 1)
    t.original_cell_ids = Vector{Int}(undef, t.capacity + 1)

    invalidate!(t, 1, capacity(t) + 1)
end
end # @muladd
