
# Composite type that represents a NDIMS-dimensional tree (parallel version).
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
mutable struct ParallelTree{NDIMS} <: AbstractTree{NDIMS}
  parent_ids::Vector{Int}
  child_ids::Matrix{Int}
  neighbor_ids::Matrix{Int}
  levels::Vector{Int}
  coordinates::Matrix{Float64}
  original_cell_ids::Vector{Int}
  mpi_ranks::Vector{Int}

  capacity::Int
  length::Int
  dummy::Int

  center_level_0::SVector{NDIMS, Float64}
  length_level_0::Float64
  periodicity::NTuple{NDIMS, Bool}

  function ParallelTree{NDIMS}(capacity::Integer) where NDIMS
    # Verify that NDIMS is an integer
    @assert NDIMS isa Integer

    # Create instance
    t = new()

    # Initialize fields with defaults
    # Note: length as capacity + 1 is to use `capacity + 1` as temporary storage for swap operations
    t.parent_ids = fill(typemin(Int), capacity + 1)
    t.child_ids = fill(typemin(Int), 2^NDIMS, capacity + 1)
    t.neighbor_ids = fill(typemin(Int), 2*NDIMS, capacity + 1)
    t.levels = fill(typemin(Int), capacity + 1)
    t.coordinates = fill(NaN, NDIMS, capacity + 1)
    t.original_cell_ids = fill(typemin(Int), capacity + 1)
    t.mpi_ranks = fill(typemin(Int), capacity + 1)

    t.capacity = capacity
    t.length = 0
    t.dummy = capacity + 1

    t.center_level_0 = @SVector fill(NaN, NDIMS)
    t.length_level_0 = NaN

    return t
  end
end


# Constructor for passing the dimension as an argument
ParallelTree(::Val{NDIMS}, args...) where NDIMS = ParallelTree{NDIMS}(args...)

# Create and initialize tree
function ParallelTree{NDIMS}(capacity::Int, center::AbstractArray{Float64},
                 length::Real, periodicity=true) where NDIMS
  # Create instance
  t = ParallelTree{NDIMS}(capacity)

  # Initialize root cell
  init!(t, center, length, periodicity)

  return t
end

# Constructor accepting a single number as center (as opposed to an array) for 1D
ParallelTree{1}(cap::Int, center::Real, len::Real, periodicity=true) = ParallelTree{1}(cap, [convert(Float64, center)], len, periodicity)


# Clear tree with deleting data structures, store center and length, and create root cell
function init!(t::ParallelTree, center::AbstractArray{Float64}, length::Real, periodicity=true)
  clear!(t)

  # Set domain information
  t.center_level_0 = center
  t.length_level_0 = length

  # Create root cell
  t.length += 1
  t.parent_ids[1] = 0
  t.child_ids[:, 1] .= 0
  t.levels[1] = 0
  t.coordinates[:, 1] .= t.center_level_0
  t.original_cell_ids[1] = 0
  t.mpi_ranks[1] = typemin(Int)

  # Set neighbor ids: for each periodic direction, the level-0 cell is its own neighbor
  if all(periodicity)
    # Also catches case where periodicity = true
    t.neighbor_ids[:, 1] .= 1
    t.periodicity = ntuple(x->true, ndims(t))
  elseif !any(periodicity)
    # Also catches case where periodicity = false
    t.neighbor_ids[:, 1] .= 0
    t.periodicity = ntuple(x->false, ndims(t))
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
function Base.show(io::IO, ::MIME"text/plain", t::ParallelTree{NDIMS}) where NDIMS
  l = t.length
  println(io, '*'^20)
  println(io, "t.parent_ids[1:l] = $(t.parent_ids[1:l])")
  println(io, "transpose(t.child_ids[:, 1:l]) = $(transpose(t.child_ids[:, 1:l]))")
  println(io, "transpose(t.neighbor_ids[:, 1:l]) = $(transpose(t.neighbor_ids[:, 1:l]))")
  println(io, "t.levels[1:l] = $(t.levels[1:l])")
  println(io, "transpose(t.coordinates[:, 1:l]) = $(transpose(t.coordinates[:, 1:l]))")
  println(io, "t.original_cell_ids[1:l] = $(t.original_cell_ids[1:l])")
  println(io, "t.mpi_ranks[1:l] = $(t.mpi_ranks[1:l])")
  println(io, "t.capacity = $(t.capacity)")
  println(io, "t.length = $(t.length)")
  println(io, "t.dummy = $(t.dummy)")
  println(io, "t.center_level_0 = $(t.center_level_0)")
  println(io, "t.length_level_0 = $(t.length_level_0)")
  println(io, '*'^20)
end


# Check if cell is own cell, i.e., belongs to this MPI rank
is_own_cell(t::ParallelTree, cell_id) = t.mpi_ranks[cell_id] == mpi_rank()


# Return an array with the ids of all leaf cells for a given rank
leaf_cells_by_rank(t::ParallelTree, rank) = filter_leaf_cells(t) do cell_id
                                              t.mpi_ranks[cell_id] == rank
                                            end

# Return an array with the ids of all local leaf cells
local_leaf_cells(t::ParallelTree) = leaf_cells_by_rank(t, mpi_rank())


# Refine given cells without rebalancing tree.
#
# Note: After a call to this method the tree may be unbalanced!
function refine_unbalanced!(t::ParallelTree, cell_ids)
  # Store actual ids refined cells (shifted due to previous insertions)
  refined = zeros(Int, length(cell_ids))

  # Loop over all cells that are to be refined
  for (count, original_cell_id) in enumerate(sort(unique(cell_ids)))
    # Determine actual cell id, taking into account previously inserted cells
    n_children = n_children_per_cell(t)
    cell_id = original_cell_id + (count - 1) * n_children
    refined[count] = cell_id

    @assert !has_children(t, cell_id) "Non-leaf cell $cell_id cannot be refined"

    # Insert new cells directly behind parent (depth-first)
    insert!(t, cell_id + 1, n_children)

    # Flip sign of refined cell such that we can easily find it later
    t.original_cell_ids[cell_id] = -t.original_cell_ids[cell_id]

    # Initialize child cells
    for child in 1:n_children
      # Set child information based on parent
      child_id = cell_id + child
      t.parent_ids[child_id] = cell_id
      t.child_ids[child, cell_id] = child_id
      t.neighbor_ids[:, child_id] .= 0
      t.child_ids[:, child_id] .= 0
      t.levels[child_id] = t.levels[cell_id] + 1
      t.coordinates[:, child_id] .= child_coordinates(
          t, t.coordinates[:, cell_id], length_at_cell(t, cell_id), child)
      t.original_cell_ids[child_id] = 0
      t.mpi_ranks[child_id] = t.mpi_ranks[cell_id]

      # For determining neighbors, use neighbor connections of parent cell
      for direction in eachdirection(t)
        # If neighbor is a sibling, establish one-sided connectivity
        # Note: two-sided is not necessary, as each sibling will do this
        if has_sibling(child, direction)
          adjacent = adjacent_child(child, direction)
          neighbor_id = cell_id + adjacent

          t.neighbor_ids[direction, child_id] = neighbor_id
          continue
        end

        # Skip if original cell does have no neighbor in direction
        if !has_neighbor(t, cell_id, direction)
          continue
        end

        # Otherwise, check if neighbor has children - if not, skip again
        neighbor_id = t.neighbor_ids[direction, cell_id]
        if !has_children(t, neighbor_id)
          continue
        end

        # Check if neighbor has corresponding child and if yes, establish connectivity
        adjacent = adjacent_child(child, direction)
        if has_child(t, neighbor_id, adjacent)
          neighbor_child_id = t.child_ids[adjacent, neighbor_id]
          opposite = opposite_direction(direction)

          t.neighbor_ids[direction, child_id] = neighbor_child_id
          t.neighbor_ids[opposite, neighbor_child_id] = child_id
        end
      end
    end
  end

  return refined
end


# Reset range of cells to values that are prone to cause errors as soon as they are used.
#
# Rationale: If an invalid cell is accidentally used, we want to know it as soon as possible.
function invalidate!(t::ParallelTree, first::Int, last::Int)
  @assert first > 0
  @assert last <= t.capacity + 1

  # Integer values are set to smallest negative value, floating point values to NaN
  t.parent_ids[first:last] .= typemin(Int)
  t.child_ids[:, first:last] .= typemin(Int)
  t.neighbor_ids[:, first:last] .= typemin(Int)
  t.levels[first:last] .= typemin(Int)
  t.coordinates[:, first:last] .= NaN
  t.original_cell_ids[first:last] .= typemin(Int)
  t.mpi_ranks[first:last] .= typemin(Int)

  return nothing
end


# Raw copy operation for ranges of cells.
#
# This method is used by the higher-level copy operations for AbstractContainer
function raw_copy!(target::ParallelTree, source::ParallelTree, first::Int, last::Int, destination::Int)
  copy_data!(target.parent_ids, source.parent_ids, first, last, destination)
  copy_data!(target.child_ids, source.child_ids, first, last, destination,
             n_children_per_cell(target))
  copy_data!(target.neighbor_ids, source.neighbor_ids, first, last,
             destination, n_directions(target))
  copy_data!(target.levels, source.levels, first, last, destination)
  copy_data!(target.coordinates, source.coordinates, first, last, destination, ndims(target))
  copy_data!(target.original_cell_ids, source.original_cell_ids, first, last, destination)
  copy_data!(target.mpi_ranks, source.mpi_ranks, first, last, destination)
end


# Reset data structures by recreating all internal storage containers and invalidating all elements
function reset_data_structures!(t::ParallelTree{NDIMS}) where NDIMS
  t.parent_ids = Vector{Int}(undef, t.capacity + 1)
  t.child_ids = Matrix{Int}(undef, 2^NDIMS, t.capacity + 1)
  t.neighbor_ids = Matrix{Int}(undef, 2*NDIMS, t.capacity + 1)
  t.levels = Vector{Int}(undef, t.capacity + 1)
  t.coordinates = Matrix{Float64}(undef, NDIMS, t.capacity + 1)
  t.original_cell_ids = Vector{Int}(undef, t.capacity + 1)
  t.mpi_ranks = Vector{Int}(undef, t.capacity + 1)

  invalidate!(t, 1, capacity(t) + 1)
end
