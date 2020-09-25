
# Composite type that represents a NDIMS-dimensional tree.
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
function Base.show(io::IO, t::ParallelTree{NDIMS}) where NDIMS
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

# Type traits to obtain dimension
@inline Base.ndims(t::Type{ParallelTree{NDIMS}}) where NDIMS = NDIMS
@inline Base.ndims(t::ParallelTree) = ndims(typeof(t))


# Auxiliary methods to allow semantic queries on the tree
# Check whether cell has parent cell
has_parent(t::ParallelTree, cell_id::Int) = t.parent_ids[cell_id] > 0

# Count number of children for a given cell
n_children(t::ParallelTree, cell_id::Int) = count(x -> (x > 0), @view t.child_ids[:, cell_id])

# Check whether cell has any child cell
has_children(t::ParallelTree, cell_id::Int) = n_children(t, cell_id) > 0

# Check whether cell is leaf cell
is_leaf(t::ParallelTree, cell_id::Int) = !has_children(t, cell_id)

# Check whether cell has specific child cell
has_child(t::ParallelTree, cell_id::Int, child::Int) = t.child_ids[child, cell_id] > 0

# Check if cell has a neighbor at the same refinement level in the given direction
has_neighbor(t::ParallelTree, cell_id::Int, direction::Int) = t.neighbor_ids[direction, cell_id] > 0

# Check if cell is own cell, i.e., belongs to this MPI rank
is_own_cell(t::ParallelTree, cell_id) = t.mpi_ranks[cell_id] == mpi_rank()

# Check if cell has a coarse neighbor, i.e., with one refinement level lower
function has_coarse_neighbor(t::ParallelTree, cell_id::Int, direction::Int)
  return has_parent(t, cell_id) && has_neighbor(t, t.parent_ids[cell_id], direction)
end

# Check if cell has any neighbor (same-level or lower-level)
function has_any_neighbor(t::ParallelTree, cell_id::Int, direction::Int)
  return has_neighbor(t, cell_id, direction) || has_coarse_neighbor(t, cell_id, direction)
end

# Return cell length for a given level
length_at_level(t::ParallelTree, level::Int) = t.length_level_0 / 2^level

# Return cell length for a given cell
length_at_cell(t::ParallelTree, cell_id::Int) = length_at_level(t, t.levels[cell_id])

# Return minimum level of any leaf cell
minimum_level(t::ParallelTree) = minimum(t.levels[leaf_cells(t)])

# Return maximum level of any leaf cell
maximum_level(t::ParallelTree) = maximum(t.levels[leaf_cells(t)])

# Check if tree is periodic
isperiodic(t::ParallelTree) = all(t.periodicity)
isperiodic(t::ParallelTree, dimension) = t.periodicity[dimension]


# Auxiliary methods for often-required calculations
# Number of potential child cells
n_children_per_cell(::ParallelTree{NDIMS}) where NDIMS = 2^NDIMS
# n_children_per_cell(dims::Integer) = 2^dims

# Number of directions
#
# Directions are indicated by numbers from 1 to 2*ndims:
# 1 -> -x
# 2 -> +x
# 3 -> -y
# 4 -> +y
# 5 -> -z
# 6 -> +z
n_directions(::ParallelTree{NDIMS}) where NDIMS = 2 * NDIMS

# For a given direction, return its opposite direction
#
# dir -> opp
#  1  ->  2
#  2  ->  1
#  3  ->  4
#  4  ->  3
#  5  ->  6
#  6  ->  5
# opposite_direction(direction::Int) = direction + 1 - 2 * ((direction + 1) % 2)

# For a given child position (from 1 to 8) and dimension (from 1 to 3),
# calculate a child cell's position relative to its parent cell.
#
# Essentially calculates the following
#         dim=1 dim=2 dim=3
# child     x     y     z
#   1       -     -     -
#   2       +     -     -
#   3       -     +     -
#   4       +     +     -
#   5       -     -     +
#   6       +     -     +
#   7       -     +     +
#   8       +     +     +
# child_sign(child::Int, dim::Int) = 1 - 2 * (div(child + 2^(dim - 1) - 1, 2^(dim-1)) % 2)


# For each child position (1 to 8) and a given direction (from 1 to 6), return
# neighboring child position.
# adjacent_child(child::Int, direction::Int) = [2 2 3 3 5 5;
#                                               1 1 4 4 6 6;
#                                               4 4 1 1 7 7;
#                                               3 3 2 2 8 8;
#                                               6 6 7 7 1 1;
#                                               5 5 8 8 2 2;
#                                               8 8 5 5 3 3;
#                                               7 7 6 6 4 4][child, direction]


# For each child position (1 to 8) and a given direction (from 1 to 6), return
# if neighbor is a sibling
# function has_sibling(child::Int, direction::Int)
#   return (child_sign(child, div(direction + 1, 2)) * (-1)^(direction - 1)) > 0
# end


# Obtain leaf cells that fulfill a given criterion.
#
# The function `f` is passed the cell id of each leaf cell
# as an argument.
function filter_leaf_cells(f, t::ParallelTree)
  filtered = Vector{Int}(undef, length(t))
  count = 0
  for cell_id in 1:length(t)
    if is_leaf(t, cell_id) && f(cell_id)
      count += 1
      filtered[count] = cell_id
    end
  end

  return filtered[1:count]
end


# Return an array with the ids of all leaf cells
leaf_cells(t::ParallelTree) = filter_leaf_cells((cell_id)->true, t)


# Return an array with the ids of all leaf cells for a given rank
leaf_cells_by_rank(t::ParallelTree, rank) = filter_leaf_cells(t) do cell_id
                                              t.mpi_ranks[cell_id] == rank
                                            end

# Return an array with the ids of all local leaf cells
local_leaf_cells(t::ParallelTree) = leaf_cells_by_rank(t, mpi_rank())


# Count the number of leaf cells.
count_leaf_cells(t::ParallelTree) = length(leaf_cells(t))


# Store cell id in each cell to use for post-AMR analysis
function reset_original_cell_ids!(t::ParallelTree)
  t.original_cell_ids[1:length(t)] .= 1:length(t)
end


# Refine entire tree by one level
refine!(t::ParallelTree) = refine!(t, leaf_cells(t))


# Refine given cells and rebalance tree.
#
# Note 1: Rebalancing is iterative, i.e., neighboring cells are refined if
#         otherwise the 2:1 rule would be violated, which can cause more
#         refinements.
# Note 2: Rebalancing currently only considers *Cartesian* neighbors, not diagonal neighbors!
function refine!(t::ParallelTree, cell_ids)
  # Reset original cell ids such that each cell knows its current id
  reset_original_cell_ids!(t)

  # Refine all requested cells
  refined = refine_unbalanced!(t, cell_ids)
  refinement_count = length(refined)

  # Iteratively rebalance the tree until it does not change anymore
  while length(refined) > 0
    refined = rebalance!(t, refined)
    refinement_count += length(refined)
  end

  # Determine list of *original* cell ids that were refined
  # Note: original_cell_ids contains the cell_id *before* refinement. At
  # refinement, the refined cell's original_cell_ids value has its sign flipped
  # to easily find it now.
  @views refined_original_cells = (
      -t.original_cell_ids[1:length(t)][t.original_cell_ids[1:length(t)] .< 0])

  # Check if count of refinement cells matches information in original_cell_ids
  @assert refinement_count == length(refined_original_cells) (
      "Mismatch in number of refined cells")

  return refined_original_cells
end


# Refine all leaf cells with coordinates in a given rectangular box
function refine_box!(t::ParallelTree{NDIMS}, coordinates_min::AbstractArray{Float64},
                     coordinates_max::AbstractArray{Float64}) where NDIMS
  for dim in 1:NDIMS
    @assert coordinates_min[dim] < coordinates_max[dim] "Minimum coordinates are not minimum."
  end

  # Find all leaf cells within box
  cells = filter_leaf_cells(t) do cell_id
    return (all(coordinates_min .< t.coordinates[:, cell_id]) &&
            all(coordinates_max .> t.coordinates[:, cell_id]))
  end

  # Refine cells
  refine!(t, cells)
end

# Convenience method for 1D
function refine_box!(t::ParallelTree{1}, coordinates_min::Real, coordinates_max::Real)
  return refine_box!(t, [convert(Float64, coordinates_min)], [convert(Float64, coordinates_max)])
end


# For the given cell ids, check if neighbors need to be refined to restore a rebalanced tree.
#
# Note 1: Rebalancing currently only considers *Cartesian* neighbors, not diagonal neighbors!
# Note 2: The current algorithm assumes that a previous refinement step has
#         created level differences of at most 2. That is, before the previous
#         refinement step, the tree was balanced.
function rebalance!(t::ParallelTree, refined_cell_ids)
  # Create buffer for newly refined cells
  to_refine = zeros(Int, n_directions(t) * length(refined_cell_ids))
  count = 0

  # Iterate over cell ids that have previously been refined
  for cell_id in refined_cell_ids
    # Go over all potential neighbors of child cell
    for direction in 1:n_directions(t)
      # Continue if refined cell has a neighbor in that direction
      if has_neighbor(t, cell_id, direction)
        continue
      end

      # Continue if refined cell has no coarse neighbor, since that would
      # mean it there is no neighbor in that direction at all (domain
      # boundary)
      if !has_coarse_neighbor(t, cell_id, direction)
        continue
      end

      # Otherwise, the coarse neighbor exists and is not refined, thus it must
      # be marked for refinement
      coarse_neighbor_id = t.neighbor_ids[direction, t.parent_ids[cell_id]]
      count += 1
      to_refine[count] = coarse_neighbor_id
    end
  end

  # Finally, refine all marked cells...
  refined = refine_unbalanced!(t, unique(to_refine[1:count]))

  # ...and return list of refined cells
  return refined
end


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
      for direction in 1:n_directions(t)
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

# Wrap single-cell refinements such that `sort(...)` does not complain
refine_unbalanced!(t::ParallelTree, cell_id::Int) = refine_unbalanced!(t, [cell_id])


# Coarsen entire tree by one level
function coarsen!(t::ParallelTree)
  # Special case: if there is only one cell (root), there is nothing to do
  if length(t) == 1
    return Int[]
  end

  # Get list of unique parent ids for all leaf cells
  parent_ids = unique(t.parent_ids[leaf_cells(t)])
  coarsen!(t, parent_ids)
end


# Coarsen given *parent* cells (= these cells must have children who are all
# leaf cells) while retaining a balanced tree.
#
# A cell to be coarsened might cause an unbalanced tree if the neighboring cell
# was already refined. Since it is generally not desired that cells are
# coarsened without specifically asking for it, these cells will then *not* be
# coarsened.
function coarsen!(t::ParallelTree, cell_ids::AbstractArray{Int})
  # Return early if array is empty
  if length(cell_ids) == 0
    return Int[]
  end

  # Reset original cell ids such that each cell knows its current id
  reset_original_cell_ids!(t)

  # To maximize the number of cells that may be coarsened, start with the cells at the highest level
  sorted_by_level = sort(cell_ids, by = i -> t.levels[i])

  # Keep track of number of cells that were actually coarsened
  n_coarsened = 0

  # Local function to adjust cell ids after some cells have been removed
  function adjust_cell_ids!(cell_ids, coarsened_cell_id, count)
    for (id, cell_id) in enumerate(cell_ids)
      if cell_id > coarsened_cell_id
        cell_ids[id] = cell_id - count
      end
    end
  end

  # Iterate backwards over cells to coarsen
  while true
    # Retrieve next cell or quit
    if length(sorted_by_level) > 0
      coarse_cell_id = pop!(sorted_by_level)
    else
      break
    end

    # Ensure that cell has children (violation is an error)
    if !has_children(t, coarse_cell_id)
      error("cell is leaf and cannot be coarsened to: $coarse_cell_id")
    end

    # Ensure that all child cells are leaf cells (violation is an error)
    for child in 1:n_children_per_cell(t)
      if has_child(t, coarse_cell_id, child)
        if !is_leaf(t, t.child_ids[child, coarse_cell_id])
          error("cell $coarse_cell_id has child cell at position $child that is not a leaf cell")
        end
      end
    end

    # Check if coarse cell has refined neighbors that would prevent coarsening
    skip = false
    # Iterate over all children (which are to be removed)
    for child in 1:n_children_per_cell(t)
      # Continue if child does not exist
      if !has_child(t, coarse_cell_id, child)
        continue
      end
      child_id = t.child_ids[child, coarse_cell_id]

      # Go over all neighbors of child cell. If it has a neighbor that is *not*
      # a sibling and that is not a leaf cell, we cannot coarsen its parent
      # without creating an unbalanced tree.
      for direction in 1:n_directions(t)
        # Continue if neighbor would be a sibling
        if has_sibling(child, direction)
          continue
        end

        # Continue if child cell has no neighbor in that direction
        if !has_neighbor(t, child_id, direction)
          continue
        end
        neighbor_id = t.neighbor_ids[direction, child_id]

        if !has_children(t, neighbor_id)
          continue
        end

        # If neighbor is not a sibling, is existing, and has children, do not coarsen
        skip = true
        break
      end
    end
    # Skip if a neighboring cell prevents coarsening
    if skip
      continue
    end

    # Flip sign of cell to be coarsened to such that we can easily find it
    t.original_cell_ids[coarse_cell_id] = -t.original_cell_ids[coarse_cell_id]

    # If a coarse cell has children that are all leaf cells, they must follow
    # immediately due to depth-first ordering of the tree
    count = n_children(t, coarse_cell_id)
    @assert count == n_children_per_cell(t) "cell $coarse_cell_id does not have all child cells"
    remove_shift!(t, coarse_cell_id + 1, coarse_cell_id + count)

    # Take into account shifts in tree that alters cell ids
    adjust_cell_ids!(sorted_by_level, coarse_cell_id, count)

    # Keep track of number of coarsened cells
    n_coarsened += 1
  end

  # Determine list of *original* cell ids that were coarsened to
  # Note: original_cell_ids contains the cell_id *before* coarsening. At
  # coarsening, the coarsened parent cell's original_cell_ids value has its sign flipped
  # to easily find it now.
  @views coarsened_original_cells = (
      -t.original_cell_ids[1:length(t)][t.original_cell_ids[1:length(t)] .< 0])

  # Check if count of coarsened cells matches information in original_cell_ids
  @assert n_coarsened == length(coarsened_original_cells) (
      "Mismatch in number of coarsened cells")

  return coarsened_original_cells
end

# Wrap single-cell coarsening such that `sort(...)` does not complain
coarsen!(t::ParallelTree, cell_id::Int) = coarsen!(t::ParallelTree, [cell_id])


# Coarsen all viable parent cells with coordinates in a given rectangular box
function coarsen_box!(t::ParallelTree{NDIMS}, coordinates_min::AbstractArray{Float64},
                     coordinates_max::AbstractArray{Float64}) where NDIMS
  for dim in 1:NDIMS
    @assert coordinates_min[dim] < coordinates_max[dim] "Minimum coordinates are not minimum."
  end

  # Find all leaf cells within box
  leaves = filter_leaf_cells(t) do cell_id
    return (all(coordinates_min .< t.coordinates[:, cell_id]) &&
            all(coordinates_max .> t.coordinates[:, cell_id]))
  end

  # Get list of unique parent ids for all leaf cells
  parent_ids = unique(t.parent_ids[leaves])

  # Filter parent ids to be within box
  parents = filter(parent_ids) do cell_id
    return (all(coordinates_min .< t.coordinates[:, cell_id]) &&
            all(coordinates_max .> t.coordinates[:, cell_id]))
  end

  # Coarsen cells
  coarsen!(t, parents)
end

# Convenience method for 1D
function coarsen_box!(t::ParallelTree{1}, coordinates_min::Real, coordinates_max::Real)
  return coarsen_box!(t, [convert(Float64, coordinates_min)], [convert(Float64, coordinates_max)])
end


# Return coordinates of a child cell based on its relative position to the parent.
function child_coordinates(::ParallelTree{NDIMS}, parent_coordinates, parent_length::Number, child::Int) where NDIMS
  # Calculate length of child cells and set up data structure
  child_length = parent_length / 2
  coordinates = MVector{NDIMS, Float64}(undef)

  # For each dimension, calculate coordinate as parent coordinate + relative position x length/2
  for d in 1:NDIMS
    coordinates[d] = parent_coordinates[d] + child_sign(child, d) * child_length / 2
  end

  return coordinates
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
invalidate!(t::ParallelTree, id::Int) = invalidate!(t, id, id)
invalidate!(t::ParallelTree) = invalidate!(t, 1, length(t))


# Delete connectivity with parents/children/neighbors before cells are erased
function delete_connectivity!(t::ParallelTree, first::Int, last::Int)
  @assert first > 0
  @assert first <= last
  @assert last <= t.capacity + 1

  # Iterate over all cells
  for cell_id in first:last
    # Delete connectivity from parent cell
    if has_parent(t, cell_id)
      parent_id = t.parent_ids[cell_id]
      for child in 1:n_children_per_cell(t)
        if t.child_ids[child, parent_id] == cell_id
          t.child_ids[child, parent_id] = 0
          break
        end
      end
    end

    # Delete connectivity from child cells
    for child in 1:n_children_per_cell(t)
      if has_child(t, cell_id, child)
        t.parent_ids[t._child_ids[child, cell_id]] = 0
      end
    end

    # Delete connectivity from neighboring cells
    for direction in 1:n_directions(t)
      if has_neighbor(t, cell_id, direction)
        t.neighbor_ids[opposite_direction(direction), t.neighbor_ids[direction, cell_id]] = 0
      end
    end
  end
end


# Move connectivity with parents/children/neighbors after cells have been moved
function move_connectivity!(t::ParallelTree, first::Int, last::Int, destination::Int)
  @assert first > 0
  @assert first <= last
  @assert last <= t.capacity + 1
  @assert destination > 0
  @assert destination <= t.capacity + 1

  # Strategy
  # 1) Loop over moved cells (at target location)
  # 2) Check if parent/children/neighbors connections are to a cell that was moved
  #    a) if cell was moved: apply offset to current cell
  #    b) if cell was not moved: go to connected cell and update connectivity there

  offset = destination - first
  has_moved(n) = (first <= n <= last)

  for source in first:last
    target = source + offset

    # Update parent
    if has_parent(t, target)
      # Get parent cell
      parent_id = t.parent_ids[target]
      if has_moved(parent_id)
        # If parent itself was moved, just update parent id accordingly
        t.parent_ids[target] += offset
      else
        # If parent was not moved, update its corresponding child id
        for child in 1:n_children_per_cell(t)
          if t.child_ids[child, parent_id] == source
            t.child_ids[child, parent_id] = target
          end
        end
      end
    end

    # Update children
    for child in 1:n_children_per_cell(t)
      if has_child(t, target, child)
        # Get child cell
        child_id = t.child_ids[child, target]
        if has_moved(child_id)
          # If child itself was moved, just update child id accordingly
          t.child_ids[child, target] += offset
        else
          # If child was not moved, update its parent id
          t.parent_ids[child_id] = target
        end
      end
    end

    # Update neighbors
    for direction in 1:n_directions(t)
      if has_neighbor(t, target, direction)
        # Get neighbor cell
        neighbor_id = t.neighbor_ids[direction, target]
        if has_moved(neighbor_id)
          # If neighbor itself was moved, just update neighbor id accordingly
          t.neighbor_ids[direction, target] += offset
        else
          # If neighbor was not moved, update its opposing neighbor id
          t.neighbor_ids[opposite_direction(direction), neighbor_id] = target
        end
      end
    end
  end
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
