module Trees

using StaticArrays: MVector, @MVector
import Base

export Tree


abstract type AbstractContainer end


# Composite type that represents a D-dimensional tree.
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
# implemented, we only have fully refined cells. That is, a cell either has 2^D children or
# no children at all (= leaf cell). This restriction is also assumed at
# multiple positions in the refinement/coarsening algorithms.
#
# An exception to the 2:1 rule exists for the low-level
# `refine_unbalanced!` and `coarsen_unbalanced!` functions, which is required
# for implementing level-wise refinement/coarsening in a sane way. Also,
# depth-first ordering *might* not by guaranteed during refinement/coarsening
# operations.
mutable struct Tree{D} <: AbstractContainer
  parent_ids::Vector{Int}
  child_ids::Matrix{Int}
  neighbor_ids::Matrix{Int}
  levels::Vector{Int}
  coordinates::Matrix{Float64}

  capacity::Int
  length::Int
  dummy::Int

  center_level_0::MVector{D, Float64}
  length_level_0::Float64

  function Tree{D}(capacity::Integer) where D
    # Verify that D is an integer
    @assert D isa Integer

    # Create instance
    t = new()

    # Initialize fields with defaults
    # Note: length as capacity + 1 is to use `capacity + 1` as temporary storage for swap operations
    t.parent_ids = fill(typemin(Int), capacity + 1)
    t.child_ids = fill(typemin(Int), 2^D, capacity + 1)
    t.neighbor_ids = fill(typemin(Int), 2*D, capacity + 1)
    t.levels = fill(typemin(Int), capacity + 1)
    t.coordinates = fill(NaN, D, capacity + 1)

    t.capacity = capacity
    t.length = 0
    t.dummy = capacity + 1

    t.center_level_0 = @MVector fill(NaN, D)
    t.length_level_0 = NaN

    return t
  end
end


# Constructor for passing the dimension as an argument
Tree(::Val{D}, args...) where D = Tree{D}(args...)

# Create and initialize tree
function Tree{D}(capacity::Int, center::AbstractArray{Float64}, length::Real) where D
  # Create instance
  t = Tree{D}(capacity)

  # Initialize root cell
  init!(t, center, length)

  return t
end

# Constructor accepting a single number as center (as opposed to an array) for 1D
Tree{1}(cap::Int, center::Real, len::Real) = Tree{1}(cap, [convert(Float64, center)], len)


# Clear tree with deleting data structures, store center and length, and create root cell
function init!(t::Tree, center::AbstractArray{Float64}, length::Real)
  clear!(t)

  # Set domain information
  t.center_level_0 = center
  t.length_level_0 = length

  # Create root cell
  t.length += 1
  t.levels[1] = 0
  t.parent_ids[1] = 0
  t.child_ids[:, 1] .= 0
  t.neighbor_ids[:, 1] .= 1 # Special case: For periodicity, the level-0 cell is its own neighbor
  t.levels[1] = 0
  t.coordinates[:, 1] .= t.center_level_0
end


# Convenience output for debugging
function Base.show(io::IO, t::Tree{D}) where D
  l = t.length
  println(io, '*'^20)
  println(io, "t.parent_ids[1:l] = $(t.parent_ids[1:l])")
  println(io, "transpose(t.child_ids[:, 1:l]) = $(transpose(t.child_ids[:, 1:l]))")
  println(io, "transpose(t.neighbor_ids[:, 1:l]) = $(transpose(t.neighbor_ids[:, 1:l]))")
  println(io, "t.levels[1:l] = $(t.levels[1:l])")
  println(io, "transpose(t.coordinates[:, 1:l]) = $(transpose(t.coordinates[:, 1:l]))")
  println(io, "t.capacity = $(t.capacity)")
  println(io, "t.length = $(t.length)")
  println(io, "t.dummy = $(t.dummy)")
  println(io, "t.center_level_0 = $(t.center_level_0)")
  println(io, "t.length_level_0 = $(t.length_level_0)")
  println(io, '*'^20)
end

# Type traits to obtain dimension
ndims(t::Type{Tree{D}}) where D = D
ndims(t::Tree) = ndims(typeof(t))


# Auxiliary methods to allow semantic queries on the tree
# Check whether cell has parent cell
has_parent(t::Tree, cell_id::Int) = t.parent_ids[cell_id] > 0

# Check whether cell has any child cell
has_child(t::Tree, cell_id::Int, child_id::Int) = t.child_ids[child_id, cell_id] > 0

# Check whether cell is leaf cell
is_leaf(t::Tree, cell_id::Int) = !has_children(t, cell_id)

# Check whether cell has specific child cell
has_children(t::Tree, cell_id::Int) = n_children(t, cell_id) > 0

# Count number of children for a given cell
n_children(t::Tree, cell_id::Int) = count(x -> (x > 0), @view t.child_ids[:, cell_id])

# Check if cell has a neighbor at the same refinement level in the given direction
has_neighbor(t::Tree, cell_id::Int, direction::Int) = t.neighbor_ids[direction, cell_id] > 0

# Check if cell has a coarse neighbor, i.e., with one refinement level lower
function has_coarse_neighbor(t::Tree, cell_id::Int, direction::Int)
  return has_parent(t, cell_id) && has_neighbor(t, t.parent_ids[cell_id], direction)
end

# Check if cell has any neighbor (same-level or lower-level)
function has_any_neighbor(t::Tree, cell_id::Int, direction::Int)
  return has_neighbor(t, cell_id, direction) || has_coarse_neighbor(t, cell_id, direction)
end

# Return cell length for a given level
length_at_level(t::Tree, level::Int) = t.length_level_0 / 2^level

# Return cell length for a given cell
length_at_cell(t::Tree, cell_id::Int) = length_at_level(t, t.levels[cell_id])

# Return minimum level of any leaf cell
minimum_level(t::Tree) = minimum(t.levels[leaf_cells(t)])

# Return maximum level of any leaf cell
maximum_level(t::Tree) = maximum(t.levels[leaf_cells(t)])


# Auxiliary methods for often-required calculations
# Number of potential child cells
n_children_per_cell(::Tree{D}) where D = 2^D

# Number of directions
#
# Directions are indicated by numbers from 1 to 2*ndims:
# 1 -> -x
# 2 -> +x
# 3 -> -y
# 4 -> +y
# 5 -> -z
# 6 -> +z
n_directions(::Tree{D}) where D = 2 * D

# For a given direction, return its opposite direction
#
# dir -> opp
#  1  ->  2
#  2  ->  1
#  3  ->  4
#  4  ->  3
#  5  ->  6
#  6  ->  5
opposite_direction(direction::Int) = direction + 1 - 2 * ((direction + 1) % 2)

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
child_sign(child::Int, dim::Int) = 1 - 2 * (div(child + 2^(dim - 1) - 1, 2^(dim-1)) % 2)


# For each child position (1 to 8) and a given direction (from 1 to 6), return
# neighboring child position.
adjacent_child(child::Int, direction::Int) = [2 2 3 3 5 5;
                                              1 1 4 4 6 6;
                                              4 4 1 1 7 7;
                                              3 3 2 2 8 8;
                                              6 6 7 7 1 1;
                                              5 5 8 8 2 2;
                                              8 8 5 5 3 3;
                                              7 7 6 6 4 4][child, direction]


# For each child position (1 to 8) and a given direction (from 1 to 6), return
# if neighbor is a sibling
function has_sibling(child::Int, direction::Int)
  return (child_sign(child, div(direction + 1, 2)) * (-1)^(direction - 1)) > 0
end


# Obtain leaf cells that fulfill a given criterion.
#
# The function `f` is passed the cell id of each leaf cell
# as an argument.
function filter_leaf_cells(f, t::Tree)
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
leaf_cells(t::Tree) = filter_leaf_cells((cell_id)->true, t)


# Count the number of leaf cells.
count_leaf_cells(t::Tree) = length(leaf_cells(t))


# Refine entire tree by one level
function refine!(t::Tree)
  refine!(t, leaf_cells(t))
end


# Refine given cells and rebalance tree.
#
# Note 1: Rebalancing is iterative, i.e., neighboring cells are refined if
#         otherwise the 2:1 rule would be violated, which can cause more
#         refinements.
# Note 2: Rebalancing currently only considers *Cartesian* neighbors, not diagonal neighbors!
function refine!(t::Tree, cell_ids)
  refined = refine_unbalanced!(t, cell_ids)
  while length(refined) > 0
    refined = rebalance!(t, refined)
  end
end


# Refine all leaf cells with coordinates in a given rectangular box
function refine_box!(t::Tree{D}, coordinates_min::AbstractArray{Float64},
                     coordinates_max::AbstractArray{Float64}) where D
  for dim in 1:D
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
function refine_box!(t::Tree{1}, coordinates_min::Real, coordinates_max::Real)
  return refine_box!(t, [convert(Float64, coordinates_min)], [convert(Float64, coordinates_max)])
end


# For the given cell ids, check if neighbors need to be refined to restore a rebalanced tree.
#
# Note 1: Rebalancing currently only considers *Cartesian* neighbors, not diagonal neighbors!
# Note 2: The current algorithm assumes that a previous refinement step has
#         created level differences of at most 2. That is, before the previous
#         refinement step, the tree was balanced.
function rebalance!(t::Tree, refined_cell_ids)
  # Create buffer for newly refined cells
  to_refine = zeros(Int, n_directions(t) * length(refined_cell_ids))
  count = 0

  # Iterate over cell ids that have previously been refined
  for cell_id in refined_cell_ids
    # Loop over all possible directions
    for direction in 1:n_directions(t)
      # Check if a neighbor exists. If yes, there is nothing else to do, since
      # our current cell is at most one level further refined
      if has_neighbor(t, cell_id, direction)
        continue
      end

      # If also no coarse neighbor exists, there is nothing to do in this direction
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
  refined = refine_unbalanced!(t, @view to_refine[1:count])

  # ...and return list of refined cells
  return refined
end


# Refine given cells without rebalancing tree.
#
# Note: After a call to this method the tree may be unbalanced!
function refine_unbalanced!(t::Tree, cell_ids)
  # Store actual ids refined cells (shifted due to previous insertions)
  refined = zeros(Int, length(cell_ids))

  # Loop over all cells that are to be refined
  for (count, original_cell_id) in enumerate(sort(cell_ids))
    # Determine actual cell id, taking into account previously inserted cells
    n_children = n_children_per_cell(t)
    cell_id = original_cell_id + (count - 1) * n_children
    refined[count] = cell_id

    @assert !has_children(t, cell_id) "Non-leaf cell $cell_id cannot be refined"

    # Insert new cells directly behind parent (depth-first)
    insert!(t, cell_id + 1, n_children)

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
refine_unbalanced!(t::Tree, cell_id::Int) = refine_unbalanced!(t, [cell_id])


# Return coordinates of a child cell based on its relative position to the parent.
function child_coordinates(::Tree{D}, parent_coordinates, parent_length::Number, child::Int) where D
  # Calculate length of child cells and set up data structure
  child_length = parent_length / 2
  coordinates = MVector{D, Float64}(undef)

  # For each dimension, calculate coordinate as parent coordinate + relative position x length/2 
  for d in 1:D
    coordinates[d] = parent_coordinates[d] + child_sign(child, d) * child_length / 2
  end

  return coordinates
end


# Reset range of cells to values that are prone to cause errors as soon as they are used.
#
# Rationale: If an invalid cell is accidentally used, we want to know it as soon as possible.
function invalidate!(t::Tree, first::Int, last::Int)
  @assert first > 0
  @assert last <= t.capacity + 1

  # Integer values are set to smallest negative value, floating point values to NaN
  t.parent_ids[first:last] .= typemin(Int)
  t.child_ids[:, first:last] .= typemin(Int)
  t.neighbor_ids[:, first:last] .= typemin(Int)
  t.levels[first:last] .= typemin(Int)
  t.coordinates[:, first:last] .= NaN
end
invalidate!(t::Tree, id::Int) = invalidate!(t, id, id)
invalidate!(t::Tree) = invalidate!(t, 1, length(t))


# Delete connectivity with parents/children/neighbors before cells are erased
function delete_connectivity!(t::Tree, first::Int, last::Int)
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
function move_connectivity!(t::Tree, first::Int, last::Int, destination::Int)
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
function raw_copy!(target::Tree, source::Tree, first::Int, last::Int, destination::Int)
  copy_data!(target.parent_ids, source.parent_ids, first, last, destination)
  copy_data!(target.child_ids, source.child_ids, first, last, destination,
             n_children_per_cell(target))
  copy_data!(target.neighbor_ids, source.neighbor_ids, first, last,
             destination, n_directions(target))
  copy_data!(target.levels, source.levels, first, last, destination)
  copy_data!(target.coordinates, source.coordinates, first, last, destination, ndims(target))
end
function raw_copy!(c::AbstractContainer, first::Int, last::Int, destination::Int)
  raw_copy!(c, c, first, last, destination)
end
function raw_copy!(target::AbstractContainer, source::AbstractContainer,
                   from::Int, destination::Int)
  raw_copy!(target, source, from, from, destination)
end
function raw_copy!(c::AbstractContainer, from::Int, destination::Int)
  raw_copy!(c, c, from, from, destination)
end


# Reset data structures by recreating all internal storage containers and invalidating all elements
function reset_data_structures!(t::Tree{D}) where D
  t.parent_ids = Vector{Int}(undef, t.capacity + 1)
  t.child_ids = Matrix{Int}(undef, 2^D, t.capacity + 1)
  t.neighbor_ids = Matrix{Int}(undef, 2*D, t.capacity + 1)
  t.levels = Vector{Int}(undef, t.capacity + 1)
  t.coordinates = Matrix{Float64}(undef, D, t.capacity + 1)

  invalidate!(t, 1, capacity(t) + 1)
end


####################################################################################################
# Here follows the implementation for a generic container
####################################################################################################


# Auxiliary copy function to copy data between containers
function copy_data!(target::AbstractArray{T, N}, source::AbstractArray{T, N},
                    first::Int, last::Int, destination::Int, block_size::Int=1) where {T, N}

  count = last - first + 1
  if destination <= first || destination > last
    # In this case it is safe to copy forward (left-to-right) without overwriting data
    for i in 0:(count-1), j in 1:block_size
      target[block_size*(destination+i-1) + j] = source[block_size*(first+i-1) + j]
    end
  else
    # In this case we need to copy backward (right-to-left) to prevent overwriting data
    for i in reverse(0:(count-1)), j in 1:block_size
      target[block_size*(destination+i-1) + j] = source[block_size*(first+i-1) + j]
    end
  end
end


# Inquire about capacity and size
capacity(c::AbstractContainer) = c.capacity
Base.length(c::AbstractContainer) = c.length
size(c::AbstractContainer) = (length(c),)


# Increase container length by `count` elements
function append!(c::AbstractContainer, count::Int)
  @assert count >= 0 "Count must be non-negative"
  @assert count + length(c) <= capacity(c) "New length would exceed capacity"

  # First, invalidate range (to be sure that no sensible values are accidentally left there)
  invalidate!(c, length(c) + 1, length(c) + count)

  # Then, increase container length
  c.length += count
end
append!(c::AbstractContainer) = append(c, 1)


# Decrease container length by `count` elements
function shrink!(c::AbstractContainer, count::Int)
  @assert count >= 0
  @assert length(c) >= count

  # Rely on remove&shift to do The Right Thing
  remove_shift!(c, length(c) - count + 1, length())
end
shrink!(c::AbstractContainer) = shrink(c, 1)


# Copy data range from source to target container.
#
# Calls `raw_copy` internally, which must be implemented for each concrete type
# inheriting from AbstractContainer.
function copy!(target::AbstractContainer, source::AbstractContainer,
               first::Int, last::Int, destination::Int)
  @assert 1 <= first <= length(source) "First cell out of range"
  @assert 1 <= last <= length(source) "Last cell out of range"
  @assert 1 <= destination <= length(target) "Destination out of range"
  @assert destination + (last - first) <= length(target) "Target range out of bounds"

  # Return if copy would be a no-op
  if last < first || (source === target && first == destination)
    return
  end

  raw_copy!(target, source, first, last, destination)
end


# Convenience method to copy a single element
function copy!(target::AbstractContainer, source::AbstractContainer, from::Int, destination::Int)
  copy!(target, source, from, from, destination)
end


# Convenience method for copies within a single container
function copy!(c::AbstractContainer, first::Int, last::Int, destination::Int)
  copy!(c, c, first, last, destination)
end


# Convenience method for copying a single element within a single container
function copy!(c::AbstractContainer, from::Int, destination::Int)
  copy!(c, c, from, from, destination)
end


# Move elements in a way that preserves connectivity.
function move!(c::AbstractContainer, first::Int, last::Int, destination::Int)
  @assert 1 <= first <= length(c) "First cell $first out of range"
  @assert 1 <= last <= length(c) "Last cell $last out of range"
  @assert 1 <= destination <= length(c) "Destination $destination out of range"
  @assert destination + (last - first) <= length(c) "Target range out of bounds"

  # Return if move would be a no-op
  if last < first || first == destination
    return
  end

  # Copy cells to new location
  raw_copy!(c, first, last, destination)

  # Move connectivity
  move_connectivity!(c, first, last, destination)


  # Invalidate original cell locations (unless they already contain new data due to overlap)
  # 1) If end of desination range is within original range, shift first_invalid to the right
  count = last - first + 1
  first_invalid = (first <= destination + count - 1 <= last) ? destination + count : first
  # 2) If beginning of destination range is within original range, shift last_invalid to the left
  last_invalid = (first <= destination <= last) ? destination - 1 : last
  # 3) Invalidate range
  invalidate!(c, first_invalid, last_invalid)
end
move!(c::AbstractContainer, from::Int, destination::Int) = move!(c, from, from, destination)


# Swap two elements in a container while preserving element connectivity.
function swap!(c::AbstractContainer, a::Int, b::Int)
  @assert 1 <= a <= length(c) "a out of range"
  @assert 1 <= b <= length(c) "b out of range"

  # Return if swap would be a no-op
  if a == b
    return
  end

  # Move a to dummy location
  raw_copy!(c, a, c.dummy)
  move_connectivity(c, a, c.dummy)

  # Move b to a
  raw_copy!(c, b, a)
  move_connectivity(c, b, a)

  # Move from dummy location to b
  raw_copy!(c, c.dummy, b)
  move_connectivity(c, c.dummy, b)

  # Invalidate dummy to be sure
  invalidate(c, c.dummy)
end


# Insert blank elements in container, shifting the following elements back.
#
# After a call to insert!, the range `position:position + count - 1` will be available for use.
function insert!(c::AbstractContainer, position::Int, count::Int)
  @assert 1 <= position <= length(c) + 1 "Insert position out of range"
  @assert count >= 0 "Count must be non-negative"
  @assert count + length(c) <= capacity(c) "New length would exceed capacity"

  # Return if insertation would be a no-op
  if count == 0
    return
  end

  # Append and return if insertion is beyond last current element
  if position == length(c) + 1
    append!(c, count)
    return
  end

  # Increase length
  c.length += count

  # Move original cells that currently occupy the insertion region, unless
  # insert position is one beyond previous length
  if position <= length(c) - count
    move!(c, position, length(c) - count, position + count)
  end
end
insert!(c) = insert!(c, position, 1)


# Erase elements from container, deleting their connectivity and then invalidating their data.
function erase!(c::AbstractContainer, first::Int, last::Int)
  @assert 1 <= first <= length(c) "First cell out of range"
  @assert 1 <= last <= length(c) "Last cell out of range"

  # Return if eraseure would be a no-op
  if last < first
    return
  end

  # Delete connectivity and invalidate cells
  delete_connectivity!(c, first, last)
  invalidate!(c, first, last)
end
erase!(c::AbstractContainer, id::Int) = erase!(c, id, id)


# Remove cells and shift existing cells forward to close the gap
function remove_shift!(c::AbstractContainer, first::Int, last::Int)
  @assert 1 <= first <= length(c) "First cell out of range"
  @assert 1 <= last <= length(c) "Last cell out of range"

  # Return if removal would be a no-op
  if last < first
    return
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
end
remove_shift!(c::AbstractContainer, id::Int) = remove_shift!(c, id, id)


# Remove cells and fill gap with cells from the end of the container (to reduce copy operations)
function remove_fill!(c::AbstractContainer, first::Int, last::Int)
  @assert 1 <= first <= length(c) "First cell out of range"
  @assert 1 <= last <= length(c) "Last cell out of range"

  # Return if removal would be a no-op
  if last < first
    return
  end

  # Delete connectivity of cells to be removed and then invalidate them
  delete_connectivity!(c, first, last)
  invalidate!(c, first, last)

  # Copy cells from end (unless last is already the last cell)
  count = last - first + 1
  if last < length(c)
    move(c, max(length(c) - count, last + 1), length(c), first)
  end

  # Reduce length
  c.length -= count
end


# Reset container to zero-length and with a new capacity
function reset!(c::AbstractContainer, capacity::Int)
  @assert capacity >=0

  c.capacity = capacity
  c.length = 0
  c.dummy = capacity + 1
  reset_data_structures!(c::AbstractContainer)
end


# Invalidate all elements and set length to zero.
function clear!(c::AbstractContainer)
  invalidate!(c)
  c.length = 0
end

end
