module Trees

using StaticArrays: MVector
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
#                                 neighboring nodes, or 2:1 rule)
#   b) we may not have all children (= some children may not exist)
#   c) the tree is stored depth-first
#
# However, the way the refinement/coarsening algorithms are currently
# implemented, we only have fully refined nodes. That is, a node either has 2^D children or
# no children at all (= leaf node). This restriction is also assumed at
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
  size::Int
  dummy::Int

  center_level_0::MVector{D, Float64}
  length_level_0::Float64

  function Tree{D}(capacity::Int, center::AbstractArray{Float64}, length::Real) where D
    # Verify that D is an integer
    @assert D isa Integer

    # Create instance
    b = new()

    # Initialize fields with defaults
    # Note: size as capacity + 1 is to use `capacity + 1` as temporary storage for swap operations
    b.parent_ids = fill(typemin(Int), capacity + 1)
    b.child_ids = fill(typemin(Int), 2^D, capacity + 1)
    b.neighbor_ids = fill(typemin(Int), 2*D, capacity + 1)
    b.levels = fill(typemin(Int), capacity + 1)
    b.coordinates = fill(NaN, D, capacity + 1)

    b.capacity = capacity
    b.size = 0
    b.dummy = capacity + 1

    b.center_level_0 = center
    b.length_level_0 = length

    # Create initial node
    b.size += 1
    b.levels[1] = 0
    b.parent_ids[1] = 0
    b.child_ids[:, 1] .= 0
    b.neighbor_ids[:, 1] .= 1 # Special case: For periodicity, the level-0 node is its own neighbor
    b.levels[1] = 0
    b.coordinates[:, 1] .= b.center_level_0

    return b
  end
end

# Constructor for passing the dimension as an argument
Tree(::Val{D}, args...) where D = Tree{D}(args...)

# Constructor accepting a single number as center (as opposed to an array) for 1D
Tree(::Val{1}, cap::Int, center::Real, len::Real) = Tree{1}(cap, [convert(Float64, center)], len)

# Convenience output for debugging
function Base.show(io::IO, t::Tree{D}) where D
  s = t.size
  println(io, '*'^20)
  println(io, "t.parent_ids[1:s] = $(t.parent_ids[1:s])")
  println(io, "transpose(t.child_ids[:, 1:s]) = $(transpose(t.child_ids[:, 1:s]))")
  println(io, "transpose(t.neighbor_ids[:, 1:s]) = $(transpose(t.neighbor_ids[:, 1:s]))")
  println(io, "t.levels[1:s] = $(t.levels[1:s])")
  println(io, "transpose(t.coordinates[:, 1:s]) = $(transpose(t.coordinates[:, 1:s]))")
  println(io, "t.capacity = $(t.capacity)")
  println(io, "t.size = $(t.size)")
  println(io, "t.dummy = $(t.dummy)")
  println(io, "t.center_level_0 = $(t.center_level_0)")
  println(io, "t.length_level_0 = $(t.length_level_0)")
  println(io, '*'^20)
end

# Type traits to obtain dimension
ndims(t::Type{Tree{D}}) where D = D
ndims(t::Tree) = ndims(typeof(t))


# Auxiliary methods to allow semantic queries on the tree
# Check whether node has parent node
has_parent(t::Tree, node_id::Int) = t.parent_ids[node_id] > 0

# Check whether node has any child node
has_child(t::Tree, node_id::Int, child_id::Int) = t.child_ids[child_id, node_id] > 0

# Check whether node is leaf node
is_leaf(t::Tree, node_id::Int) = !has_children(t, node_id)

# Check whether node has specific child node
has_children(t::Tree, node_id::Int) = n_children(t, node_id) > 0

# Count number of children for a given node
n_children(t::Tree, node_id::Int) = count(x -> (x > 0), @view t.child_ids[:, node_id])

# Check if node has a neighbor at the same refinement level in the given direction
has_neighbor(t::Tree, node_id::Int, direction::Int) = t.neighbor_ids[direction, node_id] > 0

# Check if node has a coarse neighbor, i.e., with one refinement level lower
function has_coarse_neighbor(t::Tree, node_id::Int, direction::Int)
  return has_parent(t, node_id) && has_neighbor(t, t.parent_ids[node_id], direction)
end

# Check if node has any neighbor (same-level or lower-level)
function has_any_neighbor(t::Tree, node_id::Int, direction::Int)
  return has_neighbor(t, node_id, direction) || has_coarse_neighbor(t, node_id, direction)
end

# Return node length for a given level
length_at_level(t::Tree, level::Int) = t.length_level_0 / 2^level

# Return node length for a given node
length_at_node(t::Tree, node_id::Int) = length_at_level(t, t.levels[node_id])

# Return minimum level of any leaf cell
minimum_level(t::Tree) = minimum(t.levels[leaf_nodes(t)])

# Return maximum level of any leaf cell
maximum_level(t::Tree) = maximum(t.levels[leaf_nodes(t)])


# Auxiliary methods for often-required calculations
# Number of potential child nodes
n_children_per_node(::Tree{D}) where D = 2^D

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
# calculate a child node's position relative to its parent node.
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


# Obtain leaf nodes that fulfill a given criterion.
#
# The function `f` is passed the node id of each leaf node
# as an argument.
function filter_leaf_nodes(f, t::Tree)
  filtered = Vector{Int}(undef, size(t))
  count = 0
  for node_id in 1:size(t)
    if is_leaf(t, node_id) && f(node_id)
      count += 1
      filtered[count] = node_id
    end
  end

  return filtered[1:count]
end


# Return an array with the ids of all leaf nodes
leaf_nodes(t::Tree) = filter_leaf_nodes((node_id)->true, t)


# Count the number of leaf nodes.
count_leaf_nodes(t::Tree) = length(leaf_nodes(t))


# Refine entire tree by one level
function refine!(t::Tree)
  refine!(t, leaf_nodes(t))
end


# Refine given nodes and rebalance tree.
#
# Note 1: Rebalancing is iterative, i.e., neighboring nodes are refined if
#         otherwise the 2:1 rule would be violated, which can cause more
#         refinements.
# Note 2: Rebalancing currently only considers *Cartesian* neighbors, not diagonal neighbors!
function refine!(t::Tree, node_ids)
  refined = refine_unbalanced!(t, node_ids)
  while length(refined) > 0
    refined = rebalance!(t, refined)
  end
end


# Refine all leaf nodes with coordinates in a given rectangular box
function refine_box!(t::Tree{D}, coordinates_min::AbstractArray{Float64},
                     coordinates_max::AbstractArray{Float64}) where D
  for dim in 1:D
    @assert coordinates_min[dim] < coordinates_max[dim] "Minimum coordinates are not minimum."
  end

  # Find all leaf nodes within box
  nodes = filter_leaf_nodes(t) do node_id
    return (all(coordinates_min .< t.coordinates[:, node_id]) &&
            all(coordinates_max .> t.coordinates[:, node_id]))
  end

  # Refine nodes
  refine!(t, nodes)
end

# Convenience method for 1D
function refine_box!(t::Tree{1}, coordinates_min::Real, coordinates_max::Real)
  return refine_box!(t, [convert(Float64, coordinates_min)], [convert(Float64, coordinates_max)])
end


# For the given node ids, check if neighbors need to be refined to restore a rebalanced tree.
#
# Note 1: Rebalancing currently only considers *Cartesian* neighbors, not diagonal neighbors!
# Note 2: The current algorithm assumes that a previous refinement step has
#         created level differences of at most 2. That is, before the previous
#         refinement step, the tree was balanced.
function rebalance!(t::Tree, refined_node_ids)
  # Create buffer for newly refined nodes
  to_refine = zeros(Int, n_directions(t) * length(refined_node_ids))
  count = 0

  # Iterate over node ids that have previously been refined
  for node_id in refined_node_ids
    # Loop over all possible directions
    for direction in 1:n_directions(t)
      # Check if a neighbor exists. If yes, there is nothing else to do, since
      # our current node is at most one level further refined
      if has_neighbor(t, node_id, direction)
        continue
      end

      # If also no coarse neighbor exists, there is nothing to do in this direction
      if !has_coarse_neighbor(t, node_id, direction)
        continue
      end

      # Otherwise, the coarse neighbor exists and is not refined, thus it must
      # be marked for refinement
      coarse_neighbor_id = t.neighbor_ids[direction, t.parent_ids[node_id]]
      count += 1
      to_refine[count] = coarse_neighbor_id
    end
  end

  # Finally, refine all marked nodes...
  refined = refine_unbalanced!(t, @view to_refine[1:count])

  # ...and return list of refined nodes
  return refined
end


# Refine given nodes without rebalancing tree.
#
# Note: After a call to this method the tree may be unbalanced!
function refine_unbalanced!(t::Tree, node_ids)
  # Store actual ids refined nodes (shifted due to previous insertions)
  refined = zeros(Int, length(node_ids))

  # Loop over all nodes that are to be refined
  for (count, original_node_id) in enumerate(sort(node_ids))
    # Determine actual node id, taking into account previously inserted nodes
    n_children = n_children_per_node(t)
    node_id = original_node_id + (count - 1) * n_children
    refined[count] = node_id

    @assert !has_children(t, node_id) "Non-leaf node $node_id cannot be refined"

    # Insert new nodes directly behind parent (depth-first)
    insert!(t, node_id + 1, n_children)

    # Initialize child nodes
    for child in 1:n_children
      # Set child information based on parent
      child_id = node_id + child
      t.parent_ids[child_id] = node_id
      t.child_ids[child, node_id] = child_id
      t.neighbor_ids[:, child_id] .= 0
      t.child_ids[:, child_id] .= 0
      t.levels[child_id] = t.levels[node_id] + 1
      t.coordinates[:, child_id] .= child_coordinates(
          t, t.coordinates[:, node_id], length_at_node(t, node_id), child)

      # For determining neighbors, use neighbor connections of parent node
      for direction in 1:n_directions(t)
        # If neighbor is a sibling, establish one-sided connectivity
        # Note: two-sided is not necessary, as each sibling will do this
        if has_sibling(child, direction)
          adjacent = adjacent_child(child, direction)
          neighbor_id = node_id + adjacent

          t.neighbor_ids[direction, child_id] = neighbor_id
          continue
        end

        # Skip if original node does have no neighbor in direction
        if !has_neighbor(t, node_id, direction)
          continue
        end

        # Otherwise, check if neighbor has children - if not, skip again
        neighbor_id = t.neighbor_ids[direction, node_id]
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

# Wrap single-node refinements such that `sort(...)` does not complain
refine_unbalanced!(t::Tree, node_id::Int) = refine_unbalanced!(t, [node_id])


# Return coordinates of a child node based on its relative position to the parent.
function child_coordinates(::Tree{D}, parent_coordinates, parent_length::Number, child::Int) where D
  # Calculate length of child nodes and set up data structure
  child_length = parent_length / 2
  child_coordinates = MVector{D, Float64}(undef)

  # For each dimension, calculate coordinate as parent coordinate + relative position x length/2 
  for d in 1:D
    child_coordinates[d] = parent_coordinates[d] + child_sign(child, d) * child_length / 2
  end

  return child_coordinates
end


# Reset range of nodes to values that are prone to cause errors as soon as they are used.
#
# Rationale: If an invalid node is accidentally used, we want to know it as soon as possible.
function invalidate!(t::Tree, first::Int, last::Int)
  @assert first > 0
  @assert first <= last
  @assert last <= t.capacity + 1

  # Integer values are set to smallest negative value, floating point values to NaN
  t.parent_ids[first:last] .= typemin(Int)
  t.child_ids[:, first:last] .= typemin(Int)
  t.neighbor_ids[:, first:last] .= typemin(Int)
  t.levels[first:last] .= typemin(Int)
  t.coordinates[:, first:last] .= NaN
end
invalidate!(t::Tree, id::Int) = invalidate!(t, id, id)
invalidate!(t::Tree) = invalidate!(t, 1, size(t))


# Delete connectivity with parents/children/neighbors before nodes are erased
function delete_connectivity!(t::Tree, first::Int, last::Int)
  @assert first > 0
  @assert first <= last
  @assert last <= t.capacity + 1

  # Iterate over all nodes
  for node_id in first:last
    # Delete connectivity from parent node
    if has_parent(t, node_id)
      parent_id = t.parent_ids[node_id]
      for child in 1:n_children_per_node(t)
        if t.child_ids[child, parent_id] == node_id
          t.child_ids[child, parent_id] = 0
          break
        end
      end
    end

    # Delete connectivity from child nodes
    for child in 1:n_children_per_node(t)
      if has_child(t, node_id, child)
        t.parent_ids[t._child_ids[child, node_id]] = 0
      end
    end

    # Delete connectivity from neighboring nodes
    for direction in 1:n_directions(t)
      if has_neighbor(t, node_id, direction)
        t.neighbor_ids[opposite_direction(direction), t.neighbor_ids[direction, node_id]] = 0
      end
    end
  end
end


# Move connectivity with parents/children/neighbors after nodes have been moved
function move_connectivity!(t::Tree, first::Int, last::Int, destination::Int)
  @assert first > 0
  @assert first <= last
  @assert last <= t.capacity + 1
  @assert destination > 0
  @assert destination <= t.capacity + 1

  # Strategy
  # 1) Loop over moved nodes (at target location)
  # 2) Check if parent/children/neighbors connections are to a node that was moved
  #    a) if node was moved: apply offset to current node
  #    b) if node was not moved: go to connected node and update connectivity there

  offset = destination - first
  has_moved(n) = (first <= n <= last)

  for source in first:last
    target = source + offset

    # Update parent
    if has_parent(t, target)
      # Get parent node
      parent_id = t.parent_ids[target]
      if has_moved(parent_id)
        # If parent itself was moved, just update parent id accordingly
        t.parent_ids[target] += offset
      else
        # If parent was not moved, update its corresponding child id
        for child in 1:n_children_per_node(t)
          if t.child_ids[child, parent_id] == source
            t.child_ids[child, parent_id] = target
          end
        end
      end
    end

    # Update children
    for child in 1:n_children_per_node(t)
      if has_child(t, target, child)
        # Get child node
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
        # Get neighbor node
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


# Raw copy operation for ranges of nodes.
#
# This method is used by the higher-level copy operations for AbstractContainer
function raw_copy!(target::Tree, source::Tree, first::Int, last::Int, destination::Int)
  copy_data!(target.parent_ids, source.parent_ids, first, last, destination)
  copy_data!(target.child_ids, source.child_ids, first, last, destination,
             n_children_per_node(target))
  copy_data!(target.neighbor_ids, source.neighbor_ids, first, last,
             destination, n_directions(target))
  copy_data!(target.levels, source.levels, first, last, destination)
  copy_data!(target.coordinates, source.coordinates, first, last, destination)
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
size(c::AbstractContainer) = c.size
Base.length(c::AbstractContainer) = size(c)


# Increase container size by `count` elements
function append!(c::AbstractContainer, count::Int)
  @assert count >= 0 "Count must be non-negative"
  @assert count + size(c) <= capacity(c) "New size would exceed capacity"

  # First, invalidate range (to be sure that no sensible values are accidentally left there)
  invalidate!(c, size(c) + 1, size(c) + count)

  # Then, increase container size
  c.size += count
end
append!(c::AbstractContainer) = append(c, 1)


# Decrease container size by `count` elements
function shrink!(c::AbstractContainer, count::Int)
  @assert count >= 0
  @assert size(c) >= count

  # Rely on remove&shift to do The Right Thing
  remove_shift!(c, size(c) - count + 1, size())
end
shrink!(c::AbstractContainer) = shrink(c, 1)


# Copy data range from source to target container.
#
# Calls `raw_copy` internally, which must be implemented for each concrete type
# inheriting from AbstractContainer.
function copy!(target::AbstractContainer, source::AbstractContainer,
               first::Int, last::Int, destination::Int)
  @assert 1 <= first <= size(source) "First node out of range"
  @assert 1 <= last <= size(source) "Last node out of range"
  @assert 1 <= destination <= size(target) "Destination out of range"
  @assert destination + (last - first) <= size(target) "Target range out of bounds"

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
  @assert 1 <= first <= size(c) "First node $first out of range"
  @assert 1 <= last <= size(c) "Last node $last out of range"
  @assert 1 <= destination <= size(c) "Destination $destination out of range"
  @assert destination + (last - first) <= size(c) "Target range out of bounds"

  # Return if move would be a no-op
  if last < first || first == destination
    return
  end

  # Copy nodes to new location
  raw_copy!(c, first, last, destination)

  # Move connectivity
  move_connectivity!(c, first, last, destination)


  # Invalidate original node locations (unless they already contain new data due to overlap)
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
  @assert 1 <= a <= size(c) "a out of range"
  @assert 1 <= b <= size(c) "b out of range"

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
  @assert 1 <= position <= size(c) + 1 "Insert position out of range"
  @assert count >= 0 "Count must be non-negative"
  @assert count + size(c) <= capacity(c) "New size would exceed capacity"

  # Return if insertation would be a no-op
  if count == 0
    return
  end

  # Append and return if insertion is beyond last current element
  if position == size(c) + 1
    append!(c, count)
    return
  end

  # Increase size
  c.size += count

  # Move original nodes that currently occupy the insertion region, unless
  # insert position is one beyond previous size
  if position <= size(c) - count
    move!(c, position, size(c) - count, position + count)
  end
end
insert!(c) = insert!(c, position, 1)


# Erase elements from container, deleting their connectivity and then invalidating their data.
function erase!(c::AbstractContainer, first::Int, last::Int)
  @assert 1 <= first <= size(c) "First node out of range"
  @assert 1 <= last <= size(c) "Last node out of range"

  # Return if eraseure would be a no-op
  if last < first
    return
  end

  # Delete connectivity and invalidate nodes
  delete_connectivity!(c, first, last)
  invalidate!(c, first, last)
end
erase!(c::AbstractContainer, id::Int) = erase!(c, id, id)


# Remove nodes and shift existing nodes forward to close the gap
function remove_shift!(c::AbstractContainer, first::Int, last::Int)
  @assert 1 <= first <= size(c) "First node out of range"
  @assert 1 <= last <= size(c) "Last node out of range"

  # Return if removal would be a no-op
  if last < first
    return
  end

  # Delete connectivity of nodes to be removed
  delete_connectivity!(c, first, last)

  if last == size
    # If everything up to the last node is removed, no shifting is required
    invalidate!(c, first, last)
  else
    # Otherwise, the corresponding nodes are moved forward
    move!(c, last + 1, size(c), first)
  end

  # Reduce size
  count = last - first + 1
  c.size -= count
end
remove_shift!(c::AbstractContainer, id::Int) = remove_shift!(c, id, id)


# Remove nodes and fill gap with nodes from the end of the container (to reduce copy operations)
function remove_fill!(c::AbstractContainer, first::Int, last::Int)
  @assert 1 <= first <= size(c) "First node out of range"
  @assert 1 <= last <= size(c) "Last node out of range"

  # Return if removal would be a no-op
  if last < first
    return
  end

  # Delete connectivity of nodes to be removed and then invalidate them
  delete_connectivity!(c, first, last)
  invalidate!(c, first, last)

  # Copy nodes from end (unless last is already the last node)
  count = last - first + 1
  if last < size(c)
    move(c, max(size(c) - count, last + 1), size(c), first)
  end

  # Reduce size
  c.size -= count
end


# Reset container to zero-size and with a new capacity
function reset!(c::AbstractContainer, capacity::Int)
  @assert capacity >=0

  c.capacity = capacity
  c.size = 0
  c.dummy = capacity + 1
  reset_data_structures!(c::AbstractContainer)
end


# Invalidate all elements and set size to zero.
function clear!(c::AbstractContainer)
  invalidate!(c)
  c.size = 0
end

end
