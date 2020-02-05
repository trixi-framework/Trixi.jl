module Trees

using StaticArrays: MVector


abstract type AbstractContainer end
abstract type AbstractTree{D<:Integer} <: AbstractContainer end


mutable struct Tree{D} <: AbstractTree{D}
  parent_ids::Vector{Int}
  child_ids::Matrix{Int}
  neighbor_ids::Matrix{Int}
  levels::Vector{Int}
  coordinates::Matrix{Float64}

  capacity::Int
  size::Int

  center::MVector{D, Float64}
  length::Float64

  function Tree{D}(capacity::Int, center::AbstractArray{Float64}, length::Float64) where D
    # Create instance
    b = new()

    # Initialize fields with defaults
    # Note: size as capacity + 1 is to use `capacity + 1` as temporary storage for swap operations
    b.parent_ids = Vector{Int}(0, capacity + 1)
    b.child_ids = Matrix{Int}(0, 2^D, capacity + 1)
    b.neighbor_ids = Matrix{Int}(0, 2*D, capacity + 1)
    b.levels = Vector{Int}(-1, capacity + 1)
    b.coordinates = Matrix{Float64}(NaN, D, capacity + 1)

    b.capacity = capacity
    b.size = 0

    b.center = center
    b.length = length

    # Create initial node
    b.size += 1
    b.levels[1] = 0
    b.coordinates[:, 1] = b.center
  end
end

Tree(::Val{D}, args...) where D = Tree{D}(args...)


has_parent(t::Tree, node_id::Int) = t.parent_ids[node_id] > 0
has_child(t::Tree, node_id::Int, child_id::Int) = t.parent_ids[child_id, node_id] > 0
has_children(t::Tree, node_id::Int) = n_children(t, node_id) > 0
n_children(t::Tree, node_id::Int) = count(x -> (x > 0), @view t.child_ids[:, node_id])
has_neighbor(t::Tree, node_id::Int, direction::Int) = t.neighbor_ids[direction, node_id] > 0
has_any_neighbor(t::Tree, node_id::Int, direction::Int) = (
   has_neighbor(t, node_id, direction) ||
   (has_parent(t, node_id) && has_neighbor(t, t.parent_ids[node_id], direction)))

n_children_per_node(::Tree{D}) where D = 2^D
n_neighbors_per_node(::Tree{D}) where D = 2 * D
opposite_neighbor(direction::Int) = direction + 1 - 2 * ((direction + 1) % 2)


function invalidate!(t::Tree, first::Int, last::Int)
  @assert first > 0
  @assert first <= last
  @assert last <= t.capacity + 1

  b.parent_ids[first:last] = 0
  b.child_ids[:, first:last] = 0
  b.neighbor_ids[:, first:last] = 0
  b.levels[first:last] = -1
  b.coordinates[:, first:last] = NaN
end
invalidate!(t::Tree) = invalidate!(t, 1, t.capacity + 1)


# Delete connectivity with parents/children/neighbors before nodes are erased
function delete_connectivity(t::Tree, first::Int, last::Int)
  @assert first > 0
  @assert first <= last
  @assert last <= t.capacity + 1

  # Iterate over all cells
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
    for neighbor in 1:n_neighbors_per_node(t)
      if has_neighbor(t, node_id, neighbor)
        t.neighbor_ids[opposite_neighbor(neighbor), t.neighbor_ids[neighbor, node_id]] = 0
      end
    end
  end
end


# Move connectivity with parents/children/neighbors after nodes have been moved
function move_connectivity(t::Tree, first::Int, last::Int, destination::Int)
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
          t.child_ids[child_id, target] += offset
        else
          # If child was not moved, update its parent id
          t.parent_ids[child_id] = target
        end
      end
    end

    # Update neighbors
    for neighbor in 1:n_neighbors_per_node(t)
      if has_neighbor(t, target, neighbor)
        # Get neighbor node
        neighbor_id = t.neighbor_ids[neighbor, target]
        if has_moved(neighbor_id)
          # If neighbor itself was moved, just update neighbor id accordingly
          t.neighbor_ids[neighbor, target] += offset
        else
          # If neighbor was not moved, update its opposing neighbor id
          t.neighbor_ids[opposite_neighbor(neighbor), neighbor_id] = source
        end
      end
    end
  end
end


# Raw copy operation for ranges of cells
# function raw_copy(

end
