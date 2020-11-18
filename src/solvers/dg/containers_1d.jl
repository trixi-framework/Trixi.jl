
# Container data structure (structure-of-arrays style) for DG elements
# TODO: Taal refactor, remove NVARS, POLYDEG?
mutable struct ElementContainer1D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  inverse_jacobian::Vector{RealT}      # [elements]
  node_coordinates::Array{RealT, 3}    # [orientation, i, elements]
  surface_flux_values::Array{RealT, 3} # [variables, direction, elements]
  cell_ids::Vector{Int}                # [elements]
  # internal `resize!`able storage
  _node_coordinates::Vector{RealT}
  _surface_flux_values::Vector{RealT}
end

# Only one-dimensional `Array`s are `resize!`able in Julia.
# Hence, we use `Vector`s as internal storage and `resize!`
# them whenever needed. Then, we reuse the same memory by
# `unsafe_wrap`ping multi-dimensional `Array`s around the
# internal storage.
function Base.resize!(elements::ElementContainer1D{RealT, NVARS, POLYDEG},
                      capacity) where {RealT, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  @unpack _node_coordinates, _surface_flux_values,
          inverse_jacobian, cell_ids = elements

  resize!(inverse_jacobian, capacity)

  resize!(_node_coordinates, 1 * n_nodes * capacity)
  elements.node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                          (1, n_nodes, capacity))

  resize!(_surface_flux_values, NVARS * 2 * 1 * capacity)
  elements.surface_flux_values = unsafe_wrap(Array, pointer(_surface_flux_values),
                                             (NVARS, 2 * 1, capacity))

  resize!(cell_ids, capacity)

  return nothing
end


function ElementContainer1D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  inverse_jacobian = fill(nan, capacity)

  _node_coordinates = fill(nan, 1 * n_nodes * capacity)
  node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                 (1, n_nodes, capacity))

  _surface_flux_values = fill(nan, NVARS * 2 * 1 * capacity)
  surface_flux_values = unsafe_wrap(Array, pointer(_surface_flux_values),
                                    (NVARS, 2 * 1, capacity))

  cell_ids = fill(typemin(Int), capacity)

  return ElementContainer1D{RealT, NVARS, POLYDEG}(
    inverse_jacobian, node_coordinates, surface_flux_values, cell_ids,
    _node_coordinates, _surface_flux_values)
end


# Return number of elements
@inline nelements(elements::ElementContainer1D) = length(elements.cell_ids)
# TODO: Taal performance, 1:nelements(elements) vs. Base.OneTo(nelements(elements))
@inline eachelement(elements::ElementContainer1D) = Base.OneTo(nelements(elements))


# Create element container and initialize element data
function init_elements(cell_ids, mesh::TreeMesh1D,
                       equations::AbstractEquations{1, NVARS},
                       basis::LobattoLegendreBasis{T, NNODES}, ::Type{RealT}) where {RealT<:Real, NVARS, T, NNODES}
  # Initialize container
  n_elements = length(cell_ids)
  elements = ElementContainer1D{RealT, NVARS, NNODES-1}(n_elements)

  init_elements!(elements, cell_ids, mesh, basis.nodes)
  return elements
end

function init_elements!(elements, cell_ids, mesh::TreeMesh1D, nodes)
  n_nodes = length(nodes)

  # Store cell ids
  elements.cell_ids .= cell_ids

  # Calculate inverse Jacobian and node coordinates
  for element in eachelement(elements)
    # Get cell id
    cell_id = cell_ids[element]

    # Get cell length
    dx = length_at_cell(mesh.tree, cell_id)

    # Calculate inverse Jacobian as 1/(h/2)
    elements.inverse_jacobian[element] = 2/dx

    # Calculate node coordinates
      for i in 1:n_nodes
        elements.node_coordinates[1, i, element] = (
            mesh.tree.coordinates[1, cell_id] + dx/2 * nodes[i])
      end
  end

  return elements
end



# Container data structure (structure-of-arrays style) for DG interfaces
# TODO: Taal refactor, remove NVARS, POLYDEG?
mutable struct InterfaceContainer1D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  u::Array{RealT, 3}        # [leftright, variables, interfaces]
  neighbor_ids::Matrix{Int} # [leftright, interfaces]
  orientations::Vector{Int} # [interfaces]
  # internal `resize!`able storage
  _u::Vector{RealT}
  _neighbor_ids::Vector{Int}
end

# See explanation of Base.resize! for the element container
function Base.resize!(interfaces::InterfaceContainer1D{RealT, NVARS, POLYDEG},
                      capacity) where {RealT, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  @unpack _u, _neighbor_ids, orientations = interfaces

  resize!(_u, 2 * NVARS * capacity)
  interfaces.u = unsafe_wrap(Array, pointer(_u),
                             (2, NVARS, capacity))

  resize!(_neighbor_ids, 2 * capacity)
  interfaces.neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids ),
                                        (2, capacity))

  resize!(orientations, capacity)

  return nothing
end


function InterfaceContainer1D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  _u = fill(nan, 2 * NVARS * capacity)
  u = unsafe_wrap(Array, pointer(_u),
                  (2, NVARS, capacity))

  _neighbor_ids = fill(typemin(Int), 2 * capacity)
  neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids ),
                             (2, capacity))

  orientations = fill(typemin(Int), capacity)

  return InterfaceContainer1D{RealT, NVARS, POLYDEG}(
    u, neighbor_ids, orientations,
    _u, _neighbor_ids)
end


# Return number of interfaces
@inline ninterfaces(interfaces::InterfaceContainer1D) = length(interfaces.orientations)


# Create interface container and initialize interface data in `elements`.
function init_interfaces(cell_ids, mesh::TreeMesh1D,
                         elements::ElementContainer1D{RealT, NVARS, POLYDEG}) where {RealT<:Real, NVARS, POLYDEG}
  # Initialize container
  n_interfaces = count_required_interfaces(mesh, cell_ids)
  interfaces = InterfaceContainer1D{RealT, NVARS, POLYDEG}(n_interfaces)

  # Connect elements with interfaces
  init_interfaces!(interfaces, elements, mesh)
  return interfaces
end

# Count the number of interfaces that need to be created
function count_required_interfaces(mesh::TreeMesh1D, cell_ids)
  count = 0

  # Iterate over all cells
  for cell_id in cell_ids
    for direction in eachdirection(mesh.tree)
      # Only count interfaces in positive direction to avoid double counting
      if direction == 1
        continue
      end

      # Skip if no neighbor exists
      if !has_any_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      count += 1
    end
  end

  return count
end

# Initialize connectivity between elements and interfaces
function init_interfaces!(interfaces, elements, mesh::TreeMesh1D)
  # Construct cell -> element mapping for easier algorithm implementation
  tree = mesh.tree
  c2e = zeros(Int, length(tree))
  for element in eachelement(elements)
    c2e[elements.cell_ids[element]] = element
  end

  # Reset interface count
  count = 0

  # Iterate over all elements to find neighbors and to connect via interfaces
  for element in eachelement(elements)
    # Get cell id
    cell_id = elements.cell_ids[element]

    # Loop over directions
    for direction in eachdirection(mesh.tree)
      # Only create interfaces in positive direction
      if direction == 1
        continue
      end

      # Skip if no neighbor exists and current cell is not small
      if !has_any_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      count += 1

      if has_neighbor(mesh.tree, cell_id, direction)
        neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
        if has_children(mesh.tree, neighbor_cell_id) # Cell has small neighbor
          interfaces.neighbor_ids[2, count] = c2e[mesh.tree.child_ids[1, neighbor_cell_id]]
        else # Cell has same refinement level neighbor
          interfaces.neighbor_ids[2, count] = c2e[neighbor_cell_id]
        end
      else # Cell is small and has large neighbor
        parent_id = mesh.tree.parent_ids[cell_id]
        neighbor_cell_id = mesh.tree.neighbor_ids[direction, parent_id]
        interfaces.neighbor_ids[2, count] = c2e[neighbor_cell_id]
      end

      interfaces.neighbor_ids[1, count] = element
      # Set orientation (x -> 1)
      interfaces.orientations[count] = 1
    end
  end

  @assert count == ninterfaces(interfaces) ("Actual interface count ($count) does not match " *
                                            "expectations $(ninterfaces(interfaces))")
end



# Container data structure (structure-of-arrays style) for DG boundaries
# TODO: Taal refactor, remove NVARS, POLYDEG?
mutable struct BoundaryContainer1D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  u::Array{RealT, 3}                # [leftright, variables, boundaries]
  neighbor_ids::Vector{Int}         # [boundaries]
  orientations::Vector{Int}         # [boundaries]
  neighbor_sides::Vector{Int}       # [boundaries]
  node_coordinates::Array{RealT, 2} # [orientation, elements]
  n_boundaries_per_direction::SVector{2, Int} # [direction]
  # internal `resize!`able storage
  _u::Vector{RealT}
  _node_coordinates::Vector{RealT}
end

# See explanation of Base.resize! for the element container
function Base.resize!(boundaries::BoundaryContainer1D{RealT, NVARS, POLYDEG},
                      capacity) where {RealT, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  @unpack _u, _node_coordinates,
          neighbor_ids, orientations, neighbor_sides = boundaries

  resize!(_u, 2 * NVARS * capacity)
  boundaries.u = unsafe_wrap(Array, pointer(_u),
                             (2, NVARS, capacity))

  resize!(_node_coordinates, 1 * capacity)
  boundaries.node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates ),
                                            (1, capacity))

  resize!(neighbor_ids, capacity)

  resize!(orientations, capacity)

  resize!(neighbor_sides, capacity)

  return nothing
end


function BoundaryContainer1D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  _u = fill(nan, 2 * NVARS * capacity)
  u = unsafe_wrap(Array, pointer(_u),
                  (2, NVARS, capacity))

  neighbor_ids = fill(typemin(Int), capacity)

  orientations = fill(typemin(Int), capacity)

  neighbor_sides = fill(typemin(Int), capacity)

  _node_coordinates = fill(nan, 1 * capacity)
  node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                 (1, capacity))

  n_boundaries_per_direction = SVector(0, 0)

  return BoundaryContainer1D{RealT, NVARS, POLYDEG}(
    u, neighbor_ids, orientations, neighbor_sides,
    node_coordinates, n_boundaries_per_direction,
    _u, _node_coordinates)
end


# Return number of boundaries
nboundaries(boundaries::BoundaryContainer1D) = length(boundaries.orientations)


# Create boundaries container and initialize boundary data in `elements`.
function init_boundaries(cell_ids, mesh::TreeMesh1D,
                         elements::ElementContainer1D{RealT, NVARS, POLYDEG}) where {RealT<:Real, NVARS, POLYDEG}
  # Initialize container
  n_boundaries = count_required_boundaries(mesh, cell_ids)
  boundaries = BoundaryContainer1D{RealT, NVARS, POLYDEG}(n_boundaries)

  # Connect elements with boundaries
  init_boundaries!(boundaries, elements, mesh)
  return boundaries
end

# Count the number of boundaries that need to be created
function count_required_boundaries(mesh::TreeMesh1D, cell_ids)
  count = 0

  # Iterate over all cells
  for cell_id in cell_ids
    for direction in eachdirection(mesh.tree)
      # If neighbor exists, current cell is not at a boundary
      if has_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      # If coarse neighbor exists, current cell is not at a boundary
      if has_coarse_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      # No neighbor exists in this direction -> must be a boundary
      count += 1
    end
  end

  return count
end

# Initialize connectivity between elements and boundaries
function init_boundaries!(boundaries, elements, mesh::TreeMesh1D)
  # Reset boundaries count
  count = 0

  # Initialize boundary counts
  counts_per_direction = MVector(0, 0)

  # OBS! Iterate over directions first, then over elements, and count boundaries in each direction
  # Rationale: This way the boundaries are internally sorted by the directions -x, +x, -y etc.,
  #            obviating the need to store the boundary condition to be applied explicitly.
  # Loop over directions
  for direction in eachdirection(mesh.tree)
    # Iterate over all elements to find missing neighbors and to connect to boundaries
    for element in eachelement(elements)
      # Get cell id
      cell_id = elements.cell_ids[element]

      # If neighbor exists, current cell is not at a boundary
      if has_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      # If coarse neighbor exists, current cell is not at a boundary
      if has_coarse_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      # Create boundary
      count += 1
      counts_per_direction[direction] += 1

      # Set neighbor element id
      boundaries.neighbor_ids[count] = element

      # Set neighbor side, which denotes the direction (1 -> negative, 2 -> positive) of the element
      if direction == 2
        boundaries.neighbor_sides[count] = 1
      else
        boundaries.neighbor_sides[count] = 2
      end

      # Set orientation (x -> 1)
      boundaries.orientations[count] = 1

      # Store node coordinates
      enc = elements.node_coordinates
      if direction == 1 # -x direction
        boundaries.node_coordinates[:, count] .= enc[:, 1,  element]
      elseif direction == 2 # +x direction
        boundaries.node_coordinates[:, count] .= enc[:, end, element]
      else
        error("should not happen")
      end
    end
  end

  @assert count == nboundaries(boundaries) ("Actual boundaries count ($count) does not match " *
                                            "expectations $(nboundaries(boundaries))")
  @assert sum(counts_per_direction) == count

  boundaries.n_boundaries_per_direction = SVector(counts_per_direction)

  return boundaries.n_boundaries_per_direction
end

