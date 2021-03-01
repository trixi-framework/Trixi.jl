
# Container data structure (structure-of-arrays style) for DG elements
# TODO: Taal refactor, remove NVARS, POLYDEG?
mutable struct ElementContainer2D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  inverse_jacobian::Vector{RealT}      # [elements]
  node_coordinates::Array{RealT, 4}    # [orientation, i, j, elements]
  surface_flux_values::Array{RealT, 4} # [variables, i, direction, elements]
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
function Base.resize!(elements::ElementContainer2D{RealT, NVARS, POLYDEG},
                      capacity) where {RealT, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  @unpack _node_coordinates, _surface_flux_values,
          inverse_jacobian, cell_ids = elements

  resize!(inverse_jacobian, capacity)

  resize!(_node_coordinates, 2 * n_nodes * n_nodes * capacity)
  elements.node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                          (2, n_nodes, n_nodes, capacity))

  resize!(_surface_flux_values, NVARS * n_nodes * 2 * 2 * capacity)
  elements.surface_flux_values = unsafe_wrap(Array, pointer(_surface_flux_values),
                                             (NVARS, n_nodes, 2 * 2, capacity))

  resize!(cell_ids, capacity)

  return nothing
end


function ElementContainer2D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  inverse_jacobian = fill(nan, capacity)

  _node_coordinates = fill(nan, 2 * n_nodes * n_nodes * capacity)
  node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                 (2, n_nodes, n_nodes, capacity))

  _surface_flux_values = fill(nan, NVARS * n_nodes * 2 * 2 * capacity)
  surface_flux_values = unsafe_wrap(Array, pointer(_surface_flux_values),
                                    (NVARS, n_nodes, 2 * 2, capacity))

  cell_ids = fill(typemin(Int), capacity)


  return ElementContainer2D{RealT, NVARS, POLYDEG}(
    inverse_jacobian, node_coordinates, surface_flux_values, cell_ids,
    _node_coordinates, _surface_flux_values)
end


# Return number of elements
@inline nelements(elements::ElementContainer2D) = length(elements.cell_ids)
# TODO: Taal performance, 1:nelements(elements) vs. Base.OneTo(nelements(elements))
@inline eachelement(elements::ElementContainer2D) = Base.OneTo(nelements(elements))


# Create element container and initialize element data
function init_elements(cell_ids, mesh::TreeMesh2D,
                       equations::AbstractEquations{2, NVARS},
                       basis::LobattoLegendreBasis{T, NNODES}, ::Type{RealT}) where {RealT<:Real, NVARS, T, NNODES}
  # Initialize container
  n_elements = length(cell_ids)
  elements = ElementContainer2D{RealT, NVARS, NNODES-1}(n_elements)

  init_elements!(elements, cell_ids, mesh, basis.nodes)
  return elements
end

function init_elements!(elements, cell_ids, mesh::TreeMesh2D, nodes)
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
    for j in 1:n_nodes, i in 1:n_nodes
      elements.node_coordinates[1, i, j, element] = (
          mesh.tree.coordinates[1, cell_id] + dx/2 * nodes[i])
      elements.node_coordinates[2, i, j, element] = (
          mesh.tree.coordinates[2, cell_id] + dx/2 * nodes[j])
    end
  end

  return elements
end



# Container data structure (structure-of-arrays style) for DG interfaces
# TODO: Taal refactor, remove NVARS, POLYDEG?
mutable struct InterfaceContainer2D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  u::Array{RealT, 4}        # [leftright, variables, i, interfaces]
  neighbor_ids::Matrix{Int} # [leftright, interfaces]
  orientations::Vector{Int} # [interfaces]
  # internal `resize!`able storage
  _u::Vector{RealT}
  _neighbor_ids::Vector{Int}
end

# See explanation of Base.resize! for the element container
function Base.resize!(interfaces::InterfaceContainer2D{RealT, NVARS, POLYDEG},
                      capacity) where {RealT, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  @unpack _u, _neighbor_ids, orientations = interfaces

  resize!(_u, 2 * NVARS * n_nodes * capacity)
  interfaces.u = unsafe_wrap(Array, pointer(_u),
                             (2, NVARS, n_nodes, capacity))

  resize!(_neighbor_ids, 2 * capacity)
  interfaces.neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids ),
                                        (2, capacity))

  resize!(orientations, capacity)

  return nothing
end


function InterfaceContainer2D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  _u = fill(nan, 2 * NVARS * n_nodes * capacity)
  u = unsafe_wrap(Array, pointer(_u),
                  (2, NVARS, n_nodes, capacity))

  _neighbor_ids = fill(typemin(Int), 2 * capacity)
  neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids ),
                             (2, capacity))

  orientations = fill(typemin(Int), capacity)


  return InterfaceContainer2D{RealT, NVARS, POLYDEG}(
    u, neighbor_ids, orientations,
    _u, _neighbor_ids)
end


# Return number of interfaces
@inline ninterfaces(interfaces::InterfaceContainer2D) = length(interfaces.orientations)


# Create interface container and initialize interface data in `elements`.
function init_interfaces(cell_ids, mesh::TreeMesh2D,
                         elements::ElementContainer2D{RealT, NVARS, POLYDEG}) where {RealT<:Real, NVARS, POLYDEG}
  # Initialize container
  n_interfaces = count_required_interfaces(mesh, cell_ids)
  interfaces = InterfaceContainer2D{RealT, NVARS, POLYDEG}(n_interfaces)

  # Connect elements with interfaces
  init_interfaces!(interfaces, elements, mesh)
  return interfaces
end

# Count the number of interfaces that need to be created
function count_required_interfaces(mesh::TreeMesh2D, cell_ids)
  count = 0

  # Iterate over all cells
  for cell_id in cell_ids
    for direction in eachdirection(mesh.tree)
      # Only count interfaces in positive direction to avoid double counting
      if direction % 2 == 1
        continue
      end

      # If no neighbor exists, current cell is small or at boundary and thus we need a mortar
      if !has_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      # Skip if neighbor has children
      neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
      if has_children(mesh.tree, neighbor_cell_id)
        continue
      end

      # Skip if neighbor is on different rank -> create MPI interface instead
      if mpi_isparallel() && !is_own_cell(mesh.tree, neighbor_cell_id)
        continue
      end

      count += 1
    end
  end

  return count
end

# Initialize connectivity between elements and interfaces
function init_interfaces!(interfaces, elements, mesh::TreeMesh2D)
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
      if direction % 2 == 1
        continue
      end

      # If no neighbor exists, current cell is small and thus we need a mortar
      if !has_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      # Skip if neighbor has children
      neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
      if has_children(mesh.tree, neighbor_cell_id)
        continue
      end

      # Skip if neighbor is on different rank -> create MPI interface instead
      if mpi_isparallel() && !is_own_cell(mesh.tree, neighbor_cell_id)
        continue
      end

      # Create interface between elements (1 -> "left" of interface, 2 -> "right" of interface)
      count += 1
      interfaces.neighbor_ids[2, count] = c2e[neighbor_cell_id]
      interfaces.neighbor_ids[1, count] = element

      # Set orientation (x -> 1, y -> 2)
      interfaces.orientations[count] = div(direction, 2)
    end
  end

  @assert count == ninterfaces(interfaces) ("Actual interface count ($count) does not match " *
                                            "expectations $(ninterfaces(interfaces))")
end



# Container data structure (structure-of-arrays style) for DG boundaries
# TODO: Taal refactor, remove NVARS, POLYDEG?
mutable struct BoundaryContainer2D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  u::Array{RealT, 4}                # [leftright, variables, i, boundaries]
  neighbor_ids::Vector{Int}         # [boundaries]
  orientations::Vector{Int}         # [boundaries]
  neighbor_sides::Vector{Int}       # [boundaries]
  node_coordinates::Array{RealT, 3} # [orientation, i, elements]
  n_boundaries_per_direction::SVector{4, Int} # [direction]
  # internal `resize!`able storage
  _u::Vector{RealT}
  _node_coordinates::Vector{RealT}
end

# See explanation of Base.resize! for the element container
function Base.resize!(boundaries::BoundaryContainer2D{RealT, NVARS, POLYDEG},
                      capacity) where {RealT, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  @unpack _u, _node_coordinates,
          neighbor_ids, orientations, neighbor_sides = boundaries

  resize!(_u, 2 * NVARS * n_nodes * capacity)
  boundaries.u = unsafe_wrap(Array, pointer(_u),
                             (2, NVARS, n_nodes, capacity))

  resize!(_node_coordinates, 2 * n_nodes * capacity)
  boundaries.node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates ),
                                            (2, n_nodes, capacity))

  resize!(neighbor_ids, capacity)

  resize!(orientations, capacity)

  resize!(neighbor_sides, capacity)

  return nothing
end


function BoundaryContainer2D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  _u = fill(nan, 2 * NVARS * n_nodes * capacity)
  u = unsafe_wrap(Array, pointer(_u),
                  (2, NVARS, n_nodes, capacity))

  neighbor_ids = fill(typemin(Int), capacity)

  orientations = fill(typemin(Int), capacity)

  neighbor_sides = fill(typemin(Int), capacity)

  _node_coordinates = fill(nan, 2 * n_nodes * capacity)
  node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                 (2, n_nodes, capacity))

  n_boundaries_per_direction = SVector(0, 0, 0, 0)

  return BoundaryContainer2D{RealT, NVARS, POLYDEG}(
    u, neighbor_ids, orientations, neighbor_sides,
    node_coordinates, n_boundaries_per_direction,
    _u, _node_coordinates)
end


# Return number of boundaries
@inline nboundaries(boundaries::BoundaryContainer2D) = length(boundaries.orientations)


# Create boundaries container and initialize boundary data in `elements`.
function init_boundaries(cell_ids, mesh::TreeMesh2D,
                         elements::ElementContainer2D{RealT, NVARS, POLYDEG}) where {RealT<:Real, NVARS, POLYDEG}
  # Initialize container
  n_boundaries = count_required_boundaries(mesh, cell_ids)
  boundaries = BoundaryContainer2D{RealT, NVARS, POLYDEG}(n_boundaries)

  # Connect elements with boundaries
  init_boundaries!(boundaries, elements, mesh)
  return boundaries
end

# Count the number of boundaries that need to be created
function count_required_boundaries(mesh::TreeMesh2D, cell_ids)
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
function init_boundaries!(boundaries, elements, mesh::TreeMesh2D)
  # Reset boundaries count
  count = 0

  # Initialize boundary counts
  counts_per_direction = MVector(0, 0, 0, 0)

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
      if direction in (2, 4)
        boundaries.neighbor_sides[count] = 1
      else
        boundaries.neighbor_sides[count] = 2
      end

      # Set orientation (x -> 1, y -> 2)
      if direction in (1, 2)
        boundaries.orientations[count] = 1
      else
        boundaries.orientations[count] = 2
      end

      # Store node coordinates
      enc = elements.node_coordinates
      if direction == 1 # -x direction
        boundaries.node_coordinates[:, :, count] .= enc[:, 1,   :,   element]
      elseif direction == 2 # +x direction
        boundaries.node_coordinates[:, :, count] .= enc[:, end, :,   element]
      elseif direction == 3 # -y direction
        boundaries.node_coordinates[:, :, count] .= enc[:, :,   1,   element]
      elseif direction == 4 # +y direction
        boundaries.node_coordinates[:, :, count] .= enc[:, :,   end, element]
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



# Container data structure (structure-of-arrays style) for DG L2 mortars
# Positions/directions for orientations = 1, large_sides = 2:
# mortar is orthogonal to x-axis, large side is in positive coordinate direction wrt mortar
#           |    |
# upper = 2 |    |
#           |    |
#                | 3 = large side
#           |    |
# lower = 1 |    |
#           |    |
# TODO: Taal refactor, remove NVARS, POLYDEG?
# TODO: Taal refactor, mutable struct or resize! for AMR?
mutable struct L2MortarContainer2D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  u_upper::Array{RealT, 4}  # [leftright, variables, i, mortars]
  u_lower::Array{RealT, 4}  # [leftright, variables, i, mortars]
  neighbor_ids::Matrix{Int} # [position, mortars]
  # Large sides: left -> 1, right -> 2
  large_sides::Vector{Int}  # [mortars]
  orientations::Vector{Int} # [mortars]
  # internal `resize!`able storage
  _u_upper::Vector{RealT}
  _u_lower::Vector{RealT}
  _neighbor_ids::Vector{Int}
end

# See explanation of Base.resize! for the element container
function Base.resize!(mortars::L2MortarContainer2D{RealT, NVARS, POLYDEG},
                      capacity) where {RealT, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  @unpack _u_upper, _u_lower, _neighbor_ids,
          large_sides, orientations = mortars

  resize!(_u_upper, 2 * NVARS * n_nodes * capacity)
  mortars.u_upper = unsafe_wrap(Array, pointer(_u_upper),
                                (2, NVARS, n_nodes, capacity))

  resize!(_u_lower, 2 * NVARS * n_nodes * capacity)
  mortars.u_lower = unsafe_wrap(Array, pointer(_u_lower),
                                (2, NVARS, n_nodes, capacity))

  resize!(_neighbor_ids, 3 * capacity)
  mortars.neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids ),
                                        (3, capacity))

  resize!(large_sides, capacity)

  resize!(orientations, capacity)

  return nothing
end


function L2MortarContainer2D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  _u_upper = fill(nan, 2 * NVARS * n_nodes * capacity)
  u_upper = unsafe_wrap(Array, pointer(_u_upper),
                        (2, NVARS, n_nodes, capacity))

  _u_lower = fill(nan, 2 * NVARS * n_nodes * capacity)
  u_lower = unsafe_wrap(Array, pointer(_u_lower),
                        (2, NVARS, n_nodes, capacity))

  _neighbor_ids = fill(typemin(Int), 3 * capacity)
  neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids),
                             (3, capacity))

  large_sides  = fill(typemin(Int), capacity)

  orientations = fill(typemin(Int), capacity)

  return L2MortarContainer2D{RealT, NVARS, POLYDEG}(
    u_upper, u_lower, neighbor_ids, large_sides, orientations,
    _u_upper, _u_lower, _neighbor_ids)
end


# Return number of L2 mortars
@inline nmortars(l2mortars::L2MortarContainer2D) = length(l2mortars.orientations)


# Allow printing container contents
function Base.show(io::IO, ::MIME"text/plain", c::L2MortarContainer2D)
  @nospecialize c # reduce precompilation time

  println(io, '*'^20)
  for idx in CartesianIndices(c.u_upper)
    println(io, "c.u_upper[$idx] = $(c.u_upper[idx])")
  end
  for idx in CartesianIndices(c.u_lower)
    println(io, "c.u_lower[$idx] = $(c.u_lower[idx])")
  end
  println(io, "transpose(c.neighbor_ids) = $(transpose(c.neighbor_ids))")
  println(io, "c.large_sides = $(c.large_sides)")
  println(io, "c.orientations = $(c.orientations)")
  print(io,   '*'^20)
end


# Create mortar container and initialize mortar data in `elements`.
function init_mortars(cell_ids, mesh::TreeMesh2D,
                      elements::ElementContainer2D{RealT, NVARS, POLYDEG},
                      mortar::LobattoLegendreMortarL2) where {RealT<:Real, NVARS, POLYDEG}
  # Initialize containers
  n_mortars = count_required_mortars(mesh, cell_ids)
  mortars = L2MortarContainer2D{RealT, NVARS, POLYDEG}(n_mortars)

  # Connect elements with mortars
  init_mortars!(mortars, elements, mesh)
  return mortars
end

# Count the number of mortars that need to be created
function count_required_mortars(mesh::TreeMesh2D, cell_ids)
  count = 0

  # Iterate over all cells and count mortars from perspective of coarse cells
  for cell_id in cell_ids
    for direction in eachdirection(mesh.tree)
      # If no neighbor exists, cell is small with large neighbor or at boundary -> do nothing
      if !has_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      # If neighbor has no children, this is a conforming interface -> do nothing
      neighbor_id = mesh.tree.neighbor_ids[direction, cell_id]
      if !has_children(mesh.tree, neighbor_id)
        continue
      end

      count +=1
    end
  end

  return count
end

# Initialize connectivity between elements and mortars
function init_mortars!(mortars, elements, mesh::TreeMesh2D)
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

    for direction in eachdirection(mesh.tree)
      # If no neighbor exists, cell is small with large neighbor -> do nothing
      if !has_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      # If neighbor has no children, this is a conforming interface -> do nothing
      neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
      if !has_children(mesh.tree, neighbor_cell_id)
        continue
      end

      # Create mortar between elements:
      # 1 -> small element in negative coordinate direction
      # 2 -> small element in positive coordinate direction
      # 3 -> large element
      count += 1
      mortars.neighbor_ids[3, count] = element
      if direction == 1
        mortars.neighbor_ids[1, count] = c2e[mesh.tree.child_ids[2, neighbor_cell_id]]
        mortars.neighbor_ids[2, count] = c2e[mesh.tree.child_ids[4, neighbor_cell_id]]
      elseif direction == 2
        mortars.neighbor_ids[1, count] = c2e[mesh.tree.child_ids[1, neighbor_cell_id]]
        mortars.neighbor_ids[2, count] = c2e[mesh.tree.child_ids[3, neighbor_cell_id]]
      elseif direction == 3
        mortars.neighbor_ids[1, count] = c2e[mesh.tree.child_ids[3, neighbor_cell_id]]
        mortars.neighbor_ids[2, count] = c2e[mesh.tree.child_ids[4, neighbor_cell_id]]
      elseif direction == 4
        mortars.neighbor_ids[1, count] = c2e[mesh.tree.child_ids[1, neighbor_cell_id]]
        mortars.neighbor_ids[2, count] = c2e[mesh.tree.child_ids[2, neighbor_cell_id]]
      else
        error("should not happen")
      end

      # Set large side, which denotes the direction (1 -> negative, 2 -> positive) of the large side
      if direction in [2, 4]
        mortars.large_sides[count] = 1
      else
        mortars.large_sides[count] = 2
      end

      # Set orientation (x -> 1, y -> 2)
      if direction in [1, 2]
        mortars.orientations[count] = 1
      else
        mortars.orientations[count] = 2
      end
    end
  end

  @assert count == nmortars(mortars) ("Actual mortar count ($count) does not match " *
                                      "expectations $(nmortars(mortars))")
end



# Container data structure (structure-of-arrays style) for DG MPI interfaces
mutable struct MPIInterfaceContainer2D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  u::Array{RealT, 4}             # [leftright, variables, i, interfaces]
  local_element_ids::Vector{Int} # [interfaces]
  orientations::Vector{Int}      # [interfaces]
  remote_sides::Vector{Int}      # [interfaces]
  # internal `resize!`able storage
  _u::Vector{RealT}
end

# See explanation of Base.resize! for the element container
function Base.resize!(mpi_interfaces::MPIInterfaceContainer2D{RealT, NVARS, POLYDEG},
                      capacity) where {RealT, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  @unpack _u, local_element_ids, orientations, remote_sides = mpi_interfaces

  resize!(_u, 2 * NVARS * n_nodes * capacity)
  mpi_interfaces.u = unsafe_wrap(Array, pointer(_u),
                                 (2, NVARS, n_nodes, capacity))

  resize!(local_element_ids, capacity)

  resize!(orientations, capacity)

  resize!(remote_sides, capacity)

  return nothing
end


function MPIInterfaceContainer2D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  _u = fill(nan, 2 * NVARS * n_nodes * capacity)
  u = unsafe_wrap(Array, pointer(_u),
                  (2, NVARS, n_nodes, capacity))

  local_element_ids = fill(typemin(Int), capacity)

  orientations = fill(typemin(Int), capacity)

  remote_sides = fill(typemin(Int), capacity)

  return MPIInterfaceContainer2D{RealT, NVARS, POLYDEG}(
    u, local_element_ids, orientations, remote_sides,
    _u)
end


# TODO: Taal, rename to ninterfaces?
# Return number of interfaces
nmpiinterfaces(mpi_interfaces::MPIInterfaceContainer2D) = length(mpi_interfaces.orientations)


# Create MPI interface container and initialize MPI interface data in `elements`.
function init_mpi_interfaces(cell_ids, mesh::TreeMesh2D,
                             elements::ElementContainer2D{RealT, NVARS, POLYDEG}) where {RealT<:Real, NVARS, POLYDEG}
  # Initialize container
  n_mpi_interfaces = count_required_mpi_interfaces(mesh, cell_ids)
  mpi_interfaces = MPIInterfaceContainer2D{RealT, NVARS, POLYDEG}(n_mpi_interfaces)

  # Connect elements with interfaces
  init_mpi_interfaces!(mpi_interfaces, elements, mesh)
  return mpi_interfaces
end

# Count the number of MPI interfaces that need to be created
function count_required_mpi_interfaces(mesh::TreeMesh2D, cell_ids)
  count = 0

  # Iterate over all cells
  for cell_id in cell_ids
    for direction in eachdirection(mesh.tree)
      # If no neighbor exists, current cell is small or at boundary and thus we need a mortar
      if !has_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      # Skip if neighbor has children
      neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
      if has_children(mesh.tree, neighbor_cell_id)
        continue
      end

      # Skip if neighbor is on this rank -> create regular interface instead
      if mpi_isparallel() && is_own_cell(mesh.tree, neighbor_cell_id)
        continue
      end

      count += 1
    end
  end

  return count
end

# Initialize connectivity between elements and interfaces
function init_mpi_interfaces!(mpi_interfaces, elements, mesh::TreeMesh2D)
  # Reset interface count
  count = 0

  # Iterate over all elements to find neighbors and to connect via mpi_interfaces
  for element in eachelement(elements)
    # Get cell id
    cell_id = elements.cell_ids[element]

    # Loop over directions
    for direction in eachdirection(mesh.tree)
      # If no neighbor exists, current cell is small and thus we need a mortar
      if !has_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      # Skip if neighbor has children
      neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
      if has_children(mesh.tree, neighbor_cell_id)
        continue
      end

      # Skip if neighbor is on this MPI rank -> create regular interface instead
      if mpi_isparallel() && is_own_cell(mesh.tree, neighbor_cell_id)
        continue
      end

      # Create interface between elements
      count += 1
      mpi_interfaces.local_element_ids[count] = element

      if direction in (2, 4) # element is "left" of interface, remote cell is "right" of interface
        mpi_interfaces.remote_sides[count] = 2
      else
        mpi_interfaces.remote_sides[count] = 1
      end

      # Set orientation (x -> 1, y -> 2)
      if direction in (1, 2) # x-direction
        mpi_interfaces.orientations[count] = 1
      else # y-direction
        mpi_interfaces.orientations[count] = 2
      end
    end
  end

  @assert count == nmpiinterfaces(mpi_interfaces) ("Actual interface count ($count) does not match "
                                                   * "expectations $(nmpiinterfaces(mpi_interfaces))")
end

