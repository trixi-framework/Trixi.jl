mutable struct T8codeElementContainer{NDIMS, RealT<:Real, uEltype<:Real, NDIMSP1, NDIMSP2, NDIMSP3} <: AbstractContainer
  # Physical coordinates at each node
  node_coordinates          :: Array{RealT, NDIMSP2}   # [orientation, node_i, node_j, node_k, element]
  # Jacobian matrix of the transformation
  # [jacobian_i, jacobian_j, node_i, node_j, node_k, element] where jacobian_i is the first index of the Jacobian matrix,...
  jacobian_matrix           :: Array{RealT, NDIMSP3}
  # Contravariant vectors, scaled by J, in Kopriva's blue book called Ja^i_n (i index, n dimension)
  contravariant_vectors     :: Array{RealT, NDIMSP3}   # [dimension, index, node_i, node_j, node_k, element]
  # 1/J where J is the Jacobian determinant (determinant of Jacobian matrix)
  inverse_jacobian          :: Array{RealT, NDIMSP1}   # [node_i, node_j, node_k, element]
  # Buffer for calculated surface flux
  surface_flux_values       :: Array{uEltype, NDIMSP2} # [variable, i, j, direction, element]

  # internal `resize!`able storage
  _node_coordinates         :: Vector{RealT}
  _jacobian_matrix          :: Vector{RealT}
  _contravariant_vectors    :: Vector{RealT}
  _inverse_jacobian         :: Vector{RealT}
  _surface_flux_values      :: Vector{uEltype}
end

mutable struct T8codeInterfaceContainer{NDIMS, uEltype<:Real, NDIMSP2} <: AbstractContainer
  u             :: Array{uEltype, NDIMSP2}       # [primary/secondary, variable, i, j, interface]
  neighbor_ids  :: Matrix{Int}                   # [primary/secondary, interface]
  node_indices  :: Matrix{NTuple{NDIMS, Symbol}} # [primary/secondary, interface]

  # Internal `resize!`able storage.
  _u            :: Vector{uEltype}
  _neighbor_ids :: Vector{Int}
  _node_indices :: Vector{NTuple{NDIMS, Symbol}}
end

mutable struct T8codeMortarContainer{NDIMS, uEltype<:Real, NDIMSP1, NDIMSP3} <: AbstractContainer
  u             :: Array{uEltype, NDIMSP3}        # [small/large side, variable, position, i, j, mortar]
  neighbor_ids  :: Matrix{Int}                    # [position, mortar]
  node_indices  :: Matrix{NTuple{NDIMS, Symbol}}  # [small/large, mortar]

  # internal `resize!`able storage
  _u            :: Vector{uEltype}
  _neighbor_ids :: Vector{Int}
  _node_indices :: Vector{NTuple{NDIMS, Symbol}}
end

mutable struct T8codeBoundaryContainer{NDIMS, uEltype<:Real, NDIMSP1} <: AbstractContainer
  u             :: Array{uEltype, NDIMSP1}       # [variables, i, j, boundary]
  neighbor_ids  :: Vector{Int}                   # [boundary]
  node_indices  :: Vector{NTuple{NDIMS, Symbol}} # [boundary]
  name          :: Vector{Symbol}                # [boundary]

  # Internal `resize!`able storage.
  _u            :: Vector{uEltype}
end

# ============================================================================ #
# ============================================================================ #

@inline nelements(elements::T8codeElementContainer) = size(elements.node_coordinates, ndims(elements) + 2)
@inline Base.ndims(::T8codeElementContainer{NDIMS}) where NDIMS = NDIMS
@inline Base.eltype(::T8codeElementContainer{NDIMS, RealT, uEltype}) where {NDIMS, RealT, uEltype} = uEltype

@inline ninterfaces(interfaces::T8codeInterfaceContainer) = size(interfaces.neighbor_ids, 2)
@inline Base.ndims(::T8codeInterfaceContainer{NDIMS}) where NDIMS = NDIMS

@inline nboundaries(boundaries::T8codeBoundaryContainer) = length(boundaries.neighbor_ids)
@inline Base.ndims(::T8codeBoundaryContainer{NDIMS}) where NDIMS = NDIMS

@inline nmortars(mortars::T8codeMortarContainer) = size(mortars.neighbor_ids, 2)
@inline Base.ndims(::T8codeMortarContainer{NDIMS}) where NDIMS = NDIMS

# ============================================================================ #
# ============================================================================ #

function reinitialize_containers!(mesh::T8codeMesh, equations, dg::DGSEM, cache)

  # Re-initialize elements container.
  @unpack elements = cache
  resize!(elements, ncells(mesh))
  init_elements!(elements, mesh, dg.basis)

  required = count_required_surfaces(mesh)

  # Resize interfaces container.
  @unpack interfaces = cache
  resize!(interfaces, required.interfaces)

  # Resize mortars container.
  @unpack mortars = cache
  resize!(mortars, required.mortars)

  # Resize boundaries container.
  @unpack boundaries = cache
  resize!(boundaries, required.boundaries)

  # Re-initialize containers together to reduce
  # the number of iterations over the mesh in `t8code`.
  # init_surfaces!(elements,interfaces, mortars, boundaries, mesh)

  trixi_t8_fill_mesh_info(mesh.forest, elements, interfaces, mortars, boundaries, mesh.boundary_names)

  return nothing
end

# ============================================================================ #
# ============================================================================ #

# Only one-dimensional `Array`s are `resize!`able in Julia.
# Hence, we use `Vector`s as internal storage and `resize!`
# them whenever needed. Then, we reuse the same memory by
# `unsafe_wrap`ping multi-dimensional `Array`s around the
# internal storage.
function Base.resize!(elements::T8codeElementContainer, capacity)
  @unpack _node_coordinates, _jacobian_matrix, _contravariant_vectors,
    _inverse_jacobian, _surface_flux_values = elements

  n_dims = ndims(elements)
  n_nodes = size(elements.node_coordinates, 2)
  n_variables = size(elements.surface_flux_values, 1)

  resize!(_node_coordinates, n_dims * n_nodes^n_dims * capacity)
  elements.node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
    (n_dims, ntuple(_ -> n_nodes, n_dims)..., capacity))

  resize!(_jacobian_matrix, n_dims^2 * n_nodes^n_dims * capacity)
  elements.jacobian_matrix = unsafe_wrap(Array, pointer(_jacobian_matrix),
    (n_dims, n_dims, ntuple(_ -> n_nodes, n_dims)..., capacity))

  resize!(_contravariant_vectors, length(_jacobian_matrix))
  elements.contravariant_vectors = unsafe_wrap(Array, pointer(_contravariant_vectors),
    size(elements.jacobian_matrix))

  resize!(_inverse_jacobian, n_nodes^n_dims * capacity)
  elements.inverse_jacobian = unsafe_wrap(Array, pointer(_inverse_jacobian),
    (ntuple(_ -> n_nodes, n_dims)..., capacity))

  resize!(_surface_flux_values,
    n_variables * n_nodes^(n_dims-1) * (n_dims*2) * capacity)
  elements.surface_flux_values = unsafe_wrap(Array, pointer(_surface_flux_values),
    (n_variables, ntuple(_ -> n_nodes, n_dims-1)..., n_dims*2, capacity))

  return nothing
end

# See explanation of Base.resize! for the element container.
function Base.resize!(interfaces::T8codeInterfaceContainer, capacity)
  @unpack _u, _neighbor_ids, _node_indices = interfaces

  n_dims = ndims(interfaces)
  n_nodes = size(interfaces.u, 3)
  n_variables = size(interfaces.u, 2)

  resize!(_u, 2 * n_variables * n_nodes^(n_dims-1) * capacity)
  interfaces.u = unsafe_wrap(Array, pointer(_u),
    (2, n_variables, ntuple(_ -> n_nodes, n_dims-1)..., capacity))

  resize!(_neighbor_ids, 2 * capacity)
  interfaces.neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids), (2, capacity))

  resize!(_node_indices, n_dims * capacity)
  interfaces.node_indices = unsafe_wrap(Array, pointer(_node_indices), (n_dims, capacity))

  return nothing
end

# See explanation of Base.resize! for the element container.
function Base.resize!(mortars::T8codeMortarContainer, capacity)
  @unpack _u, _neighbor_ids, _node_indices = mortars

  n_dims = ndims(mortars)
  n_nodes = size(mortars.u, 4)
  n_variables = size(mortars.u, 2)

  resize!(_u, 2 * n_variables * 2^(n_dims-1) * n_nodes^(n_dims-1) * capacity)
  mortars.u = unsafe_wrap(Array, pointer(_u),
    (2, n_variables, 2^(n_dims-1), ntuple(_ -> n_nodes, n_dims-1)..., capacity))

  resize!(_neighbor_ids, (2^(n_dims-1) + 1) * capacity)
  mortars.neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids),
    (2^(n_dims-1) + 1, capacity))

  resize!(_node_indices, n_dims * capacity)
  mortars.node_indices = unsafe_wrap(Array, pointer(_node_indices), (n_dims, capacity))

  return nothing
end

# See explanation of Base.resize! for the element container.
function Base.resize!(boundaries::T8codeBoundaryContainer, capacity)
  @unpack _u, neighbor_ids, node_indices, name = boundaries

  n_dims = ndims(boundaries)
  n_nodes = size(boundaries.u, 2)
  n_variables = size(boundaries.u, 1)

  resize!(_u, n_variables * n_nodes^(n_dims-1) * capacity)
  boundaries.u = unsafe_wrap(Array, pointer(_u),
    (n_variables, ntuple(_ -> n_nodes, n_dims-1)..., capacity))

  resize!(neighbor_ids, capacity)

  resize!(node_indices, capacity)

  resize!(name, capacity)

  return nothing
end

# ============================================================================ #
# ============================================================================ #

# Create element container and initialize element data.
function init_elements(mesh::T8codeMesh{NDIMS,RealT}, equations,
                       basis, ::Type{uEltype}) where {NDIMS, RealT<:Real, uEltype<:Real}
  nelements = ncells(mesh)

  _node_coordinates = Vector{RealT}(undef, NDIMS * nnodes(basis)^NDIMS * nelements)
  
  node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
    (NDIMS, ntuple(_ -> nnodes(basis), NDIMS)..., nelements))

  _jacobian_matrix = Vector{RealT}(undef, NDIMS^2 * nnodes(basis)^NDIMS * nelements)
  jacobian_matrix = unsafe_wrap(Array, pointer(_jacobian_matrix),
    (NDIMS, NDIMS, ntuple(_ -> nnodes(basis), NDIMS)..., nelements))

  _contravariant_vectors = similar(_jacobian_matrix)
  contravariant_vectors = unsafe_wrap(Array, pointer(_contravariant_vectors),
    size(jacobian_matrix))

  _inverse_jacobian = Vector{RealT}(undef, nnodes(basis)^NDIMS * nelements)
  inverse_jacobian = unsafe_wrap(Array, pointer(_inverse_jacobian),
    (ntuple(_ -> nnodes(basis), NDIMS)..., nelements))

  _surface_flux_values = Vector{uEltype}(undef,
    nvariables(equations) * nnodes(basis)^(NDIMS-1) * (NDIMS*2) * nelements)
  surface_flux_values = unsafe_wrap(Array, pointer(_surface_flux_values),
    (nvariables(equations), ntuple(_ -> nnodes(basis), NDIMS-1)..., NDIMS*2, nelements))

  elements = T8codeElementContainer{NDIMS, RealT, uEltype, NDIMS+1, NDIMS+2, NDIMS+3}(
    node_coordinates, jacobian_matrix, contravariant_vectors,
    inverse_jacobian, surface_flux_values,
    _node_coordinates, _jacobian_matrix, _contravariant_vectors,
    _inverse_jacobian, _surface_flux_values)

  # Implementation is found in 'containers_{2,3}d.jl'.
  init_elements!(elements, mesh, basis)
  return elements
end

# Create interface container and initialize interface data.
function init_interfaces(mesh::T8codeMesh, equations, basis, elements)
  NDIMS = ndims(elements)
  uEltype = eltype(elements)

  n_interfaces = ninterfaces(mesh)

  _u = Vector{uEltype}(undef, 2 * nvariables(equations) * nnodes(basis)^(NDIMS-1) * n_interfaces)
  u = unsafe_wrap(Array, pointer(_u),
    (2, nvariables(equations), ntuple(_ -> nnodes(basis), NDIMS-1)..., n_interfaces))

  _neighbor_ids = Vector{Int}(undef, 2 * n_interfaces)
  neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids), (2, n_interfaces))

  _node_indices = Vector{NTuple{NDIMS, Symbol}}(undef, NDIMS * n_interfaces)
  node_indices = unsafe_wrap(Array, pointer(_node_indices), (NDIMS, n_interfaces))

  interfaces = T8codeInterfaceContainer{NDIMS, uEltype, NDIMS+2}(u, neighbor_ids, node_indices,
                                                                _u, _neighbor_ids, _node_indices)

  return interfaces
end

# Create mortar container and initialize mortar data.
function init_mortars(mesh::T8codeMesh, equations, basis, elements)
  NDIMS = ndims(elements)
  uEltype = eltype(elements)

  n_mortars = nmortars(mesh)

  _u = Vector{uEltype}(undef,
    2 * nvariables(equations) * 2^(NDIMS-1) * nnodes(basis)^(NDIMS-1) * n_mortars)
  u = unsafe_wrap(Array, pointer(_u),
    (2, nvariables(equations), 2^(NDIMS-1), ntuple(_ -> nnodes(basis), NDIMS-1)..., n_mortars))

  _neighbor_ids = Vector{Int}(undef, (2^(NDIMS-1) + 1) * n_mortars)
  neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids), (2^(NDIMS-1) + 1, n_mortars))

  _node_indices = Vector{NTuple{NDIMS, Symbol}}(undef, NDIMS * n_mortars)
  node_indices = unsafe_wrap(Array, pointer(_node_indices), (NDIMS, n_mortars))

  mortars = T8codeMortarContainer{NDIMS, uEltype, NDIMS+1, NDIMS+3}(u, neighbor_ids, node_indices,
                                                                   _u, _neighbor_ids, _node_indices)

  return mortars
end

# Create interface container and initialize interface data in `elements`.
function init_boundaries(mesh::T8codeMesh, equations, basis, elements)
  NDIMS = ndims(elements)
  uEltype = eltype(elements)

  # Initialize container.
  n_boundaries = nboundaries(mesh)

  _u = Vector{uEltype}(undef, nvariables(equations) * nnodes(basis)^(NDIMS-1) * n_boundaries)
  u = unsafe_wrap(Array, pointer(_u),
    (nvariables(equations), ntuple(_ -> nnodes(basis), NDIMS-1)..., n_boundaries))

  neighbor_ids = Vector{Int}(undef, n_boundaries)
  node_indices = Vector{NTuple{NDIMS, Symbol}}(undef, n_boundaries)
  names        = Vector{Symbol}(undef, n_boundaries)

  boundaries = T8codeBoundaryContainer{NDIMS, uEltype, NDIMS+1}(u, neighbor_ids,
                                                               node_indices, names, _u)

  return boundaries
end

# ============================================================================ #
# ============================================================================ #

function count_required_surfaces(mesh::T8codeMesh)

  counts = trixi_t8_count_interfaces(mesh.forest)

  mesh.nmortars    = counts.mortars
  mesh.ninterfaces = counts.interfaces
  mesh.nboundaries = counts.boundaries

  return counts
end
