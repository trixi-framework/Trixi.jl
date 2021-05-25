# Note: This is an experimental feature and may be changed in future releases without notice.
struct ElementContainerP4est{NDIMS, RealT<:Real, uEltype<:Real, NDIMSP1, NDIMSP2, NDIMSP3}
  # Physical coordinates at each node
  node_coordinates      ::Array{RealT, NDIMSP2}   # [orientation, node_i, node_j, node_k, element]
  # Jacobian matrix of the transformation
  # [jacobian_i, jacobian_j, node_i, node_j, node_k, element] where jacobian_i is the first index of the Jacobian matrix,...
  jacobian_matrix       ::Array{RealT, NDIMSP3}
  # Contravariant vectors, scaled by J, in Kopriva's blue book called Ja^i_n (i index, n dimension)
  contravariant_vectors ::Array{RealT, NDIMSP3}   # [dimension, index, node_i, node_j, node_k, element]
  # 1/J where J is the Jacobian determinant (determinant of Jacobian matrix)
  inverse_jacobian      ::Array{RealT, NDIMSP1}   # [node_i, node_j, node_k, element]
  # Buffer for calculated surface flux
  surface_flux_values   ::Array{uEltype, NDIMSP2} # [variable, i, j, direction, element]
end


# Create element container and initialize element data
function init_elements(mesh::P4estMesh{NDIMS, RealT}, equations,
                       basis, ::Type{uEltype}) where {NDIMS, RealT<:Real, uEltype<:Real}

  nelements = ncells(mesh)
  node_coordinates      = Array{RealT, NDIMS+2}(undef, NDIMS, ntuple(_ -> nnodes(basis), NDIMS)..., nelements)
  jacobian_matrix       = Array{RealT, NDIMS+3}(undef, NDIMS, NDIMS, ntuple(_ -> nnodes(basis), NDIMS)..., nelements)
  contravariant_vectors = similar(jacobian_matrix)
  inverse_jacobian      = Array{RealT, NDIMS+1}(undef, ntuple(_ -> nnodes(basis), NDIMS)..., nelements)
  surface_flux_values   = Array{uEltype, NDIMS+2}(undef, nvariables(equations),
                                                  ntuple(_ -> nnodes(basis), NDIMS-1)..., NDIMS*2, nelements)

  elements = ElementContainerP4est{NDIMS, RealT, uEltype, NDIMS+1, NDIMS+2, NDIMS+3}(
      node_coordinates, jacobian_matrix, contravariant_vectors,
      inverse_jacobian, surface_flux_values)

  init_elements!(elements, mesh, basis)
  return elements
end

@inline nelements(elements::ElementContainerP4est) = size(elements.node_coordinates, ndims(elements) + 2)
@inline Base.ndims(::ElementContainerP4est{NDIMS}) where NDIMS = NDIMS

Base.eltype(::ElementContainerP4est{NDIMS, RealT, uEltype}) where {NDIMS, RealT, uEltype} = uEltype


struct InterfaceContainerP4est{NDIMS, uEltype<:Real, NDIMSP2} <: AbstractContainer
  u::Array{uEltype, NDIMSP2}                  # [primary/secondary, variable, i, j, interface]
  element_ids::Matrix{Int}                    # [primary/secondary, interface]
  node_indices::Matrix{NTuple{NDIMS, Symbol}} # [primary/secondary, interface]
end

# Create interface container and initialize interface data.
function init_interfaces(mesh::P4estMesh, equations, basis,
                         elements::ElementContainerP4est{NDIMS, RealT, uEltype}
                         ) where {NDIMS, RealT<:Real, uEltype<:Real}
  # Initialize container
  n_interfaces = count_required_interfaces(mesh)

  u = Array{uEltype, NDIMS+2}(undef, 2, nvariables(equations),
                              ntuple(_ -> nnodes(basis), NDIMS-1)...,
                              n_interfaces)
  element_ids = Matrix{Int}(undef, 2, n_interfaces)
  node_indices = Matrix{NTuple{NDIMS, Symbol}}(undef, 2, n_interfaces)

  interfaces = InterfaceContainerP4est{NDIMS, uEltype, NDIMS+2}(u, element_ids, node_indices)

  init_interfaces!(interfaces, mesh)

  return interfaces
end

@inline ninterfaces(interfaces::InterfaceContainerP4est) = size(interfaces.element_ids, 2)


struct BoundaryContainerP4est{NDIMS, uEltype<:Real, NDIMSP1} <: AbstractContainer
  u::Array{uEltype, NDIMSP1}                  # [variables, i, j, boundary]
  element_ids::Vector{Int}                    # [boundary]
  node_indices::Vector{NTuple{NDIMS, Symbol}} # [boundary]
  name::Vector{Symbol}                        # [boundary]
end

# Create interface container and initialize interface data in `elements`.
function init_boundaries(mesh::P4estMesh, equations, basis,
                         elements::ElementContainerP4est{NDIMS, RealT, uEltype}
                         ) where {NDIMS, RealT<:Real, uEltype<:Real}
  # Initialize container
  n_boundaries = count_required_boundaries(mesh)

  u = Array{uEltype, NDIMS+1}(undef, nvariables(equations),
                              ntuple(_ -> nnodes(basis), NDIMS-1)...,
                              n_boundaries)
  element_ids = Vector{Int}(undef, n_boundaries)
  node_indices = Vector{NTuple{NDIMS, Symbol}}(undef, n_boundaries)
  names = Vector{Symbol}(undef, n_boundaries)

  boundaries = BoundaryContainerP4est{NDIMS, uEltype, NDIMS+1}(u, element_ids, node_indices, names)

  init_boundaries!(boundaries, mesh)

  return boundaries
end

@inline nboundaries(boundaries::BoundaryContainerP4est) = length(boundaries.element_ids)


struct MortarContainerP4est{NDIMS, uEltype<:Real, NDIMSP3} <: AbstractContainer
  u::Array{uEltype, NDIMSP3}  # [small/large side, position, variable, i, j, mortar]
  element_ids::Matrix{Int}    # [position, mortar]
  node_indices::Matrix{NTuple{NDIMS, Symbol}} # [small/large, mortar]
end

# Create mortar container and initialize mortar data.
function init_mortars(mesh::P4estMesh, equations, basis,
                      elements::ElementContainerP4est{NDIMS, RealT, uEltype}
                      ) where {NDIMS, RealT<:Real, uEltype<:Real}
  # Initialize container
  n_mortars = count_required_mortars(mesh)

  u = Array{uEltype, NDIMS+3}(undef, 3, 2^(NDIMS-1),
                              nvariables(equations),
                              ntuple(_ -> nnodes(basis), NDIMS-1)...,
                              n_mortars)
  element_ids = Matrix{Int}(undef, 2^(NDIMS-1), n_mortars)
  node_indices = Matrix{NTuple{NDIMS, Symbol}}(undef, 2, n_mortars)

  mortars = MortarContainerP4est{NDIMS, uEltype, NDIMS+3}(u, element_ids, node_indices)

  init_mortars!(mortars, mesh)

  return mortars
end

@inline nmortars(mortars::MortarContainerP4est) = size(mortars.element_ids, 2)


# Iterate over all interfaces and count inner interfaces
function count_interfaces_iter_face(info, user_data)
  if info.sides.elem_count == 2
    # Extract interface data
    sides = convert_sc_array(p4est_iter_face_side_t, info.sides)

    if sides[1].is_hanging == 0 && sides[2].is_hanging == 0
      # No hanging nodes => normal interface
      # Unpack user_data = [interface_count] and increment interface_count
      ptr = Ptr{Int}(user_data)
      data_array = unsafe_wrap(Array, ptr, 1)
      data_array[1] += 1
    end
  end

  return nothing
end

# Iterate over all interfaces and count boundaries
function count_boundaries_iter_face(info, user_data)
  if info.sides.elem_count == 1
    # Unpack user_data = [boundary_count] and increment boundary_count
    ptr = Ptr{Int}(user_data)
    data_array = unsafe_wrap(Array, ptr, 1)
    data_array[1] += 1
  end

  return nothing
end

# Iterate over all interfaces and count mortars
function count_mortars_iter_face(info, user_data)
  if info.sides.elem_count == 2
    # Extract interface data
    sides = convert_sc_array(p4est_iter_face_side_t, info.sides)

    if sides[1].is_hanging != 0 || sides[2].is_hanging != 0
      # Hanging nodes => mortar
      # Unpack user_data = [mortar_count] and increment mortar_count
      ptr = Ptr{Int}(user_data)
      data_array = unsafe_wrap(Array, ptr, 1)
      data_array[1] += 1
    end
  end

  return nothing
end

function count_required_interfaces(mesh::P4estMesh)
  # Let p4est iterate over all interfaces and call count_interfaces_iter_face
  iter_face_c = @cfunction(count_interfaces_iter_face, Cvoid, (Ptr{p4est_iter_face_info_t}, Ptr{Cvoid}))

  return count_required(mesh, iter_face_c)
end

function count_required_boundaries(mesh::P4estMesh)
  # Let p4est iterate over all interfaces and call count_boundaries_iter_face
  iter_face_c = @cfunction(count_boundaries_iter_face, Cvoid, (Ptr{p4est_iter_face_info_t}, Ptr{Cvoid}))

  return count_required(mesh, iter_face_c)
end

function count_required_mortars(mesh::P4estMesh)
  # Let p4est iterate over all interfaces and call count_mortars_iter_face
  iter_face_c = @cfunction(count_mortars_iter_face, Cvoid, (Ptr{p4est_iter_face_info_t}, Ptr{Cvoid}))

  return count_required(mesh, iter_face_c)
end

function count_required(mesh::P4estMesh, iter_face_c)
  # Counter
  user_data = [0]

  iterate_faces(mesh, iter_face_c, user_data)

  # Return counter
  return user_data[1]
end

# Let p4est iterate over all interfaces and execute the C function iter_face_c
function iterate_faces(mesh::P4estMesh, iter_face_c, user_data)
  GC.@preserve user_data begin
    p4est_iterate(mesh.p4est,
                  C_NULL, # ghost layer
                  pointer(user_data),
                  C_NULL, # iter_volume
                  iter_face_c, # iter_face
                  C_NULL) # iter_corner
  end

  return nothing
end


# Convert sc_array to Julia array of the specified type
function convert_sc_array(::Type{T}, sc_array) where T
  element_count = sc_array.elem_count
  element_size = sc_array.elem_size

  @assert element_size == sizeof(T)

  return [unsafe_wrap(T, sc_array.array + element_size * i) for i in 0:element_count-1]
end


# Convert Tuple node_indices to actual indices.
# E.g., (:one, :i, :j) will be (1, i, j) for some i and j,
# (:i, :end) will be (i, size[2]),
# (:one, :i_backwards) will be (1, size[1] - i + 1).
function evaluate_index(indices, size, dim, i, j=0)
  if indices[dim] === :i
    return i
  elseif indices[dim] === :i_backwards
    return size[dim] - i + 1
  elseif indices[dim] === :j
    return j
  elseif indices[dim] === :j_backwards
    return size[dim] - j + 1
  elseif indices[dim] === :one
    return 1
  elseif indices[dim] === :end
    return size[dim]
  end

  error("Invalid identifier: Only :one, :end, :i, :j, :i_backwards, :j_backwards are valid index identifiers")
end

# Remove :one and :end to index surface_flux_values properly (2D version).
#
# Suppose some element is indexed relative to some interface as `(:end, :i_backwards)` (in 2D).
# This interface will be at the right face of the element, and it will be reversely oriented.
# When copying data to the interface, the value `u[v, end, end - i + 1, element]` will be copied
# to `u[2, v, i, interface]` in the interface container (assuming this is a secondary element).
# Now, the calculated flux at interface node i needs to be copied back to
# `surface_flux_values[v, end - i + 1, 2, element]`.
# This is the same index as in `evaluate_index` but without the `:one` or `:end`.
#
# Dispatch by dimension to ensure type stability
@inline function evaluate_index_surface(indices::NTuple{2, Symbol}, size, dim, i)
  if indices[1] in (:one, :end)
    indices_surface = indices[2:2]
  else # indices[2] in (:one, :end)
    indices_surface = indices[1:1]
  end

  return evaluate_index(indices_surface, size, dim, i)
end

# 3D version
@inline function evaluate_index_surface(indices::NTuple{3, Symbol}, size, dim, i, j=0)
  if indices[1] in (:one, :end)
    indices_surface = indices[2:3]
  elseif indices[2] in (:one, :end)
    indices_surface = (indices[1], indices[3])
  else # indices[3] in (:one, :end)
    indices_surface = indices[1:2]
  end

  return evaluate_index(indices_surface, size, dim, i)
end

# Return direction of the face, which is indexed by node_indices
@inline function indices2direction(indices)
  if indices[1] in (:one, :end)
    orientation = 1
  elseif indices[2] in (:one, :end)
    orientation = 2
  else # indices[3] in (:one, :end)
    orientation = 3
  end
  negative_direction = orientation * 2 - 1

  if indices[orientation] === :one
    return negative_direction
  else # indices[orientation] === :end
    return negative_direction + 1
  end
end


include("containers_2d.jl")
