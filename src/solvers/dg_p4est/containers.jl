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
  u::Array{uEltype, NDIMSP2}                  # [primary/secondary, variables, i, j, interface]
  element_ids::Matrix{Int}                    # [primary/secondary, interfaces]
  node_indices::Matrix{NTuple{NDIMS, Symbol}} # [primary/secondary, interfaces]
end

# Create interface container and initialize interface data in `elements`.
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

# Return number of interfaces
@inline ninterfaces(interfaces::InterfaceContainerP4est) = size(interfaces.element_ids, 2)

count_required_interfaces(mesh::P4estMesh) = ndims(mesh) * ncells(mesh)


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
# (:one, :mi) will be (1, size[1] - i + 1).
function indexfunction(indices, size, dim, i, j=0)
  if indices[dim] === :i
    return i
  elseif indices[dim] === :mi
    return size[dim] - i + 1
  elseif indices[dim] === :j
    return j
  elseif indices[dim] === :mj
    return size[dim] - j + 1
  elseif indices[dim] === :one
    return 1
  elseif indices[dim] === :end
    return size[dim]
  end

  error("Invalid identifier: Only :one, :end, :i, :j, :mi, :mj are valid index identifiers")
end

# Remove :one and :end to index surface_flux_values properly (2D version).
#
# Suppose some element is indexed relative to some interface as `(:end, :mi)` (in 2D).
# This interface will be at the right face of the element, and it will be reversely oriented.
# When copying data to the interface, the value `u[v, end, end - i + 1, element]` will be copied
# to `u[2, v, i, interface]` in the interface container (assuming this is a secondary element).
# Now, the calculated flux at interface node i needs to be copied back to
# `surface_flux_values[v, end - i + 1, 2, element]`.
# This is the same index as in `indexfunction` but without the `:one` or `:end`.
#
# Dispatch by dimension to ensure type stability
function indexfunction_surface(indices::NTuple{2, Symbol}, size, dim, i)
  if indices[1] in (:one, :end)
    indices_surface = indices[2:2]
  else # indices[2] in (:one, :end)
    indices_surface = indices[1:1]
  end

  return indexfunction(indices_surface, size, dim, i)
end

# 3D version
function indexfunction_surface(indices::NTuple{3, Symbol}, size, dim, i, j=0)
  if indices[1] in (:one, :end)
    indices_surface = indices[2:3]
  elseif indices[2] in (:one, :end)
    indices_surface = (indices[1], indices[3])
  else # indices[3] in (:one, :end)
    indices_surface = indices[1:2]
  end

  return indexfunction(indices_surface, size, dim, i)
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
