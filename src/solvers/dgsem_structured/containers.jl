# Note: This is an experimental feature and may be changed in future releases without notice.
struct ElementContainer{NDIMS, RealT<:Real, uEltype<:Real, NDIMSP1, NDIMSP2, NDIMSP3}
  # Physical coordinates at each node
  node_coordinates      ::Array{RealT, NDIMSP2}   # [orientation, node_i, node_j, node_k, element]
  # ID of neighbor element in negative direction in orientation
  left_neighbors        ::Array{Int, 2}           # [orientation, elements]
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
function init_elements(mesh::StructuredMesh{NDIMS, RealT},
                       equations::AbstractEquations,
                       basis, ::Type{uEltype}) where {NDIMS, RealT<:Real, uEltype<:Real}

  nelements = prod(size(mesh))
  node_coordinates      = Array{RealT, NDIMS+2}(undef, NDIMS, ntuple(_ -> nnodes(basis), NDIMS)..., nelements)
  left_neighbors        = Array{Int, 2}(undef, NDIMS, nelements)
  jacobian_matrix       = Array{RealT, NDIMS+3}(undef, NDIMS, NDIMS, ntuple(_ -> nnodes(basis), NDIMS)..., nelements)
  contravariant_vectors = similar(jacobian_matrix)
  inverse_jacobian      = Array{RealT, NDIMS+1}(undef, ntuple(_ -> nnodes(basis), NDIMS)..., nelements)
  surface_flux_values   = Array{uEltype, NDIMS+2}(undef, nvariables(equations),
                                                  ntuple(_ -> nnodes(basis), NDIMS-1)..., NDIMS*2, nelements)

  elements = ElementContainer{NDIMS, RealT, uEltype, NDIMS+1, NDIMS+2, NDIMS+3}(
      node_coordinates, left_neighbors, jacobian_matrix, contravariant_vectors,
      inverse_jacobian, surface_flux_values)

  init_elements!(elements, mesh, basis)
  return elements
end

@inline nelements(elements::ElementContainer) = size(elements.left_neighbors, 2)
@inline Base.ndims(::ElementContainer{NDIMS}) where NDIMS = NDIMS

Base.eltype(::ElementContainer{NDIMS, RealT, uEltype}) where {NDIMS, RealT, uEltype} = uEltype


include("containers_1d.jl")
include("containers_2d.jl")
include("containers_3d.jl")
