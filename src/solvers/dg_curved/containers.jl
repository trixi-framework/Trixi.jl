# Note: This is an experimental feature and may be changed in future releases without notice.
struct ElementContainer{NDIMS, RealT<:Real, uEltype<:Real, NDIMSP1, NDIMSP2, NDIMSP3}
  node_coordinates::Array{RealT, NDIMSP2} # [orientation, i, j, k, elements]
  left_neighbors::Array{Int, 2} # [orientation, elements]
  # Jacobian matrix of the transformation
  # [jacobian_i, jacobian_j, node_i, node_j, node_k, element] where jacobian_i is the first index of the Jacobian matrix,...
  metric_terms::Array{RealT, NDIMSP3}
  inverse_jacobian::Array{RealT, NDIMSP1} # [i, j, k, elements]
  surface_flux_values::Array{uEltype, NDIMSP2} # [variables, i, j, direction, elements]
end


# Create element container and initialize element data
function init_elements(mesh::CurvedMesh{NDIMS, RealT},
                       equations::AbstractEquations,
                       basis, ::Type{uEltype}) where {NDIMS, RealT<:Real, uEltype<:Real}

  nelements = prod(size(mesh))
  node_coordinates = Array{RealT, NDIMS+2}(undef, NDIMS, ntuple(_ -> nnodes(basis), NDIMS)..., nelements)
  left_neighbors = Array{Int, 2}(undef, NDIMS, nelements)
  metric_terms = Array{RealT, NDIMS+3}(undef, NDIMS, NDIMS, ntuple(_ -> nnodes(basis), NDIMS)..., nelements)
  inverse_jacobian = Array{RealT, NDIMS+1}(undef, ntuple(_ -> nnodes(basis), NDIMS)..., nelements)
  surface_flux_values = Array{uEltype, NDIMS+2}(undef,
      nvariables(equations), ntuple(_ -> nnodes(basis), NDIMS-1)..., NDIMS*2, nelements)

  elements = ElementContainer{NDIMS, RealT, uEltype, NDIMS+1, NDIMS+2, NDIMS+3}(
      node_coordinates, left_neighbors, metric_terms,
      inverse_jacobian, surface_flux_values)

  init_elements!(elements, mesh, basis)
  return elements
end

@inline nelements(elements::ElementContainer) = size(elements.left_neighbors, 2)

Base.eltype(::ElementContainer{NDIMS, RealT, uEltype}) where {NDIMS, RealT, uEltype} = uEltype


include("containers_1d.jl")
include("containers_2d.jl")
include("containers_3d.jl")
