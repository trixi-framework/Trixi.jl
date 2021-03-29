# Note: This is an experimental feature and may be changed in future releases without notice.
struct ElementContainer{NDIMS, RealT<:Real, uEltype<:Real, NDIMSP2}
  node_coordinates::Array{RealT, NDIMSP2} # [orientation, i, j, k, elements]
  left_neighbors::Array{Int, 2} # [orientation, elements]
  inverse_jacobian::Vector{RealT} # [elements]
  surface_flux_values::Array{uEltype, NDIMSP2} # [variables, i, j, direction, elements]
end


# Create element container and initialize element data
function init_elements(mesh::StructuredMesh{NDIMS, RealT}, 
                       equations::AbstractEquations,
                       basis, ::Type{uEltype}) where {NDIMS, RealT<:Real, uEltype<:Real}

  nelements = prod(size(mesh))
  node_coordinates = Array{RealT, NDIMS+2}(undef, NDIMS, ntuple(_ -> nnodes(basis), NDIMS)..., nelements)
  left_neighbors = Array{Int, 2}(undef, NDIMS, nelements)
  inverse_jacobian = Vector{RealT}(undef, nelements)
  surface_flux_values = Array{uEltype, NDIMS+2}(undef, 
      nvariables(equations), ntuple(_ -> nnodes(basis), NDIMS-1)..., NDIMS*2, nelements)

  elements = ElementContainer{NDIMS, RealT, uEltype, NDIMS+2}(node_coordinates, left_neighbors, inverse_jacobian, surface_flux_values)

  init_elements!(elements, mesh, basis.nodes)
  return elements
end

@inline nelements(elements::ElementContainer) = size(elements.left_neighbors, 2)

Base.eltype(::ElementContainer{NDIMS, RealT, uEltype}) where {NDIMS, RealT, uEltype} = uEltype


include("containers_1d.jl")
include("containers_2d.jl")
include("containers_3d.jl")
