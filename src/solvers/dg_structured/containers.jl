# TODO: AD, needs to be adapted to use `RealT` and `uEltype`, cf. https://github.com/trixi-framework/Trixi.jl/pull/461
#
# Note: This is an experimental feature and may be changed in future releases without notice.
struct ElementContainer{NDIMS, RealT<:Real, NDIMSP2}
  node_coordinates::Array{RealT, NDIMSP2} # [orientation, i, j, k, elements]
  left_neighbors::Array{Int, 2} # [orientation, elements]
  inverse_jacobian::Vector{RealT} # [elements]
  surface_flux_values::Array{RealT, NDIMSP2} # [variables, i, j, direction, elements]
end


# Create element container and initialize element data
function init_elements(mesh::StructuredMesh{NDIMS, RealT}, equations::AbstractEquations,
    basis) where {NDIMS, RealT<:Real}

  nelements = prod(size(mesh))
  node_coordinates = Array{RealT, NDIMS+2}(undef, NDIMS, ntuple(_ -> nnodes(basis), NDIMS)..., nelements)
  left_neighbors = Array{Int, 2}(undef, NDIMS, nelements)
  inverse_jacobian = Vector{RealT}(undef, nelements)
  surface_flux_values = Array{RealT, NDIMS+2}(undef, 
      nvariables(equations), ntuple(_ -> nnodes(basis), NDIMS-1)..., NDIMS*2, nelements)

  elements = ElementContainer{NDIMS, RealT, NDIMS+2}(node_coordinates, left_neighbors, inverse_jacobian, surface_flux_values)

  init_elements!(elements, mesh, basis.nodes)
  return elements
end

@inline nelements(elements::ElementContainer) = size(elements.left_neighbors, 2)

# TODO: AD, needs to be adapted to use `RealT` and `uEltype`, cf. https://github.com/trixi-framework/Trixi.jl/pull/461
Base.eltype(::ElementContainer{NDIMS, RealT}) where {NDIMS, RealT} = RealT


include("containers_1d.jl")
include("containers_2d.jl")
include("containers_3d.jl")
