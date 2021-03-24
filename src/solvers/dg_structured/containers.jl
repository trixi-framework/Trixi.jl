# TODO: AD, needs to be adapted to use `RealT` and `uEltype`, cf. https://github.com/trixi-framework/Trixi.jl/pull/461

struct Interface{NDIMS, RealT<:Real}
  left_element::Int
  right_element::Int
  orientation::Int
  surface_flux_values::Array{RealT, NDIMS} # [variables, i, j]
end

function Interface{NDIMS, RealT}(left_element, right_element, orientation, nvars, nnodes) where {NDIMS, RealT<:Real}
  # Dimension independent version of (undef, nvars, nnodes, nnodes, nnodes)
  surface_flux_values = Array{RealT, NDIMS}(undef, nvars, ntuple(_ -> nnodes, NDIMS-1)...)

  return Interface{NDIMS, RealT}(left_element, right_element, orientation, surface_flux_values)
end


# TODO: AD, needs to be adapted to use `RealT` and `uEltype`, cf. https://github.com/trixi-framework/Trixi.jl/pull/461

struct Element{NDIMS, RealT<:Real, NDIMSP1}
  node_coordinates::Array{RealT, NDIMSP1}
  inverse_jacobian::RealT
  interfaces::Vector{Interface{NDIMS, RealT}} # [orientation]
end

function Element(node_coordinates, inverse_jacobian)
  RealT = eltype(node_coordinates)
  NDIMS = size(node_coordinates, 1)

  interfaces = Array{Interface{NDIMS, RealT}}(undef, NDIMS * 2)

  return Element{NDIMS, RealT, NDIMS+1}(node_coordinates, inverse_jacobian, interfaces)
end


# Create element container and initialize element data
function init_elements(mesh::StructuredMesh{NDIMS, RealT}, equations::AbstractEquations,
    basis) where {NDIMS, RealT<:Real}

  elements = StructArray{Element{NDIMS, RealT, NDIMS+1}}(undef, size(mesh)...)

  init_elements!(elements, mesh, basis.nodes)
  return elements
end

@inline nelements(elements::StructArray) = prod(size(elements))

# TODO: AD, needs to be adapted to use `RealT` and `uEltype`, cf. https://github.com/trixi-framework/Trixi.jl/pull/461
Base.eltype(::StructArray{Element{NDIMS, RealT, NDIMSP1}}) where {NDIMS, RealT, NDIMSP1} = RealT


include("containers_1d.jl")
include("containers_2d.jl")
include("containers_3d.jl")
