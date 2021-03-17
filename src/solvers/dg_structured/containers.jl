
# TODO: AD, needs to be adapted to use `RealT` and `uEltype`, cf. https://github.com/trixi-framework/Trixi.jl/pull/461
struct Interface{RealT<:Real, NDIMS}
  u_left::Array{RealT, NDIMS} # [variables, i, j]
  u_right::Array{RealT, NDIMS} # [variables, i, j]
  orientation::Int
  surface_flux_values::Array{RealT, NDIMS} # [variables, i, j]
end

function Interface{RealT, NDIMS}(nvars, nnodes, orientation) where {RealT<:Real, NDIMS}
  # TODO Is there a more elegant solution for this?
  u_left = Array{RealT, NDIMS}(undef, nvars, fill(nnodes, NDIMS-1)...)
  u_right = Array{RealT, NDIMS}(undef, nvars, fill(nnodes, NDIMS-1)...)

  surface_flux_values = Array{RealT, NDIMS}(undef, nvars, fill(nnodes, NDIMS-1)...)

  return Interface{RealT, NDIMS}(u_left, u_right, orientation, surface_flux_values)
end

# TODO: AD, needs to be adapted to use `RealT` and `uEltype`, cf. https://github.com/trixi-framework/Trixi.jl/pull/461
struct Element{RealT<:Real, NDIMS}
  node_coordinates::Array{SVector{NDIMS, RealT}, NDIMS}
  # node_coordinates::Array{RealT, 2}
  inverse_jacobian::RealT
  interfaces::Vector{Interface{RealT, NDIMS}} # [orientation]
end

function Element{RealT, NDIMS}(node_coordinates, inverse_jacobian) where {RealT<:Real, NDIMS}
  interfaces = Array{Interface{RealT, NDIMS}}(undef, NDIMS * 2)

  return Element{RealT, NDIMS}(node_coordinates, inverse_jacobian, interfaces)
end


# Create element container and initialize element data
function init_elements(mesh::StructuredMesh{RealT, NDIMS}, equations::AbstractEquations,
    basis::LobattoLegendreBasis{T, NNODES}) where {RealT<:Real, NDIMS, T, NNODES}

  elements = StructArray{Element{RealT, NDIMS}}(undef, mesh.size...)

  init_elements!(elements, mesh, basis.nodes)
  return elements
end

@inline nelements(elements::StructArray) = prod(size(elements))


include("containers_1d.jl")
include("containers_2d.jl")
include("containers_3d.jl")
