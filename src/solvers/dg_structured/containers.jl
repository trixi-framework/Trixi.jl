struct Interface{NDIMS, RealT<:Real}
  u_left::Array{RealT, NDIMS} # [variables, i, j]
  u_right::Array{RealT, NDIMS} # [variables, i, j]
  orientation::Int64
  surface_flux_values::Array{RealT, NDIMS} # [variables, i, j]
end

function Interface{NDIMS, RealT}(nvars, nnodes, orientation) where {NDIMS, RealT<:Real}
  # TODO Is there a more elegant solution for this?
  u_left = Array{RealT, NDIMS}(undef, nvars, fill(nnodes, NDIMS-1)...)
  u_right = Array{RealT, NDIMS}(undef, nvars, fill(nnodes, NDIMS-1)...)

  surface_flux_values = Array{RealT, NDIMS}(undef, nvars, fill(nnodes, NDIMS-1)...)

  return Interface{NDIMS, RealT}(u_left, u_right, orientation, surface_flux_values)
end


struct Element{NDIMS, RealT<:Real}
  node_coordinates::Array{SVector{NDIMS, RealT}, NDIMS}
  # node_coordinates::Array{RealT, 2}
  inverse_jacobian::RealT
  interfaces::Vector{Interface{NDIMS, RealT}} # [orientation]
end

function Element{NDIMS, RealT}(node_coordinates, inverse_jacobian) where {NDIMS, RealT<:Real}
  interfaces = Array{Interface{NDIMS, RealT}}(undef, NDIMS * 2)

  return Element{NDIMS, RealT}(node_coordinates, inverse_jacobian, interfaces)
end


# Create element container and initialize element data
function init_elements(mesh::StructuredMesh{NDIMS, RealT}, equations::AbstractEquations,
    basis::LobattoLegendreBasis{T, NNODES}) where {NDIMS, RealT<:Real, T, NNODES}

  elements = StructArray{Element{NDIMS, RealT}}(undef, mesh.size...)

  init_elements!(elements, mesh, basis.nodes)
  return elements
end

@inline nelements(elements::StructArray) = prod(size(elements))


include("containers_1d.jl")
include("containers_2d.jl")
include("containers_3d.jl")
