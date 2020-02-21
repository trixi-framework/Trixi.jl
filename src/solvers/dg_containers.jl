import ...Auxiliary.Containers: invalidate!, raw_copy!, move_connectivity!, delete_connectivity!
using ... Auxiliary.Containers: AbstractContainer


# Container data structure (structure-of-arrays style) for DG elements
struct ElementContainer{V, N} <: AbstractContainer
  u::Array{Float64, 3}
  u_t::Array{Float64, 3}
  u_rungekutta::Array{Float64, 3}
  flux::Array{Float64, 3}
  inverse_jacobian::Vector{Float64}
  node_coordinates::Matrix{Float64}
  surface_ids::Matrix{Int}
  cell_ids::Vector{Int}
end


function ElementContainer{V, N}(capacity::Integer) where {V, N}
  # Initialize fields with defaults
  u = fill(NaN, V, N + 1, capacity)
  u_t = fill(NaN, V, N + 1, capacity)
  u_rungekutta = fill(0.0, V, N + 1, capacity) # Initialized to non-NaN since it is used directly
  flux = fill(NaN, V, N + 1, capacity)
  inverse_jacobian = fill(NaN, capacity)
  node_coordinates = fill(NaN, N + 1, capacity)
  surface_ids = fill(typemin(Int), 2, capacity)
  cell_ids = fill(typemin(Int), capacity)

  elements = ElementContainer{V, N}(u, u_t, u_rungekutta, flux,
                                    inverse_jacobian, node_coordinates, surface_ids, cell_ids)

  return elements
end


# Return number of elements
nelements(elements::ElementContainer) = length(elements.inverse_jacobian)


# Container data structure (structure-of-arrays style) for DG surfaces
struct SurfaceContainer{V, N} <: AbstractContainer
  u::Array{Float64, 3}
  flux::Matrix{Float64}
  neighbor_ids::Matrix{Int}
end


function SurfaceContainer{V, N}(capacity::Integer) where {V, N}
  # Initialize fields with defaults
  u = fill(NaN, 2, V, capacity)
  flux = fill(NaN, V, capacity)
  neighbor_ids = fill(typemin(Int), 2, capacity)

  surfaces = SurfaceContainer{V, N}(u, flux, neighbor_ids)

  return surfaces
end


# Return number of surfaces
nsurfaces(surfaces::SurfaceContainer) = size(surfaces.neighbor_ids)[2]
