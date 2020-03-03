import ...Auxiliary.Containers: invalidate!, raw_copy!, move_connectivity!, delete_connectivity!
using ... Auxiliary.Containers: AbstractContainer


# Container data structure (structure-of-arrays style) for DG elements
struct ElementContainer{V, N} <: AbstractContainer
  u::Array{Float64, 4}                # [variables, i, j, elements]
  u_t::Array{Float64, 4}              # [variables, i, j, elements]
  u_rungekutta::Array{Float64, 4}     # [variables, i, j, elements]
  inverse_jacobian::Vector{Float64}   # [elements]
  node_coordinates::Array{Float64, 4} # [orientation, i, j, elements]
  surface_ids::Matrix{Int}            # [direction, elements]
  surface_flux::Array{Float64, 4}     # [variables, i, direction, elements]
  cell_ids::Vector{Int}               # [elements]
end


function ElementContainer{V, N}(capacity::Integer) where {V, N} # V = no. variables, N = polydeg
  # Initialize fields with defaults
  n_nodes = N + 1
  u = fill(NaN, V, n_nodes, n_nodes, capacity)
  u_t = fill(NaN, V, n_nodes, n_nodes, capacity)
  # u_rungakutta is initialized to non-NaN since it is used directly
  u_rungekutta = fill(0.0, V, n_nodes, n_nodes, capacity)
  inverse_jacobian = fill(NaN, capacity)
  node_coordinates = fill(NaN, ndim, n_nodes, n_nodes, capacity)
  surface_ids = fill(typemin(Int), 2 * ndim, capacity)
  surface_flux = fill(NaN, V, n_nodes, 2 * ndim, capacity)
  cell_ids = fill(typemin(Int), capacity)

  elements = ElementContainer{V, N}(u, u_t, u_rungekutta, inverse_jacobian, node_coordinates,
                                    surface_ids, surface_flux, cell_ids)

  return elements
end


# Return number of elements
nelements(elements::ElementContainer) = length(elements.cell_ids)


# Container data structure (structure-of-arrays style) for DG surfaces
struct SurfaceContainer{V, N} <: AbstractContainer
  u::Array{Float64, 4}      # [leftright, variables, i, surfaces]
  neighbor_ids::Matrix{Int} # [leftright, surfaces]
  orientations::Vector{Int} # [surfaces]
end


function SurfaceContainer{V, N}(capacity::Integer) where {V, N}
  # Initialize fields with defaults
  n_nodes = N + 1
  u = fill(NaN, 2, V, n_nodes, capacity)
  neighbor_ids = fill(typemin(Int), 2, capacity)
  orientations = fill(typemin(Int), capacity)

  surfaces = SurfaceContainer{V, N}(u, neighbor_ids, orientations)

  return surfaces
end


# Return number of surfaces
nsurfaces(surfaces::SurfaceContainer) = length(surfaces.orientations)
