import ...Auxiliary.Containers: invalidate!, raw_copy!, move_connectivity!, delete_connectivity!
using ...Auxiliary.Containers: AbstractContainer
import Base


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


# Container data structure (structure-of-arrays style) for DG boundary surfaces
struct BoundaryContainer{V, N} <: AbstractContainer
  u::Array{Float64, 3}                # [variables, i, surfaces]
  neighbor_ids::Vector{Int}           # [surfaces]
  orientations::Vector{Int}           # [surfaces]
  neighbor_sides::Vector{Int}         # [surfaces]
  node_coordinates::Array{Float64, 3} # [orientation, i, elements]
end


function BoundaryContainer{V, N}(capacity::Integer) where {V, N}
  # Initialize fields with defaults
  n_nodes = N + 1
  u = fill(NaN, V, n_nodes, capacity)
  neighbor_ids = fill(typemin(Int), capacity)
  orientations = fill(typemin(Int), capacity)
  neighbor_sides = fill(typemin(Int), capacity)
  node_coordinates = fill(NaN, ndim, n_nodes, capacity)

  boundaries = BoundaryContainer{V, N}(u, neighbor_ids, orientations, neighbor_sides,
                                       node_coordinates)

  return boundaries
end


# Return number of boundaries
nboundaries(boundaries::BoundaryContainer) = length(boundaries.orientations)


# Container data structure (structure-of-arrays style) for DG L2 mortars
# Positions/directions for large_sides = 1, orientations = 1:
#           |    |
# upper = 2 |    |
#           |    |
#                | 3
#           |    |
# lower = 1 |    |
#           |    |
struct L2MortarContainer{V, N} <: AbstractContainer
  u_upper::Array{Float64, 4} # [leftright, variables, i, mortars]
  u_lower::Array{Float64, 4} # [leftright, variables, i, mortars]
  neighbor_ids::Matrix{Int}  # [position, mortars]
  # Large sides: left -> 1, right -> 2
  large_sides::Vector{Int}   # [mortars]
  orientations::Vector{Int}  # [mortars]
end


function L2MortarContainer{V, N}(capacity::Integer) where {V, N}
  # Initialize fields with defaults
  n_nodes = N + 1
  u_upper = fill(NaN, 2, V, n_nodes, capacity)
  u_lower = fill(NaN, 2, V, n_nodes, capacity)
  neighbor_ids = fill(typemin(Int), 3, capacity)
  large_sides = fill(typemin(Int), capacity)
  orientations = fill(typemin(Int), capacity)

  l2mortars = L2MortarContainer{V, N}(u_upper, u_lower, neighbor_ids, large_sides, orientations)

  return l2mortars
end


# Return number of L2 mortars
nmortars(l2mortars::L2MortarContainer) = length(l2mortars.orientations)


# Allow printing container contents
function Base.show(io::IO, c::L2MortarContainer{V, N}) where {V, N}
  println(io, '*'^20)
  for idx in CartesianIndices(c.u_upper)
    println(io, "c.u_upper[$idx] = $(c.u_upper[idx])")
  end
  for idx in CartesianIndices(c.u_lower)
    println(io, "c.u_lower[$idx] = $(c.u_lower[idx])")
  end
  println(io, "transpose(c.neighbor_ids) = $(transpose(c.neighbor_ids))")
  println(io, "c.large_sides = $(c.large_sides)")
  println(io, "c.orientations = $(c.orientations)")
  println(io, '*'^20)
end


# Container data structure (structure-of-arrays style) for DG Ec mortars
# Positions/directions for large_sides = 1, orientations = 1:
#           |    |
# upper = 2 |    |
#           |    |
#                | 3
#           |    |
# lower = 1 |    |
#           |    |
struct EcMortarContainer{V, N} <: AbstractContainer
  u_upper::Array{Float64, 3} # [variables, i, mortars]
  u_lower::Array{Float64, 3} # [variables, i, mortars]
  u_large::Array{Float64, 3} # [variables, i, mortars]
  neighbor_ids::Matrix{Int}  # [position, mortars]
  # Large sides: left -> 1, right -> 2
  large_sides::Vector{Int}   # [mortars]
  orientations::Vector{Int}  # [mortars]
end


function EcMortarContainer{V, N}(capacity::Integer) where {V, N}
  # Initialize fields with defaults
  n_nodes = N + 1
  u_upper = fill(NaN, V, n_nodes, capacity)
  u_lower = fill(NaN, V, n_nodes, capacity)
  u_large = fill(NaN, V, n_nodes, capacity)
  neighbor_ids = fill(typemin(Int), 3, capacity)
  large_sides = fill(typemin(Int), capacity)
  orientations = fill(typemin(Int), capacity)

  ecmortars = EcMortarContainer{V, N}(u_upper, u_lower, u_large, neighbor_ids,
                                      large_sides, orientations)

  return ecmortars
end


# Return number of EC mortars
nmortars(ecmortars::EcMortarContainer) = length(ecmortars.orientations)
