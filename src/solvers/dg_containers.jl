import ...Auxiliary.Containers: invalidate!, raw_copy!, move_connectivity!, delete_connectivity!
using ... Auxiliary.Containers: AbstractContainer
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


# Container data structure (structure-of-arrays style) for DG mpi_surfaces
struct MpiSurfaceContainer{V, N} <: AbstractContainer
  u::Array{Float64, 4}      # [leftright, variables, i, mpi_surfaces]
  element_ids::Vector{Int} # [mpi_surfaces]
  element_sides::Vector{Int} # [mpi_surfaces]
  neighbor_cell_ids::Vector{Int} # [mpi_surfaces]
  orientations::Vector{Int} # [mpi_surfaces]
end


function MpiSurfaceContainer{V, N}(capacity::Integer) where {V, N}
  # Initialize fields with defaults
  n_nodes = N + 1
  u = fill(NaN, 2, V, n_nodes, capacity)
  element_ids = fill(typemin(Int), capacity)
  # Element sides: left -> 1, right -> 2
  element_sides = fill(typemin(Int), capacity)
  neighbor_cell_ids = fill(typemin(Int), capacity)
  orientations = fill(typemin(Int), capacity)

  mpi_surfaces = MpiSurfaceContainer{V, N}(u, element_ids, element_sides,
                                           neighbor_cell_ids, orientations)

  return mpi_surfaces
end


# Return number of mpi_surfaces
nmpi_surfaces(mpi_surfaces::MpiSurfaceContainer) = length(mpi_surfaces.orientations)


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


# Return number of mortars
nl2mortars(l2mortars::L2MortarContainer) = length(l2mortars.orientations)

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
