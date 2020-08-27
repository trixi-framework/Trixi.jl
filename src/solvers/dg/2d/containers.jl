
# Container data structure (structure-of-arrays style) for DG elements
struct ElementContainer2D{NVARS, N} <: AbstractContainer
  u::Array{Float64, 4}                   # [variables, i, j, elements]
  u_t::Array{Float64, 4}                 # [variables, i, j, elements]
  u_tmp2::Array{Float64, 4}              # [variables, i, j, elements]
  u_tmp3::Array{Float64, 4}              # [variables, i, j, elements]
  inverse_jacobian::Vector{Float64}      # [elements]
  node_coordinates::Array{Float64, 4}    # [orientation, i, j, elements]
  surface_ids::Matrix{Int}               # [direction, elements]
  surface_flux_values::Array{Float64, 4} # [variables, i, direction, elements]
  cell_ids::Vector{Int}                  # [elements]
end


function ElementContainer2D{NVARS, N}(capacity::Integer) where {NVARS, N} # NVARS = no. variables, N = polydeg
  # Initialize fields with defaults
  n_nodes = N + 1
  u = fill(NaN, NVARS, n_nodes, n_nodes, capacity)
  u_t = fill(NaN, NVARS, n_nodes, n_nodes, capacity)
  # u_rungakutta is initialized to non-NaN since it is used directly
  u_tmp2 = fill(0.0, NVARS, n_nodes, n_nodes, capacity)
  u_tmp3 = fill(0.0, NVARS, n_nodes, n_nodes, capacity)
  inverse_jacobian = fill(NaN, capacity)
  node_coordinates = fill(NaN, 2, n_nodes, n_nodes, capacity)
  surface_ids = fill(typemin(Int), 2 * 2, capacity)
  surface_flux_values = fill(NaN, NVARS, n_nodes, 2 * 2, capacity)
  cell_ids = fill(typemin(Int), capacity)

  elements = ElementContainer2D{NVARS, N}(u, u_t, u_tmp2, u_tmp3, inverse_jacobian, node_coordinates,
                                    surface_ids, surface_flux_values, cell_ids)

  return elements
end


# Return number of elements
nelements(elements::ElementContainer2D) = length(elements.cell_ids)


# Container data structure (structure-of-arrays style) for DG interfaces
struct InterfaceContainer2D{NVARS, N} <: AbstractContainer
  u::Array{Float64, 4}      # [leftright, variables, i, interfaces]
  neighbor_ids::Matrix{Int} # [leftright, interfaces]
  orientations::Vector{Int} # [interfaces]
end


function InterfaceContainer2D{NVARS, N}(capacity::Integer) where {NVARS, N}
  # Initialize fields with defaults
  n_nodes = N + 1
  u = fill(NaN, 2, NVARS, n_nodes, capacity)
  neighbor_ids = fill(typemin(Int), 2, capacity)
  orientations = fill(typemin(Int), capacity)

  interfaces = InterfaceContainer2D{NVARS, N}(u, neighbor_ids, orientations)

  return interfaces
end


# Return number of interfaces
ninterfaces(interfaces::InterfaceContainer2D) = length(interfaces.orientations)


# Container data structure (structure-of-arrays style) for DG boundaries
struct BoundaryContainer2D{NVARS, N} <: AbstractContainer
  u::Array{Float64, 4}                # [leftright, variables, i, boundaries]
  neighbor_ids::Vector{Int}           # [boundaries]
  orientations::Vector{Int}           # [boundaries]
  neighbor_sides::Vector{Int}         # [boundaries]
  node_coordinates::Array{Float64, 3} # [orientation, i, elements]
end


function BoundaryContainer2D{NVARS, N}(capacity::Integer) where {NVARS, N}
  # Initialize fields with defaults
  n_nodes = N + 1
  u = fill(NaN, 2, NVARS, n_nodes, capacity)
  neighbor_ids = fill(typemin(Int), capacity)
  orientations = fill(typemin(Int), capacity)
  neighbor_sides = fill(typemin(Int), capacity)
  node_coordinates = fill(NaN, 2, n_nodes, capacity)

  boundaries = BoundaryContainer2D{NVARS, N}(u, neighbor_ids, orientations, neighbor_sides,
                                       node_coordinates)

  return boundaries
end


# Return number of boundaries
nboundaries(boundaries::BoundaryContainer2D) = length(boundaries.orientations)


# Container data structure (structure-of-arrays style) for DG L2 mortars
# Positions/directions for orientations = 1, large_sides = 2:
# mortar is orthogonal to x-axis, large side is in positive coordinate direction wrt mortar
#           |    |
# upper = 2 |    |
#           |    |
#                | 3 = large side
#           |    |
# lower = 1 |    |
#           |    |
struct L2MortarContainer2D{NVARS, N} <: AbstractContainer
  u_upper::Array{Float64, 4} # [leftright, variables, i, mortars]
  u_lower::Array{Float64, 4} # [leftright, variables, i, mortars]
  neighbor_ids::Matrix{Int}  # [position, mortars]
  # Large sides: left -> 1, right -> 2
  large_sides::Vector{Int}   # [mortars]
  orientations::Vector{Int}  # [mortars]
end


function L2MortarContainer2D{NVARS, N}(capacity::Integer) where {NVARS, N}
  # Initialize fields with defaults
  n_nodes = N + 1
  u_upper = fill(NaN, 2, NVARS, n_nodes, capacity)
  u_lower = fill(NaN, 2, NVARS, n_nodes, capacity)
  neighbor_ids = fill(typemin(Int), 3, capacity)
  large_sides = fill(typemin(Int), capacity)
  orientations = fill(typemin(Int), capacity)

  l2mortars = L2MortarContainer2D{NVARS, N}(u_upper, u_lower, neighbor_ids, large_sides, orientations)

  return l2mortars
end


# Return number of L2 mortars
nmortars(l2mortars::L2MortarContainer2D) = length(l2mortars.orientations)


# Allow printing container contents
function Base.show(io::IO, c::L2MortarContainer2D{NVARS, N}) where {NVARS, N}
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
struct EcMortarContainer2D{NVARS, N} <: AbstractContainer
  u_upper::Array{Float64, 3} # [variables, i, mortars]
  u_lower::Array{Float64, 3} # [variables, i, mortars]
  u_large::Array{Float64, 3} # [variables, i, mortars]
  neighbor_ids::Matrix{Int}  # [position, mortars]
  # Large sides: left -> 1, right -> 2
  large_sides::Vector{Int}   # [mortars]
  orientations::Vector{Int}  # [mortars]
end


function EcMortarContainer2D{NVARS, N}(capacity::Integer) where {NVARS, N}
  # Initialize fields with defaults
  n_nodes = N + 1
  u_upper = fill(NaN, NVARS, n_nodes, capacity)
  u_lower = fill(NaN, NVARS, n_nodes, capacity)
  u_large = fill(NaN, NVARS, n_nodes, capacity)
  neighbor_ids = fill(typemin(Int), 3, capacity)
  large_sides = fill(typemin(Int), capacity)
  orientations = fill(typemin(Int), capacity)

  ecmortars = EcMortarContainer2D{NVARS, N}(u_upper, u_lower, u_large, neighbor_ids,
                                      large_sides, orientations)

  return ecmortars
end


# Return number of EC mortars
nmortars(ecmortars::EcMortarContainer2D) = length(ecmortars.orientations)
