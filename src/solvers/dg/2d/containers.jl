
# Container data structure (structure-of-arrays style) for DG elements
# TODO: Taal refactor, remove u, u_t, u_tmp2, u_tmp3
# TODO: Taal refactor, remove NVARS, POLYDEG?
# TODO: Taal refactor, mutable struct or resize! for AMR?
# TODO: Taal refactor, surface_ids does not seem to be used anywhere?
mutable struct ElementContainer2D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  u::Array{RealT, 4}                   # [variables, i, j, elements]
  u_t::Array{RealT, 4}                 # [variables, i, j, elements]
  u_tmp2::Array{RealT, 4}              # [variables, i, j, elements]
  u_tmp3::Array{RealT, 4}              # [variables, i, j, elements]
  inverse_jacobian::Vector{RealT}      # [elements]
  node_coordinates::Array{RealT, 4}    # [orientation, i, j, elements]
  surface_ids::Matrix{Int}             # [direction, elements]
  surface_flux_values::Array{RealT, 4} # [variables, i, direction, elements]
  cell_ids::Vector{Int}                # [elements]
end

function Base.copy!(dst::ElementContainer2D, src::ElementContainer2D)
  dst.u                   = src.u
  dst.u_t                 = src.u_t
  dst.u_tmp2              = src.u_tmp2
  dst.u_tmp3              = src.u_tmp3
  dst.inverse_jacobian    = src.inverse_jacobian
  dst.node_coordinates    = src.node_coordinates
  dst.surface_ids         = src.surface_ids
  dst.surface_flux_values = src.surface_flux_values
  dst.cell_ids            = src.cell_ids
  return nothing
end


function ElementContainer2D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  u = fill(nan, NVARS, n_nodes, n_nodes, capacity)
  u_t = fill(nan, NVARS, n_nodes, n_nodes, capacity)
  # u_rungakutta is initialized to non-NaN since it is used directly
  u_tmp2 = fill(zero(RealT), NVARS, n_nodes, n_nodes, capacity)
  u_tmp3 = fill(zero(RealT), NVARS, n_nodes, n_nodes, capacity)

  inverse_jacobian = fill(nan, capacity)
  node_coordinates = fill(nan, 2, n_nodes, n_nodes, capacity)
  surface_ids = fill(typemin(Int), 2 * 2, capacity)
  surface_flux_values = fill(nan, NVARS, n_nodes, 2 * 2, capacity)
  cell_ids = fill(typemin(Int), capacity)

  elements = ElementContainer2D{RealT, NVARS, POLYDEG}(u, u_t, u_tmp2, u_tmp3, inverse_jacobian, node_coordinates,
                                                surface_ids, surface_flux_values, cell_ids)

  return elements
end


# Return number of elements
nelements(elements::ElementContainer2D) = length(elements.cell_ids)


# Container data structure (structure-of-arrays style) for DG interfaces
# TODO: Taal refactor, remove NVARS, POLYDEG?
# TODO: Taal refactor, mutable struct or resize! for AMR?
mutable struct InterfaceContainer2D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  u::Array{RealT, 4}        # [leftright, variables, i, interfaces]
  neighbor_ids::Matrix{Int} # [leftright, interfaces]
  orientations::Vector{Int} # [interfaces]
end

function Base.copy!(dst::InterfaceContainer2D, src::InterfaceContainer2D)
  dst.u            = src.u
  dst.neighbor_ids = src.neighbor_ids
  dst.orientations = src.orientations
  return nothing
end


function InterfaceContainer2D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  u = fill(nan, 2, NVARS, n_nodes, capacity)
  neighbor_ids = fill(typemin(Int), 2, capacity)
  orientations = fill(typemin(Int), capacity)

  interfaces = InterfaceContainer2D{RealT, NVARS, POLYDEG}(u, neighbor_ids, orientations)

  return interfaces
end


# Return number of interfaces
ninterfaces(interfaces::InterfaceContainer2D) = length(interfaces.orientations)


# Container data structure (structure-of-arrays style) for DG boundaries
# TODO: Taal refactor, remove NVARS, POLYDEG?
# TODO: Taal refactor, mutable struct or resize! for AMR?
mutable struct BoundaryContainer2D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  u::Array{RealT, 4}                # [leftright, variables, i, boundaries]
  neighbor_ids::Vector{Int}         # [boundaries]
  orientations::Vector{Int}         # [boundaries]
  neighbor_sides::Vector{Int}       # [boundaries]
  node_coordinates::Array{RealT, 3} # [orientation, i, elements]
end

function Base.copy!(dst::BoundaryContainer2D, src::BoundaryContainer2D)
  dst.u                = src.u
  dst.neighbor_ids     = src.neighbor_ids
  dst.orientations     = src.orientations
  dst.neighbor_sides   = src.neighbor_sides
  dst.node_coordinates = src.node_coordinates
  return nothing
end


function BoundaryContainer2D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  u = fill(nan, 2, NVARS, n_nodes, capacity)
  neighbor_ids = fill(typemin(Int), capacity)
  orientations = fill(typemin(Int), capacity)
  neighbor_sides = fill(typemin(Int), capacity)
  node_coordinates = fill(nan, 2, n_nodes, capacity)

  boundaries = BoundaryContainer2D{RealT, NVARS, POLYDEG}(u, neighbor_ids, orientations, neighbor_sides,
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
# TODO: Taal refactor, remove NVARS, POLYDEG?
# TODO: Taal refactor, mutable struct or resize! for AMR?
mutable struct L2MortarContainer2D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  u_upper::Array{RealT, 4}  # [leftright, variables, i, mortars]
  u_lower::Array{RealT, 4}  # [leftright, variables, i, mortars]
  neighbor_ids::Matrix{Int} # [position, mortars]
  # Large sides: left -> 1, right -> 2
  large_sides::Vector{Int}  # [mortars]
  orientations::Vector{Int} # [mortars]
end

function Base.copy!(dst::L2MortarContainer2D, src::L2MortarContainer2D)
  dst.u_upper      = src.u_upper
  dst.u_lower      = src.u_lower
  dst.neighbor_ids = src.neighbor_ids
  dst.large_sides  = src.large_sides
  dst.orientations = src.orientations
  return nothing
end


function L2MortarContainer2D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  u_upper = fill(nan, 2, NVARS, n_nodes, capacity)
  u_lower = fill(nan, 2, NVARS, n_nodes, capacity)
  neighbor_ids = fill(typemin(Int), 3, capacity)
  large_sides  = fill(typemin(Int), capacity)
  orientations = fill(typemin(Int), capacity)

  l2mortars = L2MortarContainer2D{RealT, NVARS, POLYDEG}(u_upper, u_lower, neighbor_ids, large_sides, orientations)

  return l2mortars
end


# Return number of L2 mortars
nmortars(l2mortars::L2MortarContainer2D) = length(l2mortars.orientations)


# Allow printing container contents
function Base.show(io::IO, ::MIME"text/plain", c::L2MortarContainer2D)
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
  print(io,   '*'^20)
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
# TODO: Taal refactor, remove NVARS, POLYDEG?
struct EcMortarContainer2D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  u_upper::Array{RealT, 3}  # [variables, i, mortars]
  u_lower::Array{RealT, 3}  # [variables, i, mortars]
  u_large::Array{RealT, 3}  # [variables, i, mortars]
  neighbor_ids::Matrix{Int} # [position, mortars]
  # Large sides: left -> 1, right -> 2
  large_sides::Vector{Int}  # [mortars]
  orientations::Vector{Int} # [mortars]
end


function EcMortarContainer2D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  u_upper = fill(nan, NVARS, n_nodes, capacity)
  u_lower = fill(nan, NVARS, n_nodes, capacity)
  u_large = fill(nan, NVARS, n_nodes, capacity)
  neighbor_ids = fill(typemin(Int), 3, capacity)
  large_sides  = fill(typemin(Int), capacity)
  orientations = fill(typemin(Int), capacity)

  ecmortars = EcMortarContainer2D{RealT, NVARS, POLYDEG}(u_upper, u_lower, u_large, neighbor_ids,
                                                  large_sides, orientations)

  return ecmortars
end


# Return number of EC mortars
nmortars(ecmortars::EcMortarContainer2D) = length(ecmortars.orientations)
