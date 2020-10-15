
# Container data structure (structure-of-arrays style) for DG elements
# TODO: Taal refactor, remove u, u_t, u_tmp2, u_tmp3
# TODO: Taal refactor, remove NVARS, POLYDEG?
mutable struct ElementContainer2D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  u::Array{RealT, 4}                   # [variables, i, j, elements]
  u_t::Array{RealT, 4}                 # [variables, i, j, elements]
  u_tmp2::Array{RealT, 4}              # [variables, i, j, elements]
  u_tmp3::Array{RealT, 4}              # [variables, i, j, elements]
  inverse_jacobian::Vector{RealT}      # [elements]
  node_coordinates::Array{RealT, 4}    # [orientation, i, j, elements]
  surface_flux_values::Array{RealT, 4} # [variables, i, direction, elements]
  cell_ids::Vector{Int}                # [elements]
  # internal `resize!`able storage
  _u::Vector{RealT}
  _u_t::Vector{RealT}
  _u_tmp2::Vector{RealT}
  _u_tmp3::Vector{RealT}
  _node_coordinates::Vector{RealT}
  _surface_flux_values::Vector{RealT}
end

function Base.resize!(elements::ElementContainer2D{RealT, NVARS, POLYDEG},
                      capacity) where {RealT, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  @unpack _u, _u_t, _u_tmp2, _u_tmp3, _node_coordinates, _surface_flux_values,
          inverse_jacobian, cell_ids = elements

  resize!(_u, NVARS * n_nodes * n_nodes * capacity)
  elements.u = unsafe_wrap(Array, pointer(_u),
                           (NVARS, n_nodes, n_nodes, capacity))

  resize!(_u_t, NVARS * n_nodes * n_nodes * capacity)
  elements.u_t = unsafe_wrap(Array, pointer(_u_t),
                             (NVARS, n_nodes, n_nodes, capacity))

  resize!(_u_tmp2, NVARS * n_nodes * n_nodes * capacity)
  _u_tmp2 .= zero(eltype(_u_tmp2))
  elements.u_tmp2 = unsafe_wrap(Array, pointer(_u_tmp2),
                                (NVARS, n_nodes, n_nodes, capacity))

  resize!(_u_tmp3, NVARS * n_nodes * n_nodes * capacity)
  _u_tmp3 .= zero(eltype(_u_tmp3))
  elements.u_tmp3 = unsafe_wrap(Array, pointer(_u_tmp3),
                                (NVARS, n_nodes, n_nodes, capacity))

  resize!(inverse_jacobian, capacity)

  resize!(_node_coordinates, 2 * n_nodes * n_nodes * capacity)
  elements.node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                          (2, n_nodes, n_nodes, capacity))

  resize!(_surface_flux_values, NVARS * n_nodes * 2 * 2 * capacity)
  elements.surface_flux_values = unsafe_wrap(Array, pointer(_surface_flux_values),
                                             (NVARS, n_nodes, 2 * 2, capacity))

  resize!(cell_ids, capacity)

  return nothing
end


function ElementContainer2D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  _u = fill(nan, NVARS * n_nodes * n_nodes * capacity)
  u = unsafe_wrap(Array, pointer(_u),
                  (NVARS, n_nodes, n_nodes, capacity))

  _u_t = fill(nan, NVARS * n_nodes * n_nodes * capacity)
  u_t = unsafe_wrap(Array, pointer(_u_t),
                    (NVARS, n_nodes, n_nodes, capacity))

  # u_rungakutta is initialized to non-NaN since it is used directly
  _u_tmp2 = fill(zero(RealT), NVARS * n_nodes * n_nodes * capacity)
  u_tmp2 = unsafe_wrap(Array, pointer(_u_tmp2),
                       (NVARS, n_nodes, n_nodes, capacity))

  _u_tmp3 = fill(zero(RealT), NVARS * n_nodes * n_nodes * capacity)
  u_tmp3 = unsafe_wrap(Array, pointer(_u_tmp3),
                       (NVARS, n_nodes, n_nodes, capacity))

  inverse_jacobian = fill(nan, capacity)

  _node_coordinates = fill(nan, 2 * n_nodes * n_nodes * capacity)
  node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                 (2, n_nodes, n_nodes, capacity))

  _surface_flux_values = fill(nan, NVARS * n_nodes * 2 * 2 * capacity)
  surface_flux_values = unsafe_wrap(Array, pointer(_surface_flux_values),
                                    (NVARS, n_nodes, 2 * 2, capacity))

  cell_ids = fill(typemin(Int), capacity)


  return ElementContainer2D{RealT, NVARS, POLYDEG}(
    u, u_t, u_tmp2, u_tmp3,
    inverse_jacobian, node_coordinates, surface_flux_values, cell_ids,
    _u, _u_t, _u_tmp2, _u_tmp3, _node_coordinates, _surface_flux_values)
end


# Return number of elements
@inline nelements(elements::ElementContainer2D) = length(elements.cell_ids)


# Container data structure (structure-of-arrays style) for DG interfaces
# TODO: Taal refactor, remove NVARS, POLYDEG?
mutable struct InterfaceContainer2D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  u::Array{RealT, 4}        # [leftright, variables, i, interfaces]
  neighbor_ids::Matrix{Int} # [leftright, interfaces]
  orientations::Vector{Int} # [interfaces]
  # internal `resize!`able storage
  _u::Vector{RealT}
  _neighbor_ids::Vector{Int}
end

function Base.resize!(interfaces::InterfaceContainer2D{RealT, NVARS, POLYDEG},
                      capacity) where {RealT, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  @unpack _u, _neighbor_ids, orientations = interfaces

  resize!(_u, 2 * NVARS * n_nodes * capacity)
  interfaces.u = unsafe_wrap(Array, pointer(_u),
                             (2, NVARS, n_nodes, capacity))

  resize!(_neighbor_ids, 2 * capacity)
  interfaces.neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids ),
                                        (2, capacity))

  resize!(orientations, capacity)

  return nothing
end


function InterfaceContainer2D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  _u = fill(nan, 2 * NVARS * n_nodes * capacity)
  u = unsafe_wrap(Array, pointer(_u),
                  (2, NVARS, n_nodes, capacity))

  _neighbor_ids = fill(typemin(Int), 2 * capacity)
  neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids ),
                             (2, capacity))

  orientations = fill(typemin(Int), capacity)


  return InterfaceContainer2D{RealT, NVARS, POLYDEG}(
    u, neighbor_ids, orientations,
    _u, _neighbor_ids)
end


# Return number of interfaces
@inline ninterfaces(interfaces::InterfaceContainer2D) = length(interfaces.orientations)


# Container data structure (structure-of-arrays style) for DG boundaries
# TODO: Taal refactor, remove NVARS, POLYDEG?
mutable struct BoundaryContainer2D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  u::Array{RealT, 4}                # [leftright, variables, i, boundaries]
  neighbor_ids::Vector{Int}         # [boundaries]
  orientations::Vector{Int}         # [boundaries]
  neighbor_sides::Vector{Int}       # [boundaries]
  node_coordinates::Array{RealT, 3} # [orientation, i, elements]
  n_boundaries_per_direction::SVector{4, Int} # [direction]
  # internal `resize!`able storage
  _u::Vector{RealT}
  _node_coordinates::Vector{RealT}
end

function Base.resize!(boundaries::BoundaryContainer2D{RealT, NVARS, POLYDEG},
                      capacity) where {RealT, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  @unpack _u, _node_coordinates,
          neighbor_ids, orientations, neighbor_sides = boundaries

  resize!(_u, 2 * NVARS * n_nodes * capacity)
  boundaries.u = unsafe_wrap(Array, pointer(_u),
                             (2, NVARS, n_nodes, capacity))

  resize!(_node_coordinates, 2 * n_nodes * capacity)
  boundaries.node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates ),
                                            (2, n_nodes, capacity))

  resize!(neighbor_ids, capacity)

  resize!(orientations, capacity)

  resize!(neighbor_sides, capacity)

  return nothing
end


function BoundaryContainer2D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  _u = fill(nan, 2 * NVARS * n_nodes * capacity)
  u = unsafe_wrap(Array, pointer(_u),
                  (2, NVARS, n_nodes, capacity))

  neighbor_ids = fill(typemin(Int), capacity)

  orientations = fill(typemin(Int), capacity)

  neighbor_sides = fill(typemin(Int), capacity)

  _node_coordinates = fill(nan, 2 * n_nodes * capacity)
  node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                 (2, n_nodes, capacity))

  n_boundaries_per_direction = SVector(0, 0, 0, 0)

  boundaries = BoundaryContainer2D{RealT, NVARS, POLYDEG}(
    u, neighbor_ids, orientations, neighbor_sides,
    node_coordinates, n_boundaries_per_direction,
    _u, _node_coordinates)

  return boundaries
end


# Return number of boundaries
@inline nboundaries(boundaries::BoundaryContainer2D) = length(boundaries.orientations)


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
  # internal `resize!`able storage
  _u_upper::Vector{RealT}
  _u_lower::Vector{RealT}
  _neighbor_ids::Vector{Int}
end

function Base.resize!(mortars::L2MortarContainer2D{RealT, NVARS, POLYDEG},
                      capacity) where {RealT, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  @unpack _u_upper, _u_lower, _neighbor_ids,
          large_sides, orientations = mortars

  resize!(_u_upper, 2 * NVARS * n_nodes * capacity)
  mortars.u_upper = unsafe_wrap(Array, pointer(_u_upper),
                                (2, NVARS, n_nodes, capacity))

  resize!(_u_lower, 2 * NVARS * n_nodes * capacity)
  mortars.u_lower = unsafe_wrap(Array, pointer(_u_lower),
                                (2, NVARS, n_nodes, capacity))

  resize!(_neighbor_ids, 3 * capacity)
  mortars.neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids ),
                                        (3, capacity))

  resize!(large_sides, capacity)

  resize!(orientations, capacity)

  return nothing
end


function L2MortarContainer2D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  _u_upper = fill(nan, 2 * NVARS * n_nodes * capacity)
  u_upper = unsafe_wrap(Array, pointer(_u_upper),
                        (2, NVARS, n_nodes, capacity))

  _u_lower = fill(nan, 2 * NVARS * n_nodes * capacity)
  u_lower = unsafe_wrap(Array, pointer(_u_lower),
                        (2, NVARS, n_nodes, capacity))

  _neighbor_ids = fill(typemin(Int), 3 * capacity)
  neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids),
                             (3, capacity))

  large_sides  = fill(typemin(Int), capacity)

  orientations = fill(typemin(Int), capacity)

  return L2MortarContainer2D{RealT, NVARS, POLYDEG}(
    u_upper, u_lower, neighbor_ids, large_sides, orientations,
    _u_upper, _u_lower, _neighbor_ids)
end


# Return number of L2 mortars
@inline nmortars(l2mortars::L2MortarContainer2D) = length(l2mortars.orientations)


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

  ecmortars = EcMortarContainer2D{RealT, NVARS, POLYDEG}(
    u_upper, u_lower, u_large, neighbor_ids, large_sides, orientations)

  return ecmortars
end


# Return number of EC mortars
nmortars(ecmortars::EcMortarContainer2D) = length(ecmortars.orientations)
