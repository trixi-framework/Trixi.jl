
# Container data structure (structure-of-arrays style) for DG elements
# TODO: Taal refactor, remove u, u_t, u_tmp2, u_tmp3
# TODO: Taal refactor, remove NVARS, POLYDEG?
mutable struct ElementContainer3D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  u::Array{RealT, 5}                   # [variables, i, j, k, elements]
  u_t::Array{RealT, 5}                 # [variables, i, j, k, elements]
  u_tmp2::Array{RealT, 5}              # [variables, i, j, k, elements]
  u_tmp3::Array{RealT, 5}              # [variables, i, j, k, elements]
  inverse_jacobian::Vector{RealT}      # [elements]
  node_coordinates::Array{RealT, 5}    # [orientation, i, j, k, elements]
  surface_flux_values::Array{RealT, 5} # [variables, i, j, direction, elements]
  cell_ids::Vector{Int}                # [elements]
  # internal `resize!`able storage
  _u::Vector{RealT}
  _u_t::Vector{RealT}
  _u_tmp2::Vector{RealT}
  _u_tmp3::Vector{RealT}
  _node_coordinates::Vector{RealT}
  _surface_flux_values::Vector{RealT}
end

# Only one-dimensional `Array`s are `resize!`able in Julia.
# Hence, we use `Vector`s as internal storage and `resize!`
# them whenever needed. Then, we reuse the same memory by
# `unsafe_wrap`ping multi-dimensional `Array`s around the
# internal storage.
function Base.resize!(elements::ElementContainer3D{RealT, NVARS, POLYDEG},
                      capacity) where {RealT, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  @unpack _u, _u_t, _u_tmp2, _u_tmp3, _node_coordinates, _surface_flux_values,
          inverse_jacobian, cell_ids = elements

  resize!(_u, NVARS * n_nodes * n_nodes * n_nodes * capacity)
  elements.u = unsafe_wrap(Array, pointer(_u),
                           (NVARS, n_nodes, n_nodes, n_nodes, capacity))

  resize!(_u_t, NVARS * n_nodes * n_nodes * n_nodes * capacity)
  elements.u_t = unsafe_wrap(Array, pointer(_u_t),
                             (NVARS, n_nodes, n_nodes, n_nodes, capacity))

  resize!(_u_tmp2, NVARS * n_nodes * n_nodes * n_nodes * capacity)
  elements.u_tmp2 = unsafe_wrap(Array, pointer(_u_tmp2),
                                (NVARS, n_nodes, n_nodes, n_nodes, capacity))

  resize!(_u_tmp3, NVARS * n_nodes * n_nodes * n_nodes * capacity)
  elements.u_tmp3 = unsafe_wrap(Array, pointer(_u_tmp3),
                                (NVARS, n_nodes, n_nodes, n_nodes, capacity))

  resize!(inverse_jacobian, capacity)

  resize!(_node_coordinates, 3 * n_nodes * n_nodes * n_nodes * capacity)
  elements.node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                          (3, n_nodes, n_nodes, n_nodes, capacity))

  resize!(_surface_flux_values, NVARS * n_nodes * n_nodes * 2 * 3 * capacity)
  elements.surface_flux_values = unsafe_wrap(Array, pointer(_surface_flux_values),
                                             (NVARS, n_nodes, n_nodes, 2 * 3, capacity))

  resize!(cell_ids, capacity)

  return nothing
end


function ElementContainer3D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  _u = fill(nan, NVARS * n_nodes * n_nodes * n_nodes * capacity)
  u = unsafe_wrap(Array, pointer(_u),
                  (NVARS, n_nodes, n_nodes, n_nodes, capacity))

  _u_t = fill(nan, NVARS * n_nodes * n_nodes * n_nodes * capacity)
  u_t = unsafe_wrap(Array, pointer(_u_t),
                    (NVARS, n_nodes, n_nodes, n_nodes, capacity))

  # u_rungakutta is initialized to non-NaN since it is used directly
  _u_tmp2 = fill(zero(RealT), NVARS * n_nodes * n_nodes * n_nodes * capacity)
  u_tmp2 = unsafe_wrap(Array, pointer(_u_tmp2),
                       (NVARS, n_nodes, n_nodes, n_nodes, capacity))

  _u_tmp3 = fill(zero(RealT), NVARS * n_nodes * n_nodes * n_nodes * capacity)
  u_tmp3 = unsafe_wrap(Array, pointer(_u_tmp3),
                       (NVARS, n_nodes, n_nodes, n_nodes, capacity))

  inverse_jacobian = fill(nan, capacity)

  _node_coordinates = fill(nan, 3 * n_nodes * n_nodes * n_nodes * capacity)
  node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                 (3, n_nodes, n_nodes, n_nodes, capacity))

  _surface_flux_values = fill(nan, NVARS * n_nodes * n_nodes * 2 * 3 * capacity)
  surface_flux_values = unsafe_wrap(Array, pointer(_surface_flux_values),
                                    (NVARS, n_nodes, n_nodes, 2 * 3, capacity))

  cell_ids = fill(typemin(Int), capacity)


  return ElementContainer3D{RealT, NVARS, POLYDEG}(
    u, u_t, u_tmp2, u_tmp3,
    inverse_jacobian, node_coordinates, surface_flux_values, cell_ids,
    _u, _u_t, _u_tmp2, _u_tmp3, _node_coordinates, _surface_flux_values)
end


# Return number of elements
nelements(elements::ElementContainer3D) = length(elements.cell_ids)


# Container data structure (structure-of-arrays style) for DG interfaces
# TODO: Taal refactor, remove NVARS, POLYDEG?
mutable struct InterfaceContainer3D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  u::Array{Real, 5}         # [leftright, variables, i, j, interfaces]
  neighbor_ids::Matrix{Int} # [leftright, interfaces]
  orientations::Vector{Int} # [interfaces]
  # internal `resize!`able storage
  _u::Vector{RealT}
  _neighbor_ids::Vector{Int}
end

# See explanation of Base.resize! for the element container
function Base.resize!(interfaces::InterfaceContainer3D{RealT, NVARS, POLYDEG},
                      capacity) where {RealT, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  @unpack _u, _neighbor_ids, orientations = interfaces

  resize!(_u, 2 * NVARS * n_nodes * n_nodes * capacity)
  interfaces.u = unsafe_wrap(Array, pointer(_u),
                             (2, NVARS, n_nodes, n_nodes, capacity))

  resize!(_neighbor_ids, 2 * capacity)
  interfaces.neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids ),
                                        (2, capacity))

  resize!(orientations, capacity)

  return nothing
end


function InterfaceContainer3D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  _u = fill(nan, 2 * NVARS * n_nodes * n_nodes * capacity)
  u = unsafe_wrap(Array, pointer(_u),
                  (2, NVARS, n_nodes, n_nodes, capacity))

  _neighbor_ids = fill(typemin(Int), 2 * capacity)
  neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids ),
                             (2, capacity))

  orientations = fill(typemin(Int), capacity)


  return InterfaceContainer3D{RealT, NVARS, POLYDEG}(
    u, neighbor_ids, orientations,
    _u, _neighbor_ids)
end


# Return number of interfaces
ninterfaces(interfaces::InterfaceContainer3D) = length(interfaces.orientations)


# Container data structure (structure-of-arrays style) for DG boundaries
# TODO: Taal refactor, remove NVARS, POLYDEG?
mutable struct BoundaryContainer3D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  u::Array{RealT, 5}                # [leftright, variables, i, j, boundaries]
  neighbor_ids::Vector{Int}         # [boundaries]
  orientations::Vector{Int}         # [boundaries]
  neighbor_sides::Vector{Int}       # [boundaries]
  node_coordinates::Array{RealT, 4} # [orientation, i, j, elements]
  n_boundaries_per_direction::SVector{6, Int} # [direction]
  # internal `resize!`able storage
  _u::Vector{RealT}
  _node_coordinates::Vector{RealT}
end

# See explanation of Base.resize! for the element container
function Base.resize!(boundaries::BoundaryContainer3D{RealT, NVARS, POLYDEG},
                      capacity) where {RealT, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  @unpack _u, _node_coordinates,
          neighbor_ids, orientations, neighbor_sides = boundaries

  resize!(_u, 2 * NVARS * n_nodes * n_nodes * capacity)
  boundaries.u = unsafe_wrap(Array, pointer(_u),
                             (2, NVARS, n_nodes, n_nodes, capacity))

  resize!(_node_coordinates, 3 * n_nodes * n_nodes * capacity)
  boundaries.node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates ),
                                            (3, n_nodes, n_nodes, capacity))

  resize!(neighbor_ids, capacity)

  resize!(orientations, capacity)

  resize!(neighbor_sides, capacity)

  return nothing
end


function BoundaryContainer3D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  _u = fill(nan, 2 * NVARS * n_nodes * n_nodes * capacity)
  u = unsafe_wrap(Array, pointer(_u),
                  (2, NVARS, n_nodes, n_nodes, capacity))

  neighbor_ids = fill(typemin(Int), capacity)

  orientations = fill(typemin(Int), capacity)

  neighbor_sides = fill(typemin(Int), capacity)

  _node_coordinates = fill(nan, 3 * n_nodes * n_nodes * capacity)
  node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                 (3, n_nodes, n_nodes, capacity))

  n_boundaries_per_direction = SVector(0, 0, 0, 0, 0, 0)

  return BoundaryContainer3D{RealT, NVARS, POLYDEG}(
    u, neighbor_ids, orientations, neighbor_sides,
    node_coordinates, n_boundaries_per_direction,
    _u, _node_coordinates)
end


# Return number of boundaries
nboundaries(boundaries::BoundaryContainer3D) = length(boundaries.orientations)


# Container data structure (structure-of-arrays style) for DG L2 mortars
# Positions/directions for orientations = 1, large_sides = 2:
# mortar is orthogonal to x-axis, large side is in positive coordinate direction wrt mortar
#   /----------------------------\  /----------------------------\
#   |             |              |  |                            |
#   | upper, left | upper, right |  |                            |
#   |      3      |      4       |  |                            |
#   |             |              |  |           large            |
#   |-------------|--------------|  |             5              |
# z |             |              |  |                            |
#   | lower, left | lower, right |  |                            |
# ^ |      1      |      2       |  |                            |
# | |             |              |  |                            |
# | \----------------------------/  \----------------------------/
# |
# ⋅----> y
# Left and right are always wrt to a coordinate direction:
# * left is always the negative direction
# * right is always the positive direction
#
# Left and right are used *both* for the numbering of the mortar faces *and* for the position of the
# elements with respect to the axis orthogonal to the mortar.
mutable struct L2MortarContainer3D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  u_upper_left ::Array{RealT, 5} # [leftright, variables, i, j, mortars]
  u_upper_right::Array{RealT, 5} # [leftright, variables, i, j, mortars]
  u_lower_left ::Array{RealT, 5} # [leftright, variables, i, j, mortars]
  u_lower_right::Array{RealT, 5} # [leftright, variables, i, j, mortars]
  neighbor_ids::Matrix{Int}      # [position, mortars]
  # Large sides: left -> 1, right -> 2
  large_sides::Vector{Int}   # [mortars]
  orientations::Vector{Int}  # [mortars]
  # internal `resize!`able storage
  _u_upper_left::Vector{RealT}
  _u_upper_right::Vector{RealT}
  _u_lower_left::Vector{RealT}
  _u_lower_right::Vector{RealT}
  _neighbor_ids::Vector{Int}
end

# See explanation of Base.resize! for the element container
function Base.resize!(mortars::L2MortarContainer3D{RealT, NVARS, POLYDEG},
                      capacity) where {RealT, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  @unpack _u_upper_left, _u_upper_right, _u_lower_left, _u_lower_right,
          _neighbor_ids, large_sides, orientations = mortars

  resize!(_u_upper_left, 2 * NVARS * n_nodes * n_nodes * capacity)
  mortars.u_upper_left = unsafe_wrap(Array, pointer(_u_upper_left),
                                     (2, NVARS, n_nodes, n_nodes, capacity))

  resize!(_u_upper_right, 2 * NVARS * n_nodes * n_nodes * capacity)
  mortars.u_upper_right = unsafe_wrap(Array, pointer(_u_upper_right),
                                      (2, NVARS, n_nodes, n_nodes, capacity))

  resize!(_u_lower_left, 2 * NVARS * n_nodes * n_nodes * capacity)
  mortars.u_lower_left = unsafe_wrap(Array, pointer(_u_lower_left),
                                     (2, NVARS, n_nodes, n_nodes, capacity))

  resize!(_u_lower_right, 2 * NVARS * n_nodes * n_nodes * capacity)
  mortars.u_lower_right = unsafe_wrap(Array, pointer(_u_lower_right),
                                      (2, NVARS, n_nodes, n_nodes, capacity))

  resize!(_neighbor_ids, 5 * capacity)
  mortars.neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids ),
                                        (5, capacity))

  resize!(large_sides, capacity)

  resize!(orientations, capacity)

  return nothing
end


function L2MortarContainer3D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  _u_upper_left = fill(nan, 2 * NVARS * n_nodes * n_nodes * capacity)
  u_upper_left = unsafe_wrap(Array, pointer(_u_upper_left),
                             (2, NVARS, n_nodes, n_nodes, capacity))

  _u_upper_right = fill(nan, 2 * NVARS * n_nodes * n_nodes * capacity)
  u_upper_right = unsafe_wrap(Array, pointer(_u_upper_right),
                              (2, NVARS, n_nodes, n_nodes, capacity))

  _u_lower_left = fill(nan, 2 * NVARS * n_nodes * n_nodes * capacity)
  u_lower_left = unsafe_wrap(Array, pointer(_u_lower_left),
                             (2, NVARS, n_nodes, n_nodes, capacity))

  _u_lower_right = fill(nan, 2 * NVARS * n_nodes * n_nodes * capacity)
  u_lower_right = unsafe_wrap(Array, pointer(_u_lower_right),
                              (2, NVARS, n_nodes, n_nodes, capacity))

  _neighbor_ids = fill(typemin(Int), 5 * capacity)
  neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids),
                             (5, capacity))

  large_sides = fill(typemin(Int), capacity)

  orientations = fill(typemin(Int), capacity)

  return L2MortarContainer3D{RealT, NVARS, POLYDEG}(
    u_upper_left, u_upper_right,
    u_lower_left, u_lower_right,
    neighbor_ids, large_sides, orientations,
    _u_upper_left, _u_upper_right,
    _u_lower_left, _u_lower_right,
    _neighbor_ids)
end


# Return number of L2 mortars
nmortars(l2mortars::L2MortarContainer3D) = length(l2mortars.orientations)


# Allow printing container contents
function Base.show(io::IO, ::MIME"text/plain", c::L2MortarContainer3D)
  println(io, '*'^20)
  for idx in CartesianIndices(c.u_upper_left)
    println(io, "c.u_upper_left[$idx] = $(c.u_upper_left[idx])")
  end
  for idx in CartesianIndices(c.u_upper_right)
    println(io, "c.u_upper_right[$idx] = $(c.u_upper_right[idx])")
  end
  for idx in CartesianIndices(c.u_lower_left)
    println(io, "c.u_lower_left[$idx] = $(c.u_lower_left[idx])")
  end
  for idx in CartesianIndices(c.u_lower_right)
    println(io, "c.u_lower_right[$idx] = $(c.u_lower_right[idx])")
  end
  println(io, "transpose(c.neighbor_ids) = $(transpose(c.neighbor_ids))")
  println(io, "c.large_sides = $(c.large_sides)")
  println(io, "c.orientations = $(c.orientations)")
  print(io,   '*'^20)
end
