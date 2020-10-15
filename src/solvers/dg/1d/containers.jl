
# Container data structure (structure-of-arrays style) for DG elements
# TODO: Taal refactor, remove u, u_t, u_tmp2, u_tmp3
# TODO: Taal refactor, remove NVARS, POLYDEG?
mutable struct ElementContainer1D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  u::Array{RealT, 3}                   # [variables, i, elements]
  u_t::Array{RealT, 3}                 # [variables, i, elements]
  u_tmp2::Array{RealT, 3}              # [variables, i, elements]
  u_tmp3::Array{RealT, 3}              # [variables, i, elements]
  inverse_jacobian::Vector{RealT}      # [elements]
  node_coordinates::Array{RealT, 3}    # [orientation, i, elements]
  surface_flux_values::Array{RealT, 3} # [variables, direction, elements]
  cell_ids::Vector{Int}                # [elements]
  # internal `resize!`able storage
  _u::Vector{RealT}
  _u_t::Vector{RealT}
  _u_tmp2::Vector{RealT}
  _u_tmp3::Vector{RealT}
  _node_coordinates::Vector{RealT}
  _surface_flux_values::Vector{RealT}
end

function Base.resize!(elements::ElementContainer1D{RealT, NVARS, POLYDEG},
                      capacity) where {RealT, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  @unpack _u, _u_t, _u_tmp2, _u_tmp3, _node_coordinates, _surface_flux_values,
          inverse_jacobian, cell_ids = elements

  resize!(_u, NVARS * n_nodes * capacity)
  elements.u = unsafe_wrap(Array, pointer(_u),
                           (NVARS, n_nodes, capacity))

  resize!(_u_t, NVARS * n_nodes * capacity)
  elements.u_t = unsafe_wrap(Array, pointer(_u_t),
                             (NVARS, n_nodes, capacity))

  resize!(_u_tmp2, NVARS * n_nodes * capacity)
  _u_tmp2 .= zero(eltype(_u_tmp2))
  elements.u_tmp2 = unsafe_wrap(Array, pointer(_u_tmp2),
                                (NVARS, n_nodes, capacity))

  resize!(_u_tmp3, NVARS * n_nodes * capacity)
  _u_tmp3 .= zero(eltype(_u_tmp3))
  elements.u_tmp3 = unsafe_wrap(Array, pointer(_u_tmp3),
                                (NVARS, n_nodes, capacity))

  resize!(inverse_jacobian, capacity)

  resize!(_node_coordinates, 1 * n_nodes * capacity)
  elements.node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                          (1, n_nodes, capacity))

  resize!(_surface_flux_values, NVARS * 2 * 1 * capacity)
  elements.surface_flux_values = unsafe_wrap(Array, pointer(_surface_flux_values),
                                             (NVARS, 2 * 1, capacity))

  resize!(cell_ids, capacity)

  return nothing
end


function ElementContainer1D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  _u = fill(nan, NVARS * n_nodes * capacity)
  u = unsafe_wrap(Array, pointer(_u),
                  (NVARS, n_nodes, capacity))

  _u_t = fill(nan, NVARS *  n_nodes * capacity)
  u_t = unsafe_wrap(Array, pointer(_u_t),
                    (NVARS, n_nodes, capacity))

  # u_rungakutta is initialized to non-NaN since it is used directly
  _u_tmp2 = fill(zero(RealT), NVARS * n_nodes * capacity)
  u_tmp2 = unsafe_wrap(Array, pointer(_u_tmp2),
                       (NVARS, n_nodes, capacity))

  _u_tmp3 = fill(zero(RealT), NVARS * n_nodes * capacity)
  u_tmp3 = unsafe_wrap(Array, pointer(_u_tmp3),
                       (NVARS, n_nodes, capacity))

  inverse_jacobian = fill(nan, capacity)

  _node_coordinates = fill(nan, 1 * n_nodes * capacity)
  node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                 (1, n_nodes, capacity))

  _surface_flux_values = fill(nan, NVARS * 2 * 1 * capacity)
  surface_flux_values = unsafe_wrap(Array, pointer(_surface_flux_values),
                                    (NVARS, 2 * 1, capacity))

  cell_ids = fill(typemin(Int), capacity)

  return ElementContainer1D{RealT, NVARS, POLYDEG}(
    u, u_t, u_tmp2, u_tmp3,
    inverse_jacobian, node_coordinates, surface_flux_values, cell_ids,
    _u, _u_t, _u_tmp2, _u_tmp3, _node_coordinates, _surface_flux_values)
end


# Return number of elements
@inline nelements(elements::ElementContainer1D) = length(elements.cell_ids)


# Container data structure (structure-of-arrays style) for DG interfaces
# TODO: Taal refactor, remove NVARS, POLYDEG?
mutable struct InterfaceContainer1D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  u::Array{RealT, 3}        # [leftright, variables, interfaces]
  neighbor_ids::Matrix{Int} # [leftright, interfaces]
  orientations::Vector{Int} # [interfaces]
  # internal `resize!`able storage
  _u::Vector{RealT}
  _neighbor_ids::Vector{Int}
end

function Base.resize!(interfaces::InterfaceContainer1D{RealT, NVARS, POLYDEG},
                      capacity) where {RealT, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  @unpack _u, _neighbor_ids, orientations = interfaces

  resize!(_u, 2 * NVARS * capacity)
  interfaces.u = unsafe_wrap(Array, pointer(_u),
                             (2, NVARS, capacity))

  resize!(_neighbor_ids, 2 * capacity)
  interfaces.neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids ),
                                        (2, capacity))

  resize!(orientations, capacity)

  return nothing
end


function InterfaceContainer1D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  _u = fill(nan, 2 * NVARS * capacity)
  u = unsafe_wrap(Array, pointer(_u),
                  (2, NVARS, capacity))

  _neighbor_ids = fill(typemin(Int), 2 * capacity)
  neighbor_ids = unsafe_wrap(Array, pointer(_neighbor_ids ),
                             (2, capacity))

  orientations = fill(typemin(Int), capacity)

  return InterfaceContainer1D{RealT, NVARS, POLYDEG}(
    u, neighbor_ids, orientations,
    _u, _neighbor_ids)
end


# Return number of interfaces
@inline ninterfaces(interfaces::InterfaceContainer1D) = length(interfaces.orientations)


# Container data structure (structure-of-arrays style) for DG boundaries
# TODO: Taal refactor, remove NVARS, POLYDEG?
mutable struct BoundaryContainer1D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  u::Array{RealT, 3}                # [leftright, variables, boundaries]
  neighbor_ids::Vector{Int}         # [boundaries]
  orientations::Vector{Int}         # [boundaries]
  neighbor_sides::Vector{Int}       # [boundaries]
  node_coordinates::Array{RealT, 2} # [orientation, elements]
  n_boundaries_per_direction::SVector{2, Int} # [direction]
  # internal `resize!`able storage
  _u::Vector{RealT}
  _node_coordinates::Vector{RealT}
end

function Base.resize!(boundaries::BoundaryContainer1D{RealT, NVARS, POLYDEG},
                      capacity) where {RealT, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  @unpack _u, _node_coordinates,
          neighbor_ids, orientations, neighbor_sides = boundaries

  resize!(_u, 2 * NVARS * capacity)
  boundaries.u = unsafe_wrap(Array, pointer(_u),
                             (2, NVARS, capacity))

  resize!(_node_coordinates, 1 * capacity)
  boundaries.node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates ),
                                            (1, capacity))

  resize!(neighbor_ids, capacity)

  resize!(orientations, capacity)

  resize!(neighbor_sides, capacity)

  return nothing
end


function BoundaryContainer1D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  _u = fill(nan, 2 * NVARS * capacity)
  u = unsafe_wrap(Array, pointer(_u),
                  (2, NVARS, capacity))

  neighbor_ids = fill(typemin(Int), capacity)

  orientations = fill(typemin(Int), capacity)

  neighbor_sides = fill(typemin(Int), capacity)

  _node_coordinates = fill(nan, 1 * capacity)
  node_coordinates = unsafe_wrap(Array, pointer(_node_coordinates),
                                 (1, capacity))

  n_boundaries_per_direction = SVector(0, 0)

  return BoundaryContainer1D{RealT, NVARS, POLYDEG}(
    u, neighbor_ids, orientations, neighbor_sides,
    node_coordinates, n_boundaries_per_direction,
    _u, _node_coordinates)
end


# Return number of boundaries
nboundaries(boundaries::BoundaryContainer1D) = length(boundaries.orientations)
