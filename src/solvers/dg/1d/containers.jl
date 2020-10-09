
# Container data structure (structure-of-arrays style) for DG elements
# TODO: Taal refactor, remove u, u_t, u_tmp2, u_tmp3
# TODO: Taal refactor, remove NVARS, POLYDEG?
# TODO: Taal refactor, mutable struct or resize! for AMR?
mutable struct ElementContainer1D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  u::Array{RealT, 3}                   # [variables, i, elements]
  u_t::Array{RealT, 3}                 # [variables, i, elements]
  u_tmp2::Array{RealT, 3}              # [variables, i, elements]
  u_tmp3::Array{RealT, 3}              # [variables, i, elements]
  inverse_jacobian::Vector{RealT}      # [elements]
  node_coordinates::Array{RealT, 3}    # [orientation, i, elements]
  surface_flux_values::Array{RealT, 3} # [variables, direction, elements]
  cell_ids::Vector{Int}                # [elements]
end

function Base.copy!(dst::ElementContainer1D, src::ElementContainer1D)
  dst.u                   = src.u
  dst.u_t                 = src.u_t
  dst.u_tmp2              = src.u_tmp2
  dst.u_tmp3              = src.u_tmp3
  dst.inverse_jacobian    = src.inverse_jacobian
  dst.node_coordinates    = src.node_coordinates
  dst.surface_flux_values = src.surface_flux_values
  dst.cell_ids            = src.cell_ids
  return nothing
end


function ElementContainer1D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  u = fill(nan, NVARS, n_nodes, capacity)
  u_t = fill(nan, NVARS, n_nodes, capacity)
  # u_rungakutta is initialized to non-NaN since it is used directly
  u_tmp2 = fill(zero(RealT), NVARS, n_nodes, capacity)
  u_tmp3 = fill(zero(RealT), NVARS, n_nodes, capacity)
  inverse_jacobian = fill(nan, capacity)
  node_coordinates = fill(nan, 1, n_nodes, capacity)
  surface_flux_values = fill(nan, NVARS, 2 * 1, capacity)
  cell_ids = fill(typemin(Int), capacity)

  elements = ElementContainer1D{RealT, NVARS, POLYDEG}(
    u, u_t, u_tmp2, u_tmp3,
    inverse_jacobian, node_coordinates, surface_flux_values, cell_ids)

  return elements
end


# Return number of elements
nelements(elements::ElementContainer1D) = length(elements.cell_ids)


# Container data structure (structure-of-arrays style) for DG interfaces
# TODO: Taal refactor, remove NVARS, POLYDEG?
# TODO: Taal refactor, mutable struct or resize! for AMR?
mutable struct InterfaceContainer1D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  u::Array{RealT, 3}        # [leftright, variables, interfaces]
  neighbor_ids::Matrix{Int} # [leftright, interfaces]
  orientations::Vector{Int} # [interfaces]
end

function Base.copy!(dst::InterfaceContainer1D, src::InterfaceContainer1D)
  dst.u            = src.u
  dst.neighbor_ids = src.neighbor_ids
  dst.orientations = src.orientations
  return nothing
end


function InterfaceContainer1D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  u = fill(nan, 2, NVARS, capacity)
  neighbor_ids = fill(typemin(Int), 2, capacity)
  orientations = fill(typemin(Int), capacity)

  interfaces = InterfaceContainer1D{RealT, NVARS, POLYDEG}(
    u, neighbor_ids, orientations)

  return interfaces
end


# Return number of interfaces
ninterfaces(interfaces::InterfaceContainer1D) = length(interfaces.orientations)


# Container data structure (structure-of-arrays style) for DG boundaries
# TODO: Taal refactor, remove NVARS, POLYDEG?
# TODO: Taal refactor, mutable struct or resize! for AMR?
mutable struct BoundaryContainer1D{RealT<:Real, NVARS, POLYDEG} <: AbstractContainer
  u::Array{RealT, 3}                # [leftright, variables, boundaries]
  neighbor_ids::Vector{Int}         # [boundaries]
  orientations::Vector{Int}         # [boundaries]
  neighbor_sides::Vector{Int}       # [boundaries]
  node_coordinates::Array{RealT, 2} # [orientation, elements]
  n_boundaries_per_direction::SVector{2, Int} # [direction]
end

function Base.copy!(dst::BoundaryContainer1D, src::BoundaryContainer1D)
  dst.u                          = src.u
  dst.neighbor_ids               = src.neighbor_ids
  dst.orientations               = src.orientations
  dst.neighbor_sides             = src.neighbor_sides
  dst.node_coordinates           = src.node_coordinates
  dst.n_boundaries_per_direction = src.n_boundaries_per_direction
  return nothing
end


function BoundaryContainer1D{RealT, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, NVARS, POLYDEG}
  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  # Initialize fields with defaults
  u = fill(nan, 2, NVARS, capacity)
  neighbor_ids = fill(typemin(Int), capacity)
  orientations = fill(typemin(Int), capacity)
  neighbor_sides = fill(typemin(Int), capacity)
  node_coordinates = fill(nan, 1, capacity)
  n_boundaries_per_direction = SVector(0, 0)

  boundaries = BoundaryContainer1D{RealT, NVARS, POLYDEG}(
    u, neighbor_ids, orientations, neighbor_sides,
    node_coordinates, n_boundaries_per_direction)

  return boundaries
end


# Return number of boundaries
nboundaries(boundaries::BoundaryContainer1D) = length(boundaries.orientations)
