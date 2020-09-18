
# Container data structure (structure-of-arrays style) for DG elements
struct ElementContainer1D{NVARS, POLYDEG} <: AbstractContainer
  u::Array{Float64, 3}                   # [variables, i, elements]
  u_t::Array{Float64, 3}                 # [variables, i, elements]
  u_tmp2::Array{Float64, 3}              # [variables, i, elements]
  u_tmp3::Array{Float64, 3}              # [variables, i, elements]
  inverse_jacobian::Vector{Float64}      # [elements]
  node_coordinates::Array{Float64, 3}    # [orientation, i, elements]
  surface_ids::Matrix{Int}               # [direction, elements]
  surface_flux_values::Array{Float64, 3} # [variables, direction, elements] 
  cell_ids::Vector{Int}                  # [elements]
end


function ElementContainer1D{NVARS, POLYDEG}(capacity::Integer) where {NVARS, POLYDEG}
  # Initialize fields with defaults
  n_nodes = POLYDEG + 1
  u = fill(NaN, NVARS, n_nodes, capacity)
  u_t = fill(NaN, NVARS, n_nodes, capacity)
  # u_rungakutta is initialized to non-NaN since it is used directly
  u_tmp2 = fill(0.0, NVARS, n_nodes, capacity)
  u_tmp3 = fill(0.0, NVARS, n_nodes, capacity)
  inverse_jacobian = fill(NaN, capacity)
  node_coordinates = fill(NaN, 1, n_nodes, capacity)
  surface_ids = fill(typemin(Int), 2 * 1, capacity)
  surface_flux_values = fill(NaN, NVARS, 2 * 1, capacity)
  cell_ids = fill(typemin(Int), capacity)

  elements = ElementContainer1D{NVARS, POLYDEG}(u, u_t, u_tmp2, u_tmp3, inverse_jacobian, node_coordinates,
                                                surface_ids, surface_flux_values, cell_ids)

  return elements
end


# Return number of elements
nelements(elements::ElementContainer1D) = length(elements.cell_ids)


# Container data structure (structure-of-arrays style) for DG interfaces
struct InterfaceContainer1D{NVARS, POLYDEG} <: AbstractContainer
  u::Array{Float64, 3}      # [leftright, variables, i, interfaces]
  neighbor_ids::Matrix{Int} # [leftright, interfaces]
  orientations::Vector{Int} # [interfaces]
end


function InterfaceContainer1D{NVARS, POLYDEG}(capacity::Integer) where {NVARS, POLYDEG}
  # Initialize fields with defaults
  n_nodes = POLYDEG + 1
  u = fill(NaN, 2, NVARS, capacity)
  neighbor_ids = fill(typemin(Int), 2, capacity)
  orientations = fill(typemin(Int), capacity)

  interfaces = InterfaceContainer1D{NVARS, POLYDEG}(u, neighbor_ids, orientations)

  return interfaces
end


# Return number of interfaces
ninterfaces(interfaces::InterfaceContainer1D) = length(interfaces.orientations)


# Container data structure (structure-of-arrays style) for DG boundaries
struct BoundaryContainer1D{NVARS, POLYDEG} <: AbstractContainer
  u::Array{Float64, 3}                # [leftright, variables,boundaries]
  neighbor_ids::Vector{Int}           # [boundaries]
  orientations::Vector{Int}           # [boundaries]
  neighbor_sides::Vector{Int}         # [boundaries]
  node_coordinates::Array{Float64, 2} # [orientation, elements]
end


function BoundaryContainer1D{NVARS, POLYDEG}(capacity::Integer) where {NVARS, POLYDEG}
  # Initialize fields with defaults
  n_nodes = POLYDEG + 1
  u = fill(NaN, 2, NVARS, capacity)
  neighbor_ids = fill(typemin(Int), capacity)
  orientations = fill(typemin(Int), capacity)
  neighbor_sides = fill(typemin(Int), capacity)
  node_coordinates = fill(NaN, 1, capacity)

  boundaries = BoundaryContainer1D{NVARS, POLYDEG}(u, neighbor_ids, orientations, neighbor_sides,
                                                   node_coordinates)

  return boundaries
end


# Return number of boundaries
nboundaries(boundaries::BoundaryContainer1D) = length(boundaries.orientations)
