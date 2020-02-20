import ...Auxiliary.Containers: invalidate!, raw_copy!, move_connectivity!, delete_connectivity!
using ... Auxiliary.Containers: AbstractContainer


# Container data structure (structure-of-arrays style) for DG elements
struct ElementContainer{V, N} <: AbstractContainer
  u::Array{Float64, 3}
  u_t::Array{Float64, 3}
  u_rungekutta::Array{Float64, 3}
  flux::Array{Float64, 3}
  inverse_jacobian::Array{Float64, 1}
  node_coordinates::Array{Float64, 2}
  surface_ids::Array{Int, 2}
end


function ElementContainer{V, N}(capacity::Integer) where {V, N}
  # Initialize fields with defaults
  u = fill(NaN, V, N + 1, capacity)
  u_t = fill(NaN, V, N + 1, capacity)
  u_rungekutta = fill(0.0, V, N + 1, capacity) # Initialized to non-NaN since it is used directly
  flux = fill(NaN, V, N + 1, capacity)
  inverse_jacobian = fill(NaN, capacity)
  node_coordinates = fill(NaN, N + 1, capacity)
  surface_ids = fill(typemin(Int), 2, capacity)

  elements = ElementContainer{V, N}(u, u_t, u_rungekutta, flux,
                                    inverse_jacobian, node_coordinates, surface_ids)

  return elements
end


# Return number of elements
nelements(elements::ElementContainer) = length(elements.inverse_jacobian)
