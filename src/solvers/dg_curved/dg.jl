# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::CurvedMesh, equations::AbstractEquations, dg::DG, ::Any, ::Type{uEltype}) where {uEltype<:Real}
  elements = init_elements(mesh, equations, dg.basis, uEltype)

  cache = (; elements)

  return cache
end


function calc_boundary_flux!(cache, u, t, boundary_condition,
                             equations::AbstractEquations, mesh::CurvedMesh, dg::DG)
  boundary_conditions = ntuple(_ -> boundary_condition, 2*ndims(mesh))
  calc_boundary_flux!(cache, u, t, boundary_conditions,
                      equations::AbstractEquations, mesh::CurvedMesh, dg::DG)
end


# Calc boundary flux at the specified element and nodes in the specified direction and save it in surface_flux_values
@inline function calc_boundary_flux_at_node!(surface_flux_values, node_coordinates, u, t, boundary_conditions, 
                                             equations, dg, orientation, direction, element, node_indices...)
  @unpack surface_flux = dg
  boundary_condition = boundary_conditions[direction]

  # Get boundary flux
  u_rr = get_node_vars(u, equations, dg, node_indices..., element)
  x = get_node_coords(node_coordinates, equations, dg, node_indices..., element)
  flux = boundary_condition(u_rr, orientation, direction, x, t, surface_flux, equations)

  # Copy flux to left and right element storage
  for v in eachvariable(equations)
    surface_flux_values[v, direction, element] = flux[v]
  end
end


@inline ndofs(mesh::CurvedMesh, dg::DG, cache) = nelements(cache.elements) * nnodes(dg)^ndims(mesh)


include("containers.jl")
include("dg_1d.jl")
include("dg_2d.jl")
include("dg_3d.jl")
