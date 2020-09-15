
# Everything related to a DG semidiscretization on Lobatto-Legendre nodes in 2D

function create_cache(mesh::TreeMesh{2}, equations::AbstractEquations{2},
                      boundary_conditions, dg::DG, RealT)
  # element_variables::Dict{Symbol, Union{Vector{Float64}, Vector{Int}}}
  # cache::Dict{Symbol, Any}
  # thread_cache::Any # to make fully-typed output more readable

  # Create the basic cache
  # Get cells for which an element needs to be created (i.e. all leaf cells)
  leaf_cell_ids = leaf_cells(mesh.tree)

  # TODO: Taal refactor, we should pass the basis as argument,
  # not polydeg, to all of the following initialization methods
  elements = init_elements(leaf_cell_ids, mesh,
                           RealT, nvariables(equations), polydeg(dg))

  interfaces = init_interfaces(leaf_cell_ids, mesh, elements,
                               RealT, nvariables(equations), polydeg(dg))

  boundaries = init_boundaries(leaf_cell_ids, mesh, elements,
                               RealT, nvariables(equations), polydeg(dg))

  mortars = init_mortars(leaf_cell_ids, mesh, elements,
                         RealT, nvariables(equations), polydeg(dg), dg.mortar)

  cache = (; elements, interfaces, boundaries, mortars)


  # Add specialized parts of the cache required to compute the volume integral etc.
  cache = (cache..., create_cache(mesh, equations, dg.volume_integral)...)
  cache = (cache..., create_cache(mesh, equations, dg.mortar)...)

  return cache
end


# function create_cache(mesh::TreeMesh{2}, equations, ::VolumeIntegralFluxDifferencing)
#   # TODO: Taal implement
# end

# function create_cache(mesh::TreeMesh{2}, equations, ::VolumeIntegralShockCapturingHG)
#   # TODO: Taal implement
# end

# function create_cache(mesh::TreeMesh{2}, equations, ::LobattoLegendreMortarL2)
#   # TODO: Taal implement
# end


# TODO: Taal implement
# function integrate(func, u, mesh::TreeMesh{2}, equations, dg::DG, cache; normalize=true)
# end

# TODO: Taal implement
# function calc_error_norms(func, u, t, mesh::TreeMesh{2}, equations, initial_conditions, dg::DG, cache)
# end


# TODO: Taal implement
# function allocate_coefficients(mesh::TreeMesh{2}, equations, dg::DG, cache)
# end

# TODO: Taal implement
# function compute_coefficients!(u, func, mesh::TreeMesh{2}, equations, dg::DG, cache)
# end

# TODO: Taal implement timer
function rhs!(du, u, t,
              mesh::TreeMesh{2}, equations,
              initial_conditions, boundary_conditions, source_terms,
              dg::DG, cache)
  # Reset du
  du .= zero(eltype(du))

  # Calculate volume integral
  calc_volume_integral!(du, u, equations, solver, cache)

  # Prolong solution to interfaces
  # TODO: Taal decide order of arguments, consistent vs. modified cache first?
  prolong2interfaces!(cache, u, equations, solver)

  # Calculate interface fluxes
  calc_interface_flux!(cache, equations, solver)

  # Prolong solution to boundaries
  prolong2boundaries!(cache, u, equations, solver)

  # Calculate boundary fluxes
  calc_boundary_flux!(cache, t, boundary_conditions, equations, solver)

  # Prolong solution to mortars
  prolong2mortars!(cache, u, equations, solver)

  # Calculate mortar fluxes
  calc_mortar_flux!(cache, equations, solver)

  # Calculate surface integrals
  calc_surface_integral!(du, equations, solver, cache)

  # Apply Jacobian from mapping to reference element
  apply_jacobian!(du, equations, solver, cache)

  # Calculate source terms
  calc_sources!(du, u, t, source_terms, equations, solver, cache)

  return nothing
end

# TODO: Taal implement
# function calc_volume_integral!(du, u, equations, solver, cache)
# end

# TODO: Taal implement
# function prolong2interfaces!(cache, u, equations, solver)
# end

# TODO: Taal implement
# function calc_interface_flux!(cache, equations, solver)
# end

# TODO: Taal implement
# function prolong2boundaries!(cache, u, equations, solver)
# end

# TODO: Taal implement
# function calc_boundary_flux!(cache, t, boundary_conditions, equations, solver)
# end

# TODO: Taal implement
# function prolong2mortars!(cache, u, equations, solver)
# end

# TODO: Taal implement
# function calc_mortar_flux!(cache, equations, solver)
# end

# TODO: Taal implement
# function calc_surface_integral!(du, equations, solver, cache)
# end

# TODO: Taal implement
# function apply_jacobian!(du, equations, solver, cache)
# end

# TODO: Taal implement
# function calc_sources!(du, u, t, source_terms, equations, solver, cache)
# end
