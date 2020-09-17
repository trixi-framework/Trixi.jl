
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

  # TODO: Taal refactor
  # For me,
  # - surface_ids, cell_ids in elements
  # - neighbor_ids, orientations in interfaces
  # - neighbor_ids, orientations, neighbor_sides in boundaries
  # - neighbor_ids, large_sides, orientations in mortars
  # seem to be important information about the mesh.
  # Shall we store them there?

  # Add specialized parts of the cache required to compute the volume integral etc.
  cache = (;cache..., create_cache(mesh, equations, dg.volume_integral)...)
  cache = (;cache..., create_cache(mesh, equations, dg.mortar)...)

  return cache
end


# function create_cache(mesh::TreeMesh{2}, equations, ::VolumeIntegralFluxDifferencing)
#   # TODO: Taal implement
# end

# function create_cache(mesh::TreeMesh{2}, equations, ::VolumeIntegralShockCapturingHG)
#   # TODO: Taal implement
# end

function create_cache(mesh::TreeMesh{2}, equations, ::LobattoLegendreMortarL2)
  # TODO: Taal implement if necessary
  NamedTuple()
end


# TODO: Taal implement
# function integrate(func, u, mesh::TreeMesh{2}, equations, dg::DG, cache; normalize=true)
# end

# TODO: Taal implement
# function calc_error_norms(func, u, t, mesh::TreeMesh{2}, equations, initial_conditions, dg::DG, cache)
# end


function allocate_coefficients(mesh::TreeMesh{2}, equations, dg::DG, cache)
  zeros(real(dg), nvariables(equations), nnodes(dg), nnodes(dg), nelements(dg, cache))
end

function compute_coefficients!(u, func, t, mesh::TreeMesh{2}, equations, dg::DG, cache)

  Threads.@threads for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      x_node = get_node_coords(cache.elements.node_coordinates, mesh, equations, dg, i, j, element)
      u_node = func(x_node, t, equations)
      set_node_vars!(u, u_node, mesh, equations, dg, i, j, element)
    end
  end
end

# TODO: Taal refactor timer, allowing users to pass a custom timer?

function rhs!(du::AbstractArray{<:Any,4}, u, t,
              mesh::TreeMesh{2}, equations,
              initial_conditions, boundary_conditions, source_terms,
              dg::DG, cache)
  # Reset du
  @timeit_debug timer() "reset ∂u/∂t" du .= zero(eltype(du))

  # Calculate volume integral
  @timeit_debug timer() "volume integral" calc_volume_integral!(du, u, equations, dg.volume_integral, dg, cache)

  # Prolong solution to interfaces
  # TODO: Taal decide order of arguments, consistent vs. modified cache first?
  @timeit_debug timer() "prolong2interfaces" prolong2interfaces!(cache, u, equations, dg)

  # Calculate interface fluxes
  @timeit_debug timer() "interface flux" calc_interface_flux!(cache, equations, dg)

  # Prolong solution to boundaries
  @timeit_debug timer() "prolong2boundaries" prolong2boundaries!(cache, u, equations, dg)

  # Calculate boundary fluxes
  @timeit_debug timer() "boundary flux" calc_boundary_flux!(cache, t, boundary_conditions, equations, dg)

  # Prolong solution to mortars
  @timeit_debug timer() "prolong2mortars" prolong2mortars!(cache, u, equations, dg.mortar, dg)

  # Calculate mortar fluxes
  @timeit_debug timer() "mortar flux" calc_mortar_flux!(cache, equations, dg)

  # Calculate surface integrals
  @timeit_debug timer() "surface integral" calc_surface_integral!(du, equations, dg, cache)

  # Apply Jacobian from mapping to reference element
  @timeit_debug timer() "Jacobian" apply_jacobian!(du, equations, dg, cache)

  # Calculate source terms
  @timeit_debug timer() "source terms" calc_sources!(du, u, t, source_terms, equations, dg, cache)

  return nothing
end


# TODO: Taal implement
function calc_volume_integral!(du::AbstractArray{<:Any,4}, u, equations,
                               volume_integral::VolumeIntegralWeakForm,
                               dg::DGSEM, cache)
  @unpack derivative_neg_adjoint = dg.basis

  Threads.@threads for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element)

      flux1 = calcflux(u_node, 1, equations)
      for ii in eachnode(dg)
        integral_contribution = derivative_neg_adjoint[ii, i] * flux1
        add_to_node_vars!(du, integral_contribution, equations, dg, ii, j, element)
      end

      flux2 = calcflux(u_node, 2, equations)
      for jj in eachnode(dg)
        integral_contribution = derivative_neg_adjoint[jj, j] * flux2
        add_to_node_vars!(du, integral_contribution, equations, dg, i, jj, element)
      end
    end
  end

  return nothing
end

# TODO: Taal implement
# function calc_volume_integral!(du::AbstractArray{<:Any,4}, u, equations,
#                                volume_integral::VolumeIntegralFluxDifferencing,
#                                dg::DGSEM, cache)
# end

# TODO: Taal implement
# function calc_volume_integral!(du::AbstractArray{<:Any,4}, u, equations,
#                                volume_integral::VolumeIntegralShockCapturingHG,
#                                dg::DGSEM, cache)
# end


function prolong2interfaces!(cache, u::AbstractArray{<:Any,4}, equations, dg::DG)
  @unpack interfaces = cache
  @unpack orientations = interfaces

  Threads.@threads for interface in eachinterface(dg, cache)
    left_element  = interfaces.neighbor_ids[1, interface]
    right_element = interfaces.neighbor_ids[2, interface]

    if orientations[interface] == 1
      # interface in x-direction
      for j in eachnode(dg), v in eachvariable(equations)
        interfaces.u[1, v, j, interface] = u[v, nnodes(dg), j, left_element]
        interfaces.u[2, v, j, interface] = u[v,          1, j, right_element]
      end
    else # if orientations[interface] == 2
      # interface in y-direction
      for i in eachnode(dg), v in eachvariable(equations)
        interfaces.u[1, v, i, interface] = u[v, i, nnodes(dg), left_element]
        interfaces.u[2, v, i, interface] = u[v, i,          1, right_element]
      end
    end
  end

  return nothing
end

# TODO: Taal dimension agnostic
@inline function calc_interface_flux!(cache, equations, dg::DG)
  calc_interface_flux!(cache.elements.surface_flux_values,
                       have_nonconservative_terms(equations), equations,
                       dg, cache)
end

function calc_interface_flux!(surface_flux_values::AbstractArray{<:Any,4},
                              nonconservative_terms::Val{false}, equations,
                              dg::DG, cache)
  @unpack surface_flux = dg
  @unpack u, neighbor_ids, orientations = cache.interfaces

  Threads.@threads for interface in eachinterface(dg, cache)
    # Get neighboring elements
    left_id  = neighbor_ids[1, interface]
    right_id = neighbor_ids[2, interface]

    # Determine interface direction with respect to elements:
    # orientation = 1: left -> 2, right -> 1
    # orientation = 2: left -> 4, right -> 3
    left_direction  = 2 * orientations[interface]
    right_direction = 2 * orientations[interface] - 1

    for i in eachnode(dg)
      # Call pointwise Riemann solver
      u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, interface)
      flux = surface_flux(u_ll, u_rr, orientations[interface], equations)

      # Copy flux to left and right element storage
      for v in 1:nvariables(equations)
        surface_flux_values[v, i, left_direction,  left_id]  = flux[v]
        surface_flux_values[v, i, right_direction, right_id] = flux[v]
      end
    end
  end
end

# TODO: Taal implement
# function calc_interface_flux!(surface_flux_values::AbstractArray{<:Any,4},
#   nonconservative_terms::Val{true}, equations,
#   dg::DG, cache)
# end


function prolong2boundaries!(cache, u::AbstractArray{<:Any,4}, equations, dg::DG)
  @unpack boundaries = cache
  @unpack orientations, neighbor_sides = boundaries

  Threads.@threads for boundary in eachboundary(dg, cache)
    element = boundaries.neighbor_ids[boundary]

    if orientations[b] == 1
      # boundary in x-direction
      if neighbor_sides[b] == 1
        # element in -x direction of boundary
        for l in eachnode(dg), v in eachvariable(equations)
          boundaries.u[1, v, l, b] = u[v, nnodes(dg), l, element]
        end
      else # Element in +x direction of boundary
        for l in eachnode(dg), v in eachvariable(equations)
          boundaries.u[2, v, l, b] = u[v, 1,          l, element]
        end
      end
    else # if orientations[b] == 2
      # boundary in y-direction
      if neighbor_sides[b] == 1
        # element in -y direction of boundary
        for l in eachnode(dg), v in eachvariable(equations)
          boundaries.u[1, v, l, b] = u[v, l, nnodes(dg), element]
        end
      else
        # element in +y direction of boundary
        for l in eachnode(dg), v in eachvariable(equations)
          boundaries.u[2, v, l, b] = u[v, l, 1,          element]
        end
      end
    end
  end

  return nothing
end

# TODO: Taal implement
function calc_boundary_flux!(cache, t, boundary_conditions, equations, dg::DGSEM)
  @assert isempty(eachboundary(dg, cache))
end


# TODO: Taal implement
function prolong2mortars!(cache, u::AbstractArray{<:Any,4}, equations, mortar::LobattoLegendreMortarL2, dg::DGSEM)
  @assert isempty(eachmortar(dg, cache))
end

# TODO: Taal implement
function calc_mortar_flux!(cache, equations, dg::DGSEM)
  @assert isempty(eachmortar(dg, cache))
end


function calc_surface_integral!(du::AbstractArray{<:Any,4}, equations, dg::DGSEM, cache)
  @unpack boundary_interpolation = dg.basis
  @unpack surface_flux_values = cache.elements

  Threads.@threads for element in eachelement(dg, cache)
    for l in eachnode(dg)
      for v in eachvariable(equations)
        # surface at -x
        du[v, 1,          l, element] -= surface_flux_values[v, l, 1, element] * boundary_interpolation[1,          1]
        # surface at +x
        du[v, nnodes(dg), l, element] += surface_flux_values[v, l, 2, element] * boundary_interpolation[nnodes(dg), 2]
        # surface at -y
        du[v, l, 1,          element] -= surface_flux_values[v, l, 3, element] * boundary_interpolation[1,          1]
        # surface at +y
        du[v, l, nnodes(dg), element] += surface_flux_values[v, l, 4, element] * boundary_interpolation[nnodes(dg), 2]
      end
    end
  end

  return nothing
end


function apply_jacobian!(du::AbstractArray{<:Any,4}, equations, dg::DG, cache)

  Threads.@threads for element in eachelement(dg, cache)
    factor = -cache.elements.inverse_jacobian[element]

    for j in eachnode(dg), i in eachnode(dg)
      for v in eachvariable(equations)
        du[v, i, j, element] *= factor
      end
    end
  end

  return nothing
end


# TODO: Taal implement
function calc_sources!(du::AbstractArray{<:Any,4}, u, t, source_terms, equations, dg::DG, cache)
  @assert source_terms === source_terms_nothing
end
