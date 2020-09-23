
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

  # TODO: Taal implement BCs
  boundaries, n_boundaries_per_direction = init_boundaries(leaf_cell_ids, mesh, elements,
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


function create_cache(mesh::TreeMesh{2}, equations, volume_integral::VolumeIntegralFluxDifferencing)
  create_cache(mesh, have_nonconservative_terms(equations), equations, volume_integral)
end

function create_cache(mesh::TreeMesh{2}, nonconservative_terms::Val{false}, equations, ::VolumeIntegralFluxDifferencing)
  NamedTuple()
end

# function create_cache(mesh::TreeMesh{2}, nonconservative_terms::Val{true}, equations, ::VolumeIntegralFluxDifferencing)
#   # TODO: Taal implement if necessary
#   NamedTuple()
# end

# function create_cache(mesh::TreeMesh{2}, equations, ::VolumeIntegralShockCapturingHG)
#   # TODO: Taal implement
# end

function create_cache(mesh::TreeMesh{2}, equations, mortar_l2::LobattoLegendreMortarL2)
  # TODO: Taal compare performance of different types
  MA2d = MArray{Tuple{nvariables(equations), nnodes(mortar_l2)}, real(mortar_l2)}
  # A2d  = Array{real(mortar_l2), 2}

  fstar_upper_threaded = [MA2d(undef) for _ in 1:Threads.nthreads()]
  fstar_lower_threaded = [MA2d(undef) for _ in 1:Threads.nthreads()]

  (; fstar_upper_threaded, fstar_lower_threaded)
end


function wrap_array(u::AbstractVector, mesh::TreeMesh{2}, equations, dg::DG, cache)
  unsafe_wrap(Array{eltype(u), ndims(mesh)+2}, pointer(u),
              (nvariables(equations), nnodes(dg), nnodes(dg), nelements(dg, cache)))
end


function compute_coefficients!(u, func, t, mesh::TreeMesh{2}, equations, dg::DG, cache)

  Threads.@threads for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      x_node = get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, element)
      u_node = func(x_node, t, equations)
      set_node_vars!(u, u_node, equations, dg, i, j, element)
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
  @timeit_debug timer() "volume integral" calc_volume_integral!(du, u, have_nonconservative_terms(equations), equations,
                                                                dg.volume_integral, dg, cache)

  # Prolong solution to interfaces
  # TODO: Taal decide order of arguments, consistent vs. modified cache first?
  @timeit_debug timer() "prolong2interfaces" prolong2interfaces!(cache, u, equations, dg)

  # Calculate interface fluxes
  @timeit_debug timer() "interface flux" calc_interface_flux!(cache.elements.surface_flux_values,
                                                              have_nonconservative_terms(equations), equations,
                                                              dg, cache)

  # Prolong solution to boundaries
  @timeit_debug timer() "prolong2boundaries" prolong2boundaries!(cache, u, equations, dg)

  # Calculate boundary fluxes
  @timeit_debug timer() "boundary flux" calc_boundary_flux!(cache, t, boundary_conditions, equations, dg)

  # Prolong solution to mortars
  @timeit_debug timer() "prolong2mortars" prolong2mortars!(cache, u, equations, dg.mortar, dg)

  # Calculate mortar fluxes
  @timeit_debug timer() "mortar flux" calc_mortar_flux!(cache.elements.surface_flux_values,
                                                        have_nonconservative_terms(equations), equations,
                                                        dg.mortar, dg, cache)

  # Calculate surface integrals
  @timeit_debug timer() "surface integral" calc_surface_integral!(du, equations, dg, cache)

  # Apply Jacobian from mapping to reference element
  @timeit_debug timer() "Jacobian" apply_jacobian!(du, equations, dg, cache)

  # Calculate source terms
  @timeit_debug timer() "source terms" calc_sources!(du, u, t, source_terms, equations, dg, cache)

  return nothing
end


function calc_volume_integral!(du::AbstractArray{<:Any,4}, u,
                               nonconservative_terms::Val{false}, equations,
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


function calc_volume_integral!(du::AbstractArray{<:Any,4}, u,
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralFluxDifferencing,
                               dg::DGSEM, cache)
  Threads.@threads for element in eachelement(dg, cache)
    split_form_kernel!(du, u, nonconservative_terms, equations, volume_integral.volume_flux, dg, cache, element)
  end
end

@inline function split_form_kernel!(du, u, nonconservative_terms::Val{false}, equations,
                                    volume_flux, dg::DGSEM, cache,
                                    element, alpha=true)
  # true * [some floating point value] == [exactly the same floating point value]
  # This can (hopefully) be optimized away due to constant propagation.
  @unpack derivative_split = dg.basis

  # Calculate volume integral in one element
  for j in eachnode(dg), i in eachnode(dg)
    u_node = get_node_vars(u, equations, dg, i, j, element)

    # x direction
    # use consistency of the volume flux to make this evaluation cheaper
    flux = calcflux(u_node, 1, equations)
    integral_contribution = alpha * derivative_split[i, i] * flux
    add_to_node_vars!(du, integral_contribution, equations, dg, i, j, element)
    # use symmetry of the volume flux for the remaining terms
    for ii in (i+1):nnodes(dg)
      u_node_ii = get_node_vars(u, equations, dg, ii, j, element)
      flux = volume_flux(u_node, u_node_ii, 1, equations)
      integral_contribution = alpha * derivative_split[i, ii] * flux
      add_to_node_vars!(du, integral_contribution, equations, dg, i,  j, element)
      integral_contribution = alpha * derivative_split[ii, i] * flux
      add_to_node_vars!(du, integral_contribution, equations, dg, ii, j, element)
    end

    # y direction
    # use consistency of the volume flux to make this evaluation cheaper
    flux = calcflux(u_node, 2, equations)
    integral_contribution = alpha * derivative_split[j, j] * flux
    add_to_node_vars!(du, integral_contribution, equations, dg, i, j, element)
    # use symmetry of the volume flux for the remaining terms
    for jj in (j+1):nnodes(dg)
      u_node_jj = get_node_vars(u, equations, dg, i, jj, element)
      flux = volume_flux(u_node, u_node_jj, 2, equations)
      integral_contribution = alpha * derivative_split[j, jj] * flux
      add_to_node_vars!(du, integral_contribution, equations, dg, i, j,  element)
      integral_contribution = alpha * derivative_split[jj, j] * flux
      add_to_node_vars!(du, integral_contribution, equations, dg, i, jj, element)
    end
  end
end

# TODO: Taal implement
# @inline function split_form_kernel!(du, u, nonconservative_terms::Val{true}, equations,
#                                     volume_flux, dg::DGSEM, cache,
#                                     element, alpha=true)
# end


# TODO: Taal implement
# function calc_volume_integral!(du::AbstractArray{<:Any,4}, u, nonconservative_terms::Val{false}, equations,
#                                volume_integral::VolumeIntegralShockCapturingHG,
#                                dg::DGSEM, cache)
# end

# TODO: Taal implement
# function calc_volume_integral!(du::AbstractArray{<:Any,4}, u, nonconservative_terms::Val{true}, equations,
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
#                               nonconservative_terms::Val{true}, equations,
#                               dg::DG, cache)
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


function prolong2mortars!(cache, u::AbstractArray{<:Any,4}, equations, mortar_l2::LobattoLegendreMortarL2, dg::DGSEM)

  Threads.@threads for mortar in eachmortar(dg, cache)

    large_element = cache.mortars.neighbor_ids[3, mortar]
    upper_element = cache.mortars.neighbor_ids[2, mortar]
    lower_element = cache.mortars.neighbor_ids[1, mortar]

    # Copy solution small to small
    if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
      if cache.mortars.orientations[mortar] == 1
        # L2 mortars in x-direction
        for l in eachnode(dg)
          for v in eachvariable(equations)
            cache.mortars.u_upper[2, v, l, mortar] = u[v, 1, l, upper_element]
            cache.mortars.u_lower[2, v, l, mortar] = u[v, 1, l, lower_element]
          end
        end
      else
        # L2 mortars in y-direction
        for l in eachnode(dg)
          for v in eachvariable(equations)
            cache.mortars.u_upper[2, v, l, mortar] = u[v, l, 1, upper_element]
            cache.mortars.u_lower[2, v, l, mortar] = u[v, l, 1, lower_element]
          end
        end
      end
    else # large_sides[mortar] == 2 -> small elements on left side
      if cache.mortars.orientations[mortar] == 1
        # L2 mortars in x-direction
        for l in eachnode(dg)
          for v in eachvariable(equations)
            cache.mortars.u_upper[1, v, l, mortar] = u[v, nnodes(dg), l, upper_element]
            cache.mortars.u_lower[1, v, l, mortar] = u[v, nnodes(dg), l, lower_element]
          end
        end
      else
        # L2 mortars in y-direction
        for l in eachnode(dg)
          for v in eachvariable(equations)
            cache.mortars.u_upper[1, v, l, mortar] = u[v, l, nnodes(dg), upper_element]
            cache.mortars.u_lower[1, v, l, mortar] = u[v, l, nnodes(dg), lower_element]
          end
        end
      end
    end

    # Interpolate large element face data to small interface locations
    if cache.mortars.large_sides[mortar] == 1 # -> large element on left side
      leftright = 1
      if cache.mortars.orientations[mortar] == 1
        # L2 mortars in x-direction
        u_large = view(u, :, nnodes(dg), :, large_element)
        element_solutions_to_mortars!(cache, mortar_l2, leftright, mortar, u_large)
      else
        # L2 mortars in y-direction
        u_large = view(u, :, :, nnodes(dg), large_element)
        element_solutions_to_mortars!(cache, mortar_l2, leftright, mortar, u_large)
      end
    else # large_sides[mortar] == 2 -> large element on right side
      leftright = 2
      if cache.mortars.orientations[mortar] == 1
        # L2 mortars in x-direction
        u_large = view(u, :, 1, :, large_element)
        element_solutions_to_mortars!(cache, mortar_l2, leftright, mortar, u_large)
      else
        # L2 mortars in y-direction
        u_large = view(u, :, :, 1, large_element)
        element_solutions_to_mortars!(cache, mortar_l2, leftright, mortar, u_large)
      end
    end
  end

  return nothing
end

@inline function element_solutions_to_mortars!(cache, mortar_l2::LobattoLegendreMortarL2, leftright, mortar,
                                               u_large::AbstractArray{<:Any,2})
  multiply_dimensionwise!(view(cache.mortars.u_upper, leftright, :, :, mortar), mortar_l2.forward_upper, u_large)
  multiply_dimensionwise!(view(cache.mortars.u_lower, leftright, :, :, mortar), mortar_l2.forward_lower, u_large)
  return nothing
end


function calc_mortar_flux!(surface_flux_values, nonconservative_terms::Val{false}, equations,
                           mortar_l2::LobattoLegendreMortarL2, dg::DG, cache)
  @unpack neighbor_ids, u_lower, u_upper, orientations = cache.mortars
  @unpack fstar_upper_threaded, fstar_lower_threaded = cache

  Threads.@threads for mortar in eachmortar(dg, cache)
    # Choose thread-specific pre-allocated container
    fstar_upper = fstar_upper_threaded[Threads.threadid()]
    fstar_lower = fstar_lower_threaded[Threads.threadid()]

    # Calculate fluxes
    orientation = orientations[mortar]
    calc_fstar!(fstar_upper, equations, dg, u_upper, mortar, orientation)
    calc_fstar!(fstar_lower, equations, dg, u_lower, mortar, orientation)

    mortar_fluxes_to_elements!(surface_flux_values, equations, mortar_l2, dg, cache,
                               mortar, fstar_upper, fstar_lower)
  end

  return nothing
end

# TODO: Taal implement
# function calc_mortar_flux!(surface_flux_values, nonconservative_terms::Val{true}, equations,
#                            mortar_l2::LobattoLegendreMortarL2, dg::DG, cache)
# end

@inline function calc_fstar!(destination::AbstractArray{<:Any,2}, equations, dg::DGSEM, u_interfaces, mortar, orientation)
  @unpack surface_flux = dg

  for i in eachnode(dg)
    # Call pointwise two-point numerical flux function
    u_ll, u_rr = get_surface_node_vars(u_interfaces, equations, dg, i, mortar)
    flux = surface_flux(u_ll, u_rr, orientation, equations)

    # Copy flux to left and right element storage
    set_node_vars!(destination, flux, equations, dg, i)
  end

  return nothing
end

@inline function mortar_fluxes_to_elements!(surface_flux_values, equations, mortar_l2::LobattoLegendreMortarL2, dg::DGSEM, cache,
                                            mortar, fstar_upper, fstar_lower)
  large_element = cache.mortars.neighbor_ids[3, mortar]
  upper_element = cache.mortars.neighbor_ids[2, mortar]
  lower_element = cache.mortars.neighbor_ids[1, mortar]

  # Copy flux small to small
  if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
    if cache.mortars.orientations[mortar] == 1
      # L2 mortars in x-direction
      direction = 1
    else
      # L2 mortars in y-direction
      direction = 3
    end
  else # large_sides[mortar] == 2 -> small elements on left side
    if cache.mortars.orientations[mortar] == 1
      # L2 mortars in x-direction
      direction = 2
    else
      # L2 mortars in y-direction
      direction = 4
    end
  end
  surface_flux_values[:, :, direction, upper_element] .= fstar_upper
  surface_flux_values[:, :, direction, lower_element] .= fstar_lower

  # Project small fluxes to large element
  if cache.mortars.large_sides[mortar] == 1 # -> large element on left side
    if cache.mortars.orientations[mortar] == 1
      # L2 mortars in x-direction
      direction = 2
    else
      # L2 mortars in y-direction
      direction = 4
    end
  else # large_sides[mortar] == 2 -> large element on right side
    if cache.mortars.orientations[mortar] == 1
      # L2 mortars in x-direction
      direction = 1
    else
      # L2 mortars in y-direction
      direction = 3
    end
  end

  for v in eachvariable(equations)
    @views surface_flux_values[v, :, direction, large_element] .=
      (mortar_l2.reverse_upper * fstar_upper[v, :] + mortar_l2.reverse_lower * fstar_lower[v, :])
  end
  # The code above could be replaced by the following code. However, the relative efficiency
  # depends on the types of fstar_upper/fstar_lower and dg.l2mortar_reverse_upper.
  # Using StaticArrays for both makes the code above faster for common test cases.
  # multiply_dimensionwise!(
  #   view(surface_flux_values, :, :, direction, large_element), mortar_l2.reverse_upper, fstar_upper,
  #                                                              mortar_l2.reverse_lower, fstar_lower)

  return nothing
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
