
# everything related to a DG semidiscretization in 3D,
# currently limited to Lobatto-Legendre nodes

# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::TreeMesh{3}, equations::AbstractEquations{3},
                      dg::DG, RealT)
  # Get cells for which an element needs to be created (i.e. all leaf cells)
  leaf_cell_ids = local_leaf_cells(mesh.tree)

  elements = init_elements(leaf_cell_ids, mesh, equations, dg.basis, RealT)

  interfaces = init_interfaces(leaf_cell_ids, mesh, elements)

  boundaries = init_boundaries(leaf_cell_ids, mesh, elements)

  mortars = init_mortars(leaf_cell_ids, mesh, elements, dg.mortar)

  cache = (; elements, interfaces, boundaries, mortars)

  # Add specialized parts of the cache required to compute the volume integral etc.
  cache = (;cache..., create_cache(mesh, equations, dg.volume_integral, dg)...)
  cache = (;cache..., create_cache(mesh, equations, dg.mortar)...)

  return cache
end


# The methods below are specialized on the volume integral type
# and called from the basic `create_cache` method at the top.
function create_cache(mesh::TreeMesh{3}, equations, volume_integral::VolumeIntegralFluxDifferencing, dg::DG)
  create_cache(mesh, have_nonconservative_terms(equations), equations, volume_integral, dg)
end

function create_cache(mesh::TreeMesh{3}, nonconservative_terms::Val{false}, equations, ::VolumeIntegralFluxDifferencing, dg)
  NamedTuple()
end

function create_cache(mesh::TreeMesh{3}, nonconservative_terms::Val{true}, equations, ::VolumeIntegralFluxDifferencing, dg)

  A = Array{real(dg), 5}
  f1_threaded = A[A(undef, nvariables(equations), nnodes(dg), nnodes(dg), nnodes(dg), nnodes(dg))
                  for _ in 1:Threads.nthreads()]
  f2_threaded = A[A(undef, nvariables(equations), nnodes(dg), nnodes(dg), nnodes(dg), nnodes(dg))
                  for _ in 1:Threads.nthreads()]
  f3_threaded = A[A(undef, nvariables(equations), nnodes(dg), nnodes(dg), nnodes(dg), nnodes(dg))
                  for _ in 1:Threads.nthreads()]

  A3d = Array{real(dg), 3}
  fstar_upper_left_threaded  = A3d[A3d(undef, nvariables(equations), nnodes(dg), nnodes(dg))
                                   for _ in 1:Threads.nthreads()]
  fstar_upper_right_threaded = A3d[A3d(undef, nvariables(equations), nnodes(dg), nnodes(dg))
                                   for _ in 1:Threads.nthreads()]
  fstar_lower_left_threaded  = A3d[A3d(undef, nvariables(equations), nnodes(dg), nnodes(dg))
                                   for _ in 1:Threads.nthreads()]
  fstar_lower_right_threaded = A3d[A3d(undef, nvariables(equations), nnodes(dg), nnodes(dg))
                                   for _ in 1:Threads.nthreads()]
  fstar_tmp1_threaded        = A3d[A3d(undef, nvariables(equations), nnodes(dg), nnodes(dg))
                                   for _ in 1:Threads.nthreads()]
  noncons_diamond_upper_left_threaded  = A3d[A3d(undef, nvariables(equations), nnodes(dg), nnodes(dg))
                                             for _ in 1:Threads.nthreads()]
  noncons_diamond_upper_right_threaded = A3d[A3d(undef, nvariables(equations), nnodes(dg), nnodes(dg))
                                             for _ in 1:Threads.nthreads()]
  noncons_diamond_lower_left_threaded  = A3d[A3d(undef, nvariables(equations), nnodes(dg), nnodes(dg))
                                             for _ in 1:Threads.nthreads()]
  noncons_diamond_lower_right_threaded = A3d[A3d(undef, nvariables(equations), nnodes(dg), nnodes(dg))
                                             for _ in 1:Threads.nthreads()]

  return (; f1_threaded, f2_threaded, f3_threaded,
            fstar_upper_left_threaded, fstar_upper_right_threaded,
            fstar_lower_left_threaded, fstar_lower_right_threaded,
            fstar_tmp1_threaded,
            noncons_diamond_upper_left_threaded, noncons_diamond_upper_right_threaded,
            noncons_diamond_lower_left_threaded, noncons_diamond_lower_right_threaded,)
end


function create_cache(mesh::TreeMesh{3}, equations,
                      volume_integral::VolumeIntegralShockCapturingHG, dg::DG)
  element_ids_dg   = Int[]
  element_ids_dgfv = Int[]

  cache = create_cache(mesh, equations,
                       VolumeIntegralFluxDifferencing(volume_integral.volume_flux_dg),
                       dg)

  A4dp1_x = Array{real(dg), 4}
  A4dp1_y = Array{real(dg), 4}
  A4dp1_z = Array{real(dg), 4}
  fstar1_threaded  = A4dp1_x[A4dp1_x(undef, nvariables(equations), nnodes(dg)+1, nnodes(dg), nnodes(dg))
                             for _ in 1:Threads.nthreads()]
  fstar2_threaded  = A4dp1_y[A4dp1_y(undef, nvariables(equations), nnodes(dg), nnodes(dg)+1, nnodes(dg))
                             for _ in 1:Threads.nthreads()]
  fstar3_threaded  = A4dp1_z[A4dp1_y(undef, nvariables(equations), nnodes(dg), nnodes(dg), nnodes(dg)+1)
                             for _ in 1:Threads.nthreads()]

  return (; cache..., element_ids_dg, element_ids_dgfv, fstar1_threaded, fstar2_threaded, fstar3_threaded)
end


# The methods below are specialized on the mortar type
# and called from the basic `create_cache` method at the top.
function create_cache(mesh::TreeMesh{3}, equations, mortar_l2::LobattoLegendreMortarL2)
  # TODO: Taal compare performance of different types
  A3d = Array{real(mortar_l2), 3}
  fstar_upper_left_threaded  = A3d[A3d(undef, nvariables(equations), nnodes(mortar_l2), nnodes(mortar_l2))
                                   for _ in 1:Threads.nthreads()]
  fstar_upper_right_threaded = A3d[A3d(undef, nvariables(equations), nnodes(mortar_l2), nnodes(mortar_l2))
                                   for _ in 1:Threads.nthreads()]
  fstar_lower_left_threaded  = A3d[A3d(undef, nvariables(equations), nnodes(mortar_l2), nnodes(mortar_l2))
                                   for _ in 1:Threads.nthreads()]
  fstar_lower_right_threaded = A3d[A3d(undef, nvariables(equations), nnodes(mortar_l2), nnodes(mortar_l2))
                                   for _ in 1:Threads.nthreads()]
  fstar_tmp1_threaded        = A3d[A3d(undef, nvariables(equations), nnodes(mortar_l2), nnodes(mortar_l2))
                                   for _ in 1:Threads.nthreads()]

  (; fstar_upper_left_threaded, fstar_upper_right_threaded,
     fstar_lower_left_threaded, fstar_lower_right_threaded,
     fstar_tmp1_threaded)
end


@inline function wrap_array(u_ode::AbstractVector, mesh::TreeMesh{3}, equations, dg::DG, cache)
  @boundscheck begin
    @assert length(u_ode) == nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache)
  end
  # We would like to use
  #   reshape(u_ode, (nvariables(equations), nnodes(dg), nnodes(dg), nnodes(dg), nelements(dg, cache)))
  # but that results in
  #   ERROR: LoadError: cannot resize array with shared data
  # when we resize! `u_ode` during AMR.

  # The following version is fast and allows us to `resize!(u_ode, ...)`.
  # OBS! Remember to `GC.@preserve` temporaries such as copies of `u_ode`
  #      and other stuff that is only used indirectly via `wrap_array` afterwards!
  unsafe_wrap(Array{eltype(u_ode), ndims(mesh)+2}, pointer(u_ode),
              (nvariables(equations), nnodes(dg), nnodes(dg), nnodes(dg), nelements(dg, cache)))
end


function compute_coefficients!(u, func, t, mesh::TreeMesh{3}, equations, dg::DG, cache)

  @threaded for element in eachelement(dg, cache)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      x_node = get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, k, element)
      u_node = func(x_node, t, equations)
      set_node_vars!(u, u_node, equations, dg, i, j, k, element)
    end
  end
end

# TODO: Taal discuss/refactor timer, allowing users to pass a custom timer?

function rhs!(du::AbstractArray{<:Any,5}, u, t,
              mesh::TreeMesh{3}, equations,
              initial_condition, boundary_conditions::BoundaryConditions,
              source_terms::SourceTerms,
              dg::DG, cache) where {BoundaryConditions, SourceTerms}
  # Reset du
  @timeit_debug timer() "reset ∂u/∂t" du .= zero(eltype(du))

  # Calculate volume integral
  @timeit_debug timer() "volume integral" calc_volume_integral!(du, u, have_nonconservative_terms(equations), equations,
                                                                dg.volume_integral, dg, cache)

  # Prolong solution to interfaces
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


function calc_volume_integral!(du::AbstractArray{<:Any,5}, u,
                               nonconservative_terms::Val{false}, equations,
                               volume_integral::VolumeIntegralWeakForm,
                               dg::DGSEM, cache)
  @unpack derivative_dhat = dg.basis

  @threaded for element in eachelement(dg, cache)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, k, element)

      flux1 = calcflux(u_node, 1, equations)
      for ii in eachnode(dg)
        integral_contribution = derivative_dhat[ii, i] * flux1
        add_to_node_vars!(du, integral_contribution, equations, dg, ii, j, k, element)
      end

      flux2 = calcflux(u_node, 2, equations)
      for jj in eachnode(dg)
        integral_contribution = derivative_dhat[jj, j] * flux2
        add_to_node_vars!(du, integral_contribution, equations, dg, i, jj, k, element)
      end

      flux3 = calcflux(u_node, 3, equations)
      for kk in eachnode(dg)
        integral_contribution = derivative_dhat[kk, k] * flux3
        add_to_node_vars!(du, integral_contribution, equations, dg, i, j, kk, element)
      end
    end
  end

  return nothing
end


# Calculate 3D twopoint flux (element version)
@inline function calcflux_twopoint!(f1, f2, f3, u::AbstractArray{<:Any,5}, element,
                                    volume_flux, equations, dg::DG, cache)

  for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
    # Set diagonal entries (= regular volume fluxes due to consistency)
    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    flux1 = calcflux(u_node, 1, equations)
    flux2 = calcflux(u_node, 2, equations)
    flux3 = calcflux(u_node, 3, equations)
    set_node_vars!(f1, flux1, equations, dg, i, i, j, k)
    set_node_vars!(f2, flux2, equations, dg, j, i, j, k)
    set_node_vars!(f3, flux3, equations, dg, k, i, j, k)

    # Flux in x-direction
    for ii in (i+1):nnodes(dg)
      u_ll = get_node_vars(u, equations, dg, i,  j, k, element)
      u_rr = get_node_vars(u, equations, dg, ii, j, k, element)
      flux = volume_flux(u_ll, u_rr, 1, equations) # 1-> x-direction
      set_node_vars!(f1, flux, equations, dg, i, ii, j, k)
      set_node_vars!(f1, flux, equations, dg, ii, i, j, k)
    end

    # Flux in y-direction
    for jj in (j+1):nnodes(dg)
      u_ll = get_node_vars(u, equations, dg, i, j,  k, element)
      u_rr = get_node_vars(u, equations, dg, i, jj, k, element)
      flux = volume_flux(u_ll, u_rr, 2, equations) # 2 -> y-direction
      set_node_vars!(f2, flux, equations, dg, j, i, jj, k)
      set_node_vars!(f2, flux, equations, dg, jj, i, j, k)
    end

    # Flux in z-direction
    for kk in (k+1):nnodes(dg)
      u_ll = get_node_vars(u, equations, dg, i, j, k,  element)
      u_rr = get_node_vars(u, equations, dg, i, j, kk, element)
      flux = volume_flux(u_ll, u_rr, 3, equations) # 3 -> z-direction
      set_node_vars!(f3, flux, equations, dg, k, i, j, kk)
      set_node_vars!(f3, flux, equations, dg, kk, i, j, k)
    end
  end

  calcflux_twopoint_nonconservative!(f1, f2, f3, u, element,
                                     have_nonconservative_terms(equations),
                                     equations, dg, cache)
end

function calcflux_twopoint_nonconservative!(f1, f2, f3, u::AbstractArray{<:Any,5}, element,
                                            nonconservative_terms::Val{false},
                                            equations, dg::DG, cache)
  return nothing
end

function calcflux_twopoint_nonconservative!(f1, f2, f3, u::AbstractArray{<:Any,5}, element,
                                            nonconservative_terms::Val{true},
                                            equations, dg::DG, cache)
  #TODO: Create a unified interface, e.g. using non-symmetric two-point (extended) volume fluxes
  #      For now, just dispatch to an existing function for the IdealMhdEquations
  calcflux_twopoint_nonconservative!(f1, f2, f3, u, element, equations, dg, cache)
end


function calc_volume_integral!(du::AbstractArray{<:Any,5}, u,
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralFluxDifferencing,
                               dg::DGSEM, cache)
  @threaded for element in eachelement(dg, cache)
    split_form_kernel!(du, u, nonconservative_terms, equations, volume_integral.volume_flux, dg, cache, element)
  end
end

@inline function split_form_kernel!(du::AbstractArray{<:Any,5}, u,
                                    nonconservative_terms::Val{false}, equations,
                                    volume_flux, dg::DGSEM, cache,
                                    element, alpha=true)
  # true * [some floating point value] == [exactly the same floating point value]
  # This can (hopefully) be optimized away due to constant propagation.
  @unpack derivative_split = dg.basis

  # Calculate volume integral in one element
  for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
    u_node = get_node_vars(u, equations, dg, i, j, k, element)

    # x direction
    # use consistency of the volume flux to make this evaluation cheaper
    flux = calcflux(u_node, 1, equations)
    integral_contribution = alpha * derivative_split[i, i] * flux
    add_to_node_vars!(du, integral_contribution, equations, dg, i, j, k, element)
    # use symmetry of the volume flux for the remaining terms
    for ii in (i+1):nnodes(dg)
      u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
      flux = volume_flux(u_node, u_node_ii, 1, equations)
      integral_contribution = alpha * derivative_split[i, ii] * flux
      add_to_node_vars!(du, integral_contribution, equations, dg, i,  j, k, element)
      integral_contribution = alpha * derivative_split[ii, i] * flux
      add_to_node_vars!(du, integral_contribution, equations, dg, ii, j, k, element)
    end

    # y direction
    # use consistency of the volume flux to make this evaluation cheaper
    flux = calcflux(u_node, 2, equations)
    integral_contribution = alpha * derivative_split[j, j] * flux
    add_to_node_vars!(du, integral_contribution, equations, dg, i, j, k, element)
    # use symmetry of the volume flux for the remaining terms
    for jj in (j+1):nnodes(dg)
      u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
      flux = volume_flux(u_node, u_node_jj, 2, equations)
      integral_contribution = alpha * derivative_split[j, jj] * flux
      add_to_node_vars!(du, integral_contribution, equations, dg, i, j,  k, element)
      integral_contribution = alpha * derivative_split[jj, j] * flux
      add_to_node_vars!(du, integral_contribution, equations, dg, i, jj, k, element)
    end

    # z direction
    # use consistency of the volume flux to make this evaluation cheaper
    flux = calcflux(u_node, 3, equations)
    integral_contribution = alpha * derivative_split[k, k] * flux
    add_to_node_vars!(du, integral_contribution, equations, dg, i, j, k, element)
    # use symmetry of the volume flux for the remaining terms
    for kk in (k+1):nnodes(dg)
      u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
      flux = volume_flux(u_node, u_node_kk, 3, equations)
      integral_contribution = alpha * derivative_split[k, kk] * flux
      add_to_node_vars!(du, integral_contribution, equations, dg, i, j, k,  element)
      integral_contribution = alpha * derivative_split[kk, k] * flux
      add_to_node_vars!(du, integral_contribution, equations, dg, i, j, kk, element)
    end
  end
end

@inline function split_form_kernel!(du::AbstractArray{<:Any,5}, u,
                                    nonconservative_terms::Val{true}, equations,
                                    volume_flux, dg::DGSEM, cache,
                                    element, alpha=true)
  @unpack derivative_split_transpose = dg.basis
  @unpack f1_threaded, f2_threaded, f3_threaded = cache

  # Choose thread-specific pre-allocated container
  f1 = f1_threaded[Threads.threadid()]
  f2 = f2_threaded[Threads.threadid()]
  f3 = f3_threaded[Threads.threadid()]

  # Calculate volume fluxes (one more dimension than weak form)
  calcflux_twopoint!(f1, f2, f3, u, element, volume_flux, equations, dg, cache)

  # Calculate volume integral in one element
  for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
    for v in eachvariable(equations)
      # Use local accumulator to improve performance
      acc = zero(eltype(du))
      for l in eachnode(dg)
        acc += (derivative_split_transpose[l, i] * f1[v, l, i, j, k] +
                derivative_split_transpose[l, j] * f2[v, l, i, j, k] +
                derivative_split_transpose[l, k] * f3[v, l, i, j, k] )
      end
      du[v, i, j, k, element] += alpha * acc
    end
  end

  return nothing
end


# TODO: Taal dimension agnostic
function calc_volume_integral!(du::AbstractArray{<:Any,5}, u, nonconservative_terms, equations,
                               volume_integral::VolumeIntegralShockCapturingHG,
                               dg::DGSEM, cache)
  @unpack element_ids_dg, element_ids_dgfv = cache
  @unpack volume_flux_dg, volume_flux_fv, indicator = volume_integral

  # Calculate blending factors α: u = u_DG * (1 - α) + u_FV * α
  alpha = @timeit_debug timer() "blending factors" indicator(u, equations, dg, cache)

  # Determine element ids for DG-only and blended DG-FV volume integral
  pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, dg, cache)

  # Loop over pure DG elements
  @timeit_debug timer() "pure DG" @threaded for element in element_ids_dg
    split_form_kernel!(du, u, nonconservative_terms, equations, volume_flux_dg, dg, cache, element)
  end

  # Loop over blended DG-FV elements
  @timeit_debug timer() "blended DG-FV" @threaded for element in element_ids_dgfv
    alpha_element = alpha[element]

    # Calculate DG volume integral contribution
    split_form_kernel!(du, u, nonconservative_terms, equations, volume_flux_dg, dg, cache, element, 1 - alpha_element)

    # Calculate FV volume integral contribution
    fv_kernel!(du, u, equations, volume_flux_fv, dg, cache, element, alpha_element)
  end

  return nothing
end

@inline function fv_kernel!(du::AbstractArray{<:Any,5}, u::AbstractArray{<:Any,5},
                            equations, volume_flux_fv, dg::DGSEM, cache, element, alpha=true)
  @unpack fstar1_threaded, fstar2_threaded, fstar3_threaded = cache
  @unpack inverse_weights = dg.basis

  # Calculate FV two-point fluxes
  fstar1 = fstar1_threaded[Threads.threadid()]
  fstar2 = fstar2_threaded[Threads.threadid()]
  fstar3 = fstar3_threaded[Threads.threadid()]
  calcflux_fv!(fstar1, fstar2, fstar3, u, equations, volume_flux_fv, dg, element)

  # Calculate FV volume integral contribution
  for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
    for v in eachvariable(equations)
      du[v, i, j, k, element] += ( alpha *
                                   (inverse_weights[i] * (fstar1[v, i+1, j, k] - fstar1[v, i, j, k]) +
                                    inverse_weights[j] * (fstar2[v, i, j+1, k] - fstar2[v, i, j, k]) +
                                    inverse_weights[k] * (fstar3[v, i, j, k+1] - fstar3[v, i, j, k])) )

    end
  end

  return nothing
end

@inline function calcflux_fv!(fstar1, fstar2, fstar3, u::AbstractArray{<:Any,5},
                              equations, volume_flux_fv, dg::DGSEM, element)

  fstar1[:, 1,            :, :] .= zero(eltype(fstar1))
  fstar1[:, nnodes(dg)+1, :, :] .= zero(eltype(fstar1))

  for k in eachnode(dg), j in eachnode(dg), i in 2:nnodes(dg)
    u_ll = get_node_vars(u, equations, dg, i-1, j, k, element)
    u_rr = get_node_vars(u, equations, dg, i,   j, k, element)
    flux = volume_flux_fv(u_ll, u_rr, 1, equations) # orientation 1: x direction
    set_node_vars!(fstar1, flux, equations, dg, i, j, k)
  end

  fstar2[:, :, 1           , :] .= zero(eltype(fstar2))
  fstar2[:, :, nnodes(dg)+1, :] .= zero(eltype(fstar2))

  for k in eachnode(dg), j in 2:nnodes(dg), i in eachnode(dg)
    u_ll = get_node_vars(u, equations, dg, i, j-1, k, element)
    u_rr = get_node_vars(u, equations, dg, i, j,   k, element)
    flux = volume_flux_fv(u_ll, u_rr, 2, equations) # orientation 2: y direction
    set_node_vars!(fstar2, flux, equations, dg, i, j, k)
  end

  fstar3[:, :, :, 1           ] .= zero(eltype(fstar3))
  fstar3[:, :, :, nnodes(dg)+1] .= zero(eltype(fstar3))

  for k in 2:nnodes(dg), j in eachnode(dg), i in eachnode(dg)
    u_ll = get_node_vars(u, equations, dg, i, j, k-1, element)
    u_rr = get_node_vars(u, equations, dg, i, j, k,   element)
    flux = volume_flux_fv(u_ll, u_rr, 3, equations) # orientation 3: z direction
    set_node_vars!(fstar3, flux, equations, dg, i, j, k)
  end

  return nothing
end


function prolong2interfaces!(cache, u::AbstractArray{<:Any,5}, equations, dg::DG)
  @unpack interfaces = cache
  @unpack orientations = interfaces

  @threaded for interface in eachinterface(dg, cache)
    left_element  = interfaces.neighbor_ids[1, interface]
    right_element = interfaces.neighbor_ids[2, interface]

    if orientations[interface] == 1
      # interface in x-direction
      for k in eachnode(dg), j in eachnode(dg), v in eachvariable(equations)
        interfaces.u[1, v, j, k, interface] = u[v, nnodes(dg), j, k, left_element]
        interfaces.u[2, v, j, k, interface] = u[v,          1, j, k, right_element]
      end
    elseif orientations[interface] == 2
      # interface in y-direction
      for k in eachnode(dg), i in eachnode(dg), v in eachvariable(equations)
        interfaces.u[1, v, i, k, interface] = u[v, i, nnodes(dg), k, left_element]
        interfaces.u[2, v, i, k, interface] = u[v, i,          1, k, right_element]
      end
    else # if orientations[interface] == 3
      # interface in z-direction
      for j in eachnode(dg), i in eachnode(dg), v in eachvariable(equations)
        interfaces.u[1, v, i, j, interface] = u[v, i, j, nnodes(dg), left_element]
        interfaces.u[2, v, i, j, interface] = u[v, i, j,          1, right_element]
      end
    end
  end

  return nothing
end

function calc_interface_flux!(surface_flux_values::AbstractArray{<:Any,5},
                              nonconservative_terms::Val{false}, equations,
                              dg::DG, cache)
  @unpack surface_flux = dg
  @unpack u, neighbor_ids, orientations = cache.interfaces

  @threaded for interface in eachinterface(dg, cache)
    # Get neighboring elements
    left_id  = neighbor_ids[1, interface]
    right_id = neighbor_ids[2, interface]

    # Determine interface direction with respect to elements:
    # orientation = 1: left -> 2, right -> 1
    # orientation = 2: left -> 4, right -> 3
    # orientation = 3: left -> 6, right -> 5
    left_direction  = 2 * orientations[interface]
    right_direction = 2 * orientations[interface] - 1

    for j in eachnode(dg), i in eachnode(dg)
      # Call pointwise Riemann solver
      u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, j, interface)
      flux = surface_flux(u_ll, u_rr, orientations[interface], equations)

      # Copy flux to left and right element storage
      for v in eachvariable(equations)
        surface_flux_values[v, i, j, left_direction,  left_id]  = flux[v]
        surface_flux_values[v, i, j, right_direction, right_id] = flux[v]
      end
    end
  end
end

function calc_interface_flux!(surface_flux_values::AbstractArray{<:Any,5},
                              nonconservative_terms::Val{true}, equations,
                              dg::DG, cache)
  @unpack u, neighbor_ids, orientations = cache.interfaces
  fstar_threaded                     = cache.fstar_upper_left_threaded
  noncons_diamond_primary_threaded   = cache.noncons_diamond_upper_left_threaded
  noncons_diamond_secondary_threaded = cache.noncons_diamond_upper_right_threaded

  @threaded for interface in eachinterface(dg, cache)
    # Choose thread-specific pre-allocated container
    fstar                     = fstar_threaded[Threads.threadid()]
    noncons_diamond_primary   = noncons_diamond_primary_threaded[Threads.threadid()]
    noncons_diamond_secondary = noncons_diamond_secondary_threaded[Threads.threadid()]

    # Calculate flux
    calc_fstar!(fstar, equations, dg, u, interface, orientations[interface])

    # Compute the nonconservative numerical "flux" along an interface
    # Done twice because left/right orientation matters så
    # 1 -> primary element and 2 -> secondary element
    # See Bohm et al. 2018 for details on the nonconservative diamond "flux"
    for j in eachnode(dg), i in eachnode(dg)
      # Call pointwise nonconservative term
      u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, j, interface)
      noncons_primary   = noncons_interface_flux(u_ll, u_rr, orientations[interface], equations)
      noncons_secondary = noncons_interface_flux(u_rr, u_ll, orientations[interface], equations)
      # Save to primary and secondary temporay storage
      set_node_vars!(noncons_diamond_primary,   noncons_primary,   equations, dg, i, j)
      set_node_vars!(noncons_diamond_secondary, noncons_secondary, equations, dg, i, j)
    end

    # Get neighboring elements
    left_neighbor  = neighbor_ids[1, interface]
    right_neighbor = neighbor_ids[2, interface]

    # Determine interface direction with respect to elements:
    # orientation = 1: left -> 2, right -> 1
    # orientation = 2: left -> 4, right -> 3
    # orientation = 3: left -> 6, right -> 5
    left_neighbor_direction  = 2 * orientations[interface]
    right_neighbor_direction = 2 * orientations[interface] - 1

    # Copy flux to left and right element storage
    for j in eachnode(dg), i in eachnode(dg)
      for v in eachvariable(equations)
        surface_flux_values[v, i, j, left_neighbor_direction,  left_neighbor]  = (fstar[v, i, j] +
            noncons_diamond_primary[v, i, j])
        surface_flux_values[v, i, j, right_neighbor_direction, right_neighbor] = (fstar[v, i, j] +
            noncons_diamond_secondary[v, i, j])
      end
    end
  end

  return nothing
end


function prolong2boundaries!(cache, u::AbstractArray{<:Any,5}, equations, dg::DG)
  @unpack boundaries = cache
  @unpack orientations, neighbor_sides = boundaries

  @threaded for boundary in eachboundary(dg, cache)
    element = boundaries.neighbor_ids[boundary]

    if orientations[boundary] == 1
      # boundary in x-direction
      if neighbor_sides[boundary] == 1
        # element in -x direction of boundary
        for k in eachnode(dg), j in eachnode(dg), v in eachvariable(equations)
          boundaries.u[1, v, j, k, boundary] = u[v, nnodes(dg), j, k, element]
        end
      else # Element in +x direction of boundary
        for k in eachnode(dg), j in eachnode(dg), v in eachvariable(equations)
          boundaries.u[2, v, j, k, boundary] = u[v, 1,          j, k, element]
        end
      end
    elseif orientations[boundary] == 2
      # boundary in y-direction
      if neighbor_sides[boundary] == 1
        # element in -y direction of boundary
        for k in eachnode(dg), i in eachnode(dg), v in eachvariable(equations)
          boundaries.u[1, v, i, k, boundary] = u[v, i, nnodes(dg), k, element]
        end
      else
        # element in +y direction of boundary
        for k in eachnode(dg), i in eachnode(dg), v in eachvariable(equations)
          boundaries.u[2, v, i, k, boundary] = u[v, i, 1,          k, element]
        end
      end
    else #if orientations[boundary] == 3
      # boundary in z-direction
      if neighbor_sides[boundary] == 1
        # element in -z direction of boundary
        for j in eachnode(dg), i in eachnode(dg), v in eachvariable(equations)
          boundaries.u[1, v, i, j, boundary] = u[v, i, j, nnodes(dg), element]
        end
      else
        # element in +z direction of boundary
        for j in eachnode(dg), i in eachnode(dg), v in eachvariable(equations)
          boundaries.u[2, v, i, j, boundary] = u[v, i, j, 1,          element]
        end
      end
    end
  end

  return nothing
end

# TODO: Taal dimension agnostic
function calc_boundary_flux!(cache, t, boundary_condition::BoundaryConditionPeriodic,
                             equations::AbstractEquations{3}, dg::DG)
  @assert isempty(eachboundary(dg, cache))
end

# TODO: Taal dimension agnostic
function calc_boundary_flux!(cache, t, boundary_condition,
                             equations::AbstractEquations{3}, dg::DG)
  @unpack surface_flux_values = cache.elements
  @unpack n_boundaries_per_direction = cache.boundaries

  # Calculate indices
  lasts = accumulate(+, n_boundaries_per_direction)
  firsts = lasts - n_boundaries_per_direction .+ 1

  # Calc boundary fluxes in each direction
  for direction in eachindex(firsts)
    calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_condition,
                                     equations, dg, cache,
                                     direction, firsts[direction], lasts[direction])
  end
end

function calc_boundary_flux!(cache, t, boundary_conditions::Union{NamedTuple,Tuple},
                             equations::AbstractEquations{3}, dg::DG)
  @unpack surface_flux_values = cache.elements
  @unpack n_boundaries_per_direction = cache.boundaries

  # Calculate indices
  lasts = accumulate(+, n_boundaries_per_direction)
  firsts = lasts - n_boundaries_per_direction .+ 1

  # Calc boundary fluxes in each direction
  calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[1],
                                   equations, dg, cache, 1, firsts[1], lasts[1])
  calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[2],
                                   equations, dg, cache, 2, firsts[2], lasts[2])
  calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[3],
                                   equations, dg, cache, 3, firsts[3], lasts[3])
  calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[4],
                                   equations, dg, cache, 4, firsts[4], lasts[4])
  calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[5],
                                   equations, dg, cache, 5, firsts[5], lasts[5])
  calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[6],
                                   equations, dg, cache, 6, firsts[6], lasts[6])
end

function calc_boundary_flux_by_direction!(surface_flux_values::AbstractArray{<:Any,5}, t,
                                          boundary_condition, equations, dg::DG, cache,
                                          direction, first_boundary, last_boundary)
  @unpack surface_flux = dg
  @unpack u, neighbor_ids, neighbor_sides, node_coordinates, orientations = cache.boundaries

  @threaded for boundary in first_boundary:last_boundary
    # Get neighboring element
    neighbor = neighbor_ids[boundary]

    for j in eachnode(dg), i in eachnode(dg)
      # Get boundary flux
      u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, j, boundary)
      if neighbor_sides[boundary] == 1 # Element is on the left, boundary on the right
        u_inner = u_ll
      else # Element is on the right, boundary on the left
        u_inner = u_rr
      end
      x = get_node_coords(node_coordinates, equations, dg, i, j, boundary)
      flux = boundary_condition(u_inner, orientations[boundary], direction, x, t, surface_flux,
                                equations)

      # Copy flux to left and right element storage
      for v in eachvariable(equations)
        surface_flux_values[v, i, j, direction, neighbor] = flux[v]
      end
    end
  end

  return nothing
end


function prolong2mortars!(cache, u::AbstractArray{<:Any,5}, equations,
                          mortar_l2::LobattoLegendreMortarL2, dg::DGSEM)
  # temporary buffer for projections
  @unpack fstar_tmp1_threaded = cache

  @threaded for mortar in eachmortar(dg, cache)
    fstar_tmp1 = fstar_tmp1_threaded[Threads.threadid()]

    lower_left_element  = cache.mortars.neighbor_ids[1, mortar]
    lower_right_element = cache.mortars.neighbor_ids[2, mortar]
    upper_left_element  = cache.mortars.neighbor_ids[3, mortar]
    upper_right_element = cache.mortars.neighbor_ids[4, mortar]
    large_element       = cache.mortars.neighbor_ids[5, mortar]

    # Copy solution small to small
    if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
      if cache.mortars.orientations[mortar] == 1
        # L2 mortars in x-direction
        for k in eachnode(dg), j in eachnode(dg)
          for v in eachvariable(equations)
            cache.mortars.u_upper_left[2, v, j, k, mortar]  = u[v, 1, j, k, upper_left_element]
            cache.mortars.u_upper_right[2, v, j, k, mortar] = u[v, 1, j, k, upper_right_element]
            cache.mortars.u_lower_left[2, v, j, k, mortar]  = u[v, 1, j, k, lower_left_element]
            cache.mortars.u_lower_right[2, v, j, k, mortar] = u[v, 1, j, k, lower_right_element]
          end
        end
      elseif cache.mortars.orientations[mortar] == 2
        # L2 mortars in y-direction
        for k in eachnode(dg), i in eachnode(dg)
          for v in eachvariable(equations)
            cache.mortars.u_upper_left[2, v, i, k, mortar]  = u[v, i, 1, k, upper_left_element]
            cache.mortars.u_upper_right[2, v, i, k, mortar] = u[v, i, 1, k, upper_right_element]
            cache.mortars.u_lower_left[2, v, i, k, mortar]  = u[v, i, 1, k, lower_left_element]
            cache.mortars.u_lower_right[2, v, i, k, mortar] = u[v, i, 1, k, lower_right_element]
          end
        end
      else # orientations[mortar] == 3
        # L2 mortars in z-direction
        for j in eachnode(dg), i in eachnode(dg)
          for v in eachvariable(equations)
            cache.mortars.u_upper_left[2, v, i, j, mortar]  = u[v, i, j, 1, upper_left_element]
            cache.mortars.u_upper_right[2, v, i, j, mortar] = u[v, i, j, 1, upper_right_element]
            cache.mortars.u_lower_left[2, v, i, j, mortar]  = u[v, i, j, 1, lower_left_element]
            cache.mortars.u_lower_right[2, v, i, j, mortar] = u[v, i, j, 1, lower_right_element]
          end
        end
      end
    else # large_sides[mortar] == 2 -> small elements on left side
      if cache.mortars.orientations[mortar] == 1
        # L2 mortars in x-direction
        for k in eachnode(dg), j in eachnode(dg)
          for v in eachvariable(equations)
            cache.mortars.u_upper_left[1, v, j, k, mortar]  = u[v, nnodes(dg), j, k, upper_left_element]
            cache.mortars.u_upper_right[1, v, j, k, mortar] = u[v, nnodes(dg), j, k, upper_right_element]
            cache.mortars.u_lower_left[1, v, j, k, mortar]  = u[v, nnodes(dg), j, k, lower_left_element]
            cache.mortars.u_lower_right[1, v, j, k, mortar] = u[v, nnodes(dg), j, k, lower_right_element]
          end
        end
      elseif cache.mortars.orientations[mortar] == 2
        # L2 mortars in y-direction
        for k in eachnode(dg), i in eachnode(dg)
          for v in eachvariable(equations)
            cache.mortars.u_upper_left[1, v, i, k, mortar]  = u[v, i, nnodes(dg), k, upper_left_element]
            cache.mortars.u_upper_right[1, v, i, k, mortar] = u[v, i, nnodes(dg), k, upper_right_element]
            cache.mortars.u_lower_left[1, v, i, k, mortar]  = u[v, i, nnodes(dg), k, lower_left_element]
            cache.mortars.u_lower_right[1, v, i, k, mortar] = u[v, i, nnodes(dg), k, lower_right_element]
          end
        end
      else # if cache.mortars.orientations[mortar] == 3
        # L2 mortars in z-direction
        for j in eachnode(dg), i in eachnode(dg)
          for v in eachvariable(equations)
            cache.mortars.u_upper_left[1, v, i, j, mortar]  = u[v, i, j, nnodes(dg), upper_left_element]
            cache.mortars.u_upper_right[1, v, i, j, mortar] = u[v, i, j, nnodes(dg), upper_right_element]
            cache.mortars.u_lower_left[1, v, i, j, mortar]  = u[v, i, j, nnodes(dg), lower_left_element]
            cache.mortars.u_lower_right[1, v, i, j, mortar] = u[v, i, j, nnodes(dg), lower_right_element]
          end
        end
      end
    end

    # Interpolate large element face data to small interface locations
    if cache.mortars.large_sides[mortar] == 1 # -> large element on left side
      leftright = 1
      if cache.mortars.orientations[mortar] == 1
        # L2 mortars in x-direction
        u_large = view(u, :, nnodes(dg), :, :, large_element)
        element_solutions_to_mortars!(cache, mortar_l2, leftright, mortar, u_large, fstar_tmp1)
      elseif cache.mortars.orientations[mortar] == 2
        # L2 mortars in y-direction
        u_large = view(u, :, :, nnodes(dg), :, large_element)
        element_solutions_to_mortars!(cache, mortar_l2, leftright, mortar, u_large, fstar_tmp1)
      else # cache.mortars.orientations[mortar] == 3
        # L2 mortars in z-direction
        u_large = view(u, :, :, :, nnodes(dg), large_element)
        element_solutions_to_mortars!(cache, mortar_l2, leftright, mortar, u_large, fstar_tmp1)
      end
    else # large_sides[mortar] == 2 -> large element on right side
      leftright = 2
      if cache.mortars.orientations[mortar] == 1
        # L2 mortars in x-direction
        u_large = view(u, :, 1, :, :, large_element)
        element_solutions_to_mortars!(cache, mortar_l2, leftright, mortar, u_large, fstar_tmp1)
      elseif cache.mortars.orientations[mortar] == 2
        # L2 mortars in y-direction
        u_large = view(u, :, :, 1, :, large_element)
        element_solutions_to_mortars!(cache, mortar_l2, leftright, mortar, u_large, fstar_tmp1)
      else # cache.mortars.orientations[mortar] == 3
        # L2 mortars in z-direction
        u_large = view(u, :, :, :, 1, large_element)
        element_solutions_to_mortars!(cache, mortar_l2, leftright, mortar, u_large, fstar_tmp1)
      end
    end
  end

  return nothing
end

@inline function element_solutions_to_mortars!(cache, mortar_l2::LobattoLegendreMortarL2, leftright, mortar,
                                               u_large::AbstractArray{<:Any,3}, fstar_tmp1)
  multiply_dimensionwise!(view(cache.mortars.u_upper_left,  leftright, :, :, :, mortar), mortar_l2.forward_lower, mortar_l2.forward_upper, u_large, fstar_tmp1)
  multiply_dimensionwise!(view(cache.mortars.u_upper_right, leftright, :, :, :, mortar), mortar_l2.forward_upper, mortar_l2.forward_upper, u_large, fstar_tmp1)
  multiply_dimensionwise!(view(cache.mortars.u_lower_left,  leftright, :, :, :, mortar), mortar_l2.forward_lower, mortar_l2.forward_lower, u_large, fstar_tmp1)
  multiply_dimensionwise!(view(cache.mortars.u_lower_right, leftright, :, :, :, mortar), mortar_l2.forward_upper, mortar_l2.forward_lower, u_large, fstar_tmp1)
  return nothing
end


function calc_mortar_flux!(surface_flux_values::AbstractArray{<:Any,5},
                           nonconservative_terms::Val{false}, equations,
                           mortar_l2::LobattoLegendreMortarL2, dg::DG, cache)
  @unpack (u_lower_left, u_lower_right, u_upper_left, u_upper_right,
           neighbor_ids, orientations) = cache.mortars
  @unpack (fstar_upper_left_threaded, fstar_upper_right_threaded,
           fstar_lower_left_threaded, fstar_lower_right_threaded,
           fstar_tmp1_threaded) = cache

  @threaded for mortar in eachmortar(dg, cache)
    # Choose thread-specific pre-allocated container
    fstar_upper_left  = fstar_upper_left_threaded[Threads.threadid()]
    fstar_upper_right = fstar_upper_right_threaded[Threads.threadid()]
    fstar_lower_left  = fstar_lower_left_threaded[Threads.threadid()]
    fstar_lower_right = fstar_lower_right_threaded[Threads.threadid()]
    fstar_tmp1        = fstar_tmp1_threaded[Threads.threadid()]

    # Calculate fluxes
    orientation = orientations[mortar]
    calc_fstar!(fstar_upper_left,  equations, dg, u_upper_left,  mortar, orientation)
    calc_fstar!(fstar_upper_right, equations, dg, u_upper_right, mortar, orientation)
    calc_fstar!(fstar_lower_left,  equations, dg, u_lower_left,  mortar, orientation)
    calc_fstar!(fstar_lower_right, equations, dg, u_lower_right, mortar, orientation)

    mortar_fluxes_to_elements!(surface_flux_values, equations, mortar_l2, dg, cache, mortar,
                               fstar_upper_left, fstar_upper_right,
                               fstar_lower_left, fstar_lower_right,
                               fstar_tmp1)
  end

  return nothing
end

function calc_mortar_flux!(surface_flux_values::AbstractArray{<:Any,5},
                           nonconservative_terms::Val{true}, equations,
                           mortar_l2::LobattoLegendreMortarL2, dg::DG, cache)
  @unpack (u_lower_left, u_lower_right, u_upper_left, u_upper_right,
           neighbor_ids, orientations) = cache.mortars
  @unpack (fstar_upper_left_threaded, fstar_upper_right_threaded,
           fstar_lower_left_threaded, fstar_lower_right_threaded,
           noncons_diamond_upper_left_threaded, noncons_diamond_upper_right_threaded,
           noncons_diamond_lower_left_threaded, noncons_diamond_lower_right_threaded,
           fstar_tmp1_threaded) = cache

  @threaded for mortar in eachmortar(dg, cache)
    # Choose thread-specific pre-allocated container
    fstar_upper_left  = fstar_upper_left_threaded[Threads.threadid()]
    fstar_upper_right = fstar_upper_right_threaded[Threads.threadid()]
    fstar_lower_left  = fstar_lower_left_threaded[Threads.threadid()]
    fstar_lower_right = fstar_lower_right_threaded[Threads.threadid()]
    fstar_tmp1        = fstar_tmp1_threaded[Threads.threadid()]

    noncons_diamond_upper_left  = noncons_diamond_upper_left_threaded[Threads.threadid()]
    noncons_diamond_upper_right = noncons_diamond_upper_right_threaded[Threads.threadid()]
    noncons_diamond_lower_left  = noncons_diamond_lower_left_threaded[Threads.threadid()]
    noncons_diamond_lower_right = noncons_diamond_lower_right_threaded[Threads.threadid()]

    # Calculate fluxes
    orientation = orientations[mortar]
    calc_fstar!(fstar_upper_left,  equations, dg, u_upper_left,  mortar, orientation)
    calc_fstar!(fstar_upper_right, equations, dg, u_upper_right, mortar, orientation)
    calc_fstar!(fstar_lower_left,  equations, dg, u_lower_left,  mortar, orientation)
    calc_fstar!(fstar_lower_right, equations, dg, u_lower_right, mortar, orientation)

    # Compute the nonconservative numerical terms along the upper/lower interfaces
    # Done twice because left/right orientation matters
    # 1 -> primary element and 2 -> secondary element
    # See Bohm et al. 2018 for details on the nonconservative diamond "flux"
    if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
      for j in eachnode(dg), i in eachnode(dg)
        # pull the left/right solutions of the four faces
        u_upper_left_ll,  u_upper_left_rr  = get_surface_node_vars(u_upper_left,  equations, dg, i, j, mortar)
        u_upper_right_ll, u_upper_right_rr = get_surface_node_vars(u_upper_right, equations, dg, i, j, mortar)
        u_lower_left_ll,  u_lower_left_rr  = get_surface_node_vars(u_lower_left,  equations, dg, i, j, mortar)
        u_lower_right_ll, u_lower_right_rr = get_surface_node_vars(u_lower_right, equations, dg, i, j, mortar)
        # compute pointwise nonconservative terms
        noncons_upper_left  = noncons_interface_flux(u_upper_left_ll,  u_upper_left_rr,
                                                      orientations[mortar], equations)
        noncons_upper_right = noncons_interface_flux(u_upper_right_ll, u_upper_right_rr,
                                                      orientations[mortar], equations)
        noncons_lower_left  = noncons_interface_flux(u_lower_left_ll,  u_lower_left_rr,
                                                      orientations[mortar], equations)
        noncons_lower_right = noncons_interface_flux(u_lower_right_ll, u_lower_right_rr,
                                                      orientations[mortar], equations)
        # Save into temporay storage
        set_node_vars!(noncons_diamond_upper_left,  noncons_upper_left,  equations, dg, i, j)
        set_node_vars!(noncons_diamond_upper_right, noncons_upper_right, equations, dg, i, j)
        set_node_vars!(noncons_diamond_lower_left,  noncons_lower_left,  equations, dg, i, j)
        set_node_vars!(noncons_diamond_lower_right, noncons_lower_right, equations, dg, i, j)
      end
    else # large_sides[mortar] == 2 -> small elements on the left
      for j in eachnode(dg), i in eachnode(dg)
        # pull the left/right solutions of the four faces
        u_upper_left_ll,  u_upper_left_rr  = get_surface_node_vars(u_upper_left,  equations, dg, i, j, mortar)
        u_upper_right_ll, u_upper_right_rr = get_surface_node_vars(u_upper_right, equations, dg, i, j, mortar)
        u_lower_left_ll,  u_lower_left_rr  = get_surface_node_vars(u_lower_left,  equations, dg, i, j, mortar)
        u_lower_right_ll, u_lower_right_rr = get_surface_node_vars(u_lower_right, equations, dg, i, j, mortar)
        # compute pointwise nonconservative terms
        noncons_upper_left  = noncons_interface_flux(u_upper_left_rr,  u_upper_left_ll,
                                                      orientations[mortar], equations)
        noncons_upper_right = noncons_interface_flux(u_upper_right_rr, u_upper_right_ll,
                                                      orientations[mortar], equations)
        noncons_lower_left  = noncons_interface_flux(u_lower_left_rr,  u_lower_left_ll,
                                                      orientations[mortar], equations)
        noncons_lower_right = noncons_interface_flux(u_lower_right_rr, u_lower_right_ll,
                                                      orientations[mortar], equations)
        # Save into temporay storage
        set_node_vars!(noncons_diamond_upper_left,  noncons_upper_left,  equations, dg, i, j)
        set_node_vars!(noncons_diamond_upper_right, noncons_upper_right, equations, dg, i, j)
        set_node_vars!(noncons_diamond_lower_left,  noncons_lower_left,  equations, dg, i, j)
        set_node_vars!(noncons_diamond_lower_right, noncons_lower_right, equations, dg, i, j)
      end
    end

    # the code below is equivalent to
    # @. fstar_upper_left  += noncons_diamond_upper_left
    # @. fstar_upper_right += noncons_diamond_upper_right
    # @. fstar_lower_left  += noncons_diamond_lower_left
    # @. fstar_lower_right += noncons_diamond_lower_right
    # but slightly faster
    for j in eachnode(dg), i in eachnode(dg), v in eachvariable(equations)
      fstar_upper_left[v, i, j]  += noncons_diamond_upper_left[v, i, j]
    end
    for j in eachnode(dg), i in eachnode(dg), v in eachvariable(equations)
      fstar_upper_right[v, i, j] += noncons_diamond_upper_right[v, i, j]
    end
    for j in eachnode(dg), i in eachnode(dg), v in eachvariable(equations)
      fstar_lower_left[v, i, j]  += noncons_diamond_lower_left[v, i, j]
    end
    for j in eachnode(dg), i in eachnode(dg), v in eachvariable(equations)
      fstar_lower_right[v, i, j] += noncons_diamond_lower_right[v, i, j]
    end
    mortar_fluxes_to_elements!(surface_flux_values, equations, mortar_l2, dg, cache, mortar,
                               fstar_upper_left, fstar_upper_right,
                               fstar_lower_left, fstar_lower_right,
                               fstar_tmp1)
  end

  return nothing
end

@inline function calc_fstar!(destination::AbstractArray{<:Any,3}, equations, dg::DGSEM,
                             u_interfaces, interface, orientation)
  @unpack surface_flux = dg

  for j in eachnode(dg), i in eachnode(dg)
    # Call pointwise two-point numerical flux function
    u_ll, u_rr = get_surface_node_vars(u_interfaces, equations, dg, i, j, interface)
    flux = surface_flux(u_ll, u_rr, orientation, equations)

    # Copy flux to left and right element storage
    set_node_vars!(destination, flux, equations, dg, i, j)
  end

  return nothing
end

@inline function mortar_fluxes_to_elements!(surface_flux_values::AbstractArray{<:Any,5}, equations,
                                            mortar_l2::LobattoLegendreMortarL2, dg::DGSEM, cache, mortar,
                                            fstar_upper_left, fstar_upper_right,
                                            fstar_lower_left, fstar_lower_right,
                                            fstar_tmp1)
  lower_left_element  = cache.mortars.neighbor_ids[1, mortar]
  lower_right_element = cache.mortars.neighbor_ids[2, mortar]
  upper_left_element  = cache.mortars.neighbor_ids[3, mortar]
  upper_right_element = cache.mortars.neighbor_ids[4, mortar]
  large_element       = cache.mortars.neighbor_ids[5, mortar]

  # Copy flux small to small
  if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
    if cache.mortars.orientations[mortar] == 1
      # L2 mortars in x-direction
      direction = 1
    elseif cache.mortars.orientations[mortar] == 2
      # L2 mortars in y-direction
      direction = 3
    else # if cache.mortars.orientations[mortar] == 3
      # L2 mortars in z-direction
      direction = 5
    end
  else # large_sides[mortar] == 2 -> small elements on left side
    if cache.mortars.orientations[mortar] == 1
      # L2 mortars in x-direction
      direction = 2
    elseif cache.mortars.orientations[mortar] == 2
      # L2 mortars in y-direction
      direction = 4
    else # if cache.mortars.orientations[mortar] == 3
      # L2 mortars in z-direction
      direction = 6
    end
  end
  surface_flux_values[:, :, :, direction, upper_left_element]  .= fstar_upper_left
  surface_flux_values[:, :, :, direction, upper_right_element] .= fstar_upper_right
  surface_flux_values[:, :, :, direction, lower_left_element]  .= fstar_lower_left
  surface_flux_values[:, :, :, direction, lower_right_element] .= fstar_lower_right

  # Project small fluxes to large element
  if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
    if cache.mortars.orientations[mortar] == 1
      # L2 mortars in x-direction
      direction = 2
    elseif cache.mortars.orientations[mortar] == 2
      # L2 mortars in y-direction
      direction = 4
    else # if cache.mortars.orientations[mortar] == 3
      # L2 mortars in z-direction
      direction = 6
    end
  else # large_sides[mortar] == 2 -> small elements on left side
    if cache.mortars.orientations[mortar] == 1
      # L2 mortars in x-direction
      direction = 1
    elseif cache.mortars.orientations[mortar] == 2
      # L2 mortars in y-direction
      direction = 3
    else # if cache.mortars.orientations[mortar] == 3
      # L2 mortars in z-direction
      direction = 5
    end
  end

  multiply_dimensionwise!(
    view(surface_flux_values, :, :, :, direction, large_element),
    mortar_l2.reverse_lower, mortar_l2.reverse_upper, fstar_upper_left, fstar_tmp1)
  add_multiply_dimensionwise!(
    view(surface_flux_values, :, :, :, direction, large_element),
    mortar_l2.reverse_upper, mortar_l2.reverse_upper, fstar_upper_right, fstar_tmp1)
  add_multiply_dimensionwise!(
    view(surface_flux_values, :, :, :, direction, large_element),
    mortar_l2.reverse_lower, mortar_l2.reverse_lower, fstar_lower_left, fstar_tmp1)
  add_multiply_dimensionwise!(
    view(surface_flux_values, :, :, :, direction, large_element),
    mortar_l2.reverse_upper, mortar_l2.reverse_lower, fstar_lower_right, fstar_tmp1)

  return nothing
end


function calc_surface_integral!(du::AbstractArray{<:Any,5}, equations, dg::DGSEM, cache)
  @unpack boundary_interpolation = dg.basis
  @unpack surface_flux_values = cache.elements

  @threaded for element in eachelement(dg, cache)
    for m in eachnode(dg), l in eachnode(dg)
      for v in eachvariable(equations)
        # surface at -x
        du[v, 1,          l, m, element] -= surface_flux_values[v, l, m, 1, element] * boundary_interpolation[1,          1]
        # surface at +x
        du[v, nnodes(dg), l, m, element] += surface_flux_values[v, l, m, 2, element] * boundary_interpolation[nnodes(dg), 2]
        # surface at -y
        du[v, l, 1,          m, element] -= surface_flux_values[v, l, m, 3, element] * boundary_interpolation[1,          1]
        # surface at +y
        du[v, l, nnodes(dg), m, element] += surface_flux_values[v, l, m, 4, element] * boundary_interpolation[nnodes(dg), 2]
        # surface at -z
        du[v, l, m, 1,          element] -= surface_flux_values[v, l, m, 5, element] * boundary_interpolation[1,          1]
        # surface at +z
        du[v, l, m, nnodes(dg), element] += surface_flux_values[v, l, m, 6, element] * boundary_interpolation[nnodes(dg), 2]
      end
    end
  end

  return nothing
end


function apply_jacobian!(du::AbstractArray{<:Any,5}, equations, dg::DG, cache)

  @threaded for element in eachelement(dg, cache)
    factor = -cache.elements.inverse_jacobian[element]

    @avx for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      for v in eachvariable(equations)
        du[v, i, j, k, element] *= factor
      end
    end
  end

  return nothing
end


# TODO: Taal dimension agnostic
function calc_sources!(du::AbstractArray{<:Any,5}, u, t, source_terms::Nothing, equations, dg::DG, cache)
  return nothing
end

function calc_sources!(du::AbstractArray{<:Any,5}, u, t, source_terms, equations, dg::DG, cache)

  @threaded for element in eachelement(dg, cache)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, j, k, element)
      x_local = get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, k, element)
      du_local = source_terms(u_local, x_local, t, equations)
      add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
    end
  end

  return nothing
end
