# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


# everything related to a DG semidiscretization in 2D,
# currently limited to Lobatto-Legendre nodes

# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::TreeMesh{2}, equations,
                      dg::DG, RealT, uEltype)
  # Get cells for which an element needs to be created (i.e. all leaf cells)
  leaf_cell_ids = local_leaf_cells(mesh.tree)

  elements = init_elements(leaf_cell_ids, mesh, equations, dg.basis, RealT, uEltype)

  interfaces = init_interfaces(leaf_cell_ids, mesh, elements)

  boundaries = init_boundaries(leaf_cell_ids, mesh, elements)

  mortars = init_mortars(leaf_cell_ids, mesh, elements, dg.mortar)

  cache = (; elements, interfaces, boundaries, mortars)

  # Add specialized parts of the cache required to compute the volume integral etc.
  cache = (;cache..., create_cache(mesh, equations, dg.volume_integral, dg, uEltype)...)
  cache = (;cache..., create_cache(mesh, equations, dg.mortar, uEltype)...)

  return cache
end


# The methods below are specialized on the volume integral type
# and called from the basic `create_cache` method at the top.
function create_cache(mesh::Union{TreeMesh{2}, StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                      equations, volume_integral::VolumeIntegralFluxDifferencing, dg::DG, uEltype)
  NamedTuple()
end


function create_cache(mesh::Union{TreeMesh{2}, StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}}, equations,
                      volume_integral::VolumeIntegralShockCapturingHG, dg::DG, uEltype)
  element_ids_dg   = Int[]
  element_ids_dgfv = Int[]

  cache = create_cache(mesh, equations,
                       VolumeIntegralFluxDifferencing(volume_integral.volume_flux_dg),
                       dg, uEltype)

  A3dp1_x = Array{uEltype, 3}
  A3dp1_y = Array{uEltype, 3}

  fstar1_L_threaded = A3dp1_x[A3dp1_x(undef, nvariables(equations), nnodes(dg)+1, nnodes(dg)) for _ in 1:Threads.nthreads()]
  fstar1_R_threaded = A3dp1_x[A3dp1_x(undef, nvariables(equations), nnodes(dg)+1, nnodes(dg)) for _ in 1:Threads.nthreads()]
  fstar2_L_threaded = A3dp1_y[A3dp1_y(undef, nvariables(equations), nnodes(dg), nnodes(dg)+1) for _ in 1:Threads.nthreads()]
  fstar2_R_threaded = A3dp1_y[A3dp1_y(undef, nvariables(equations), nnodes(dg), nnodes(dg)+1) for _ in 1:Threads.nthreads()]

  return (; cache..., element_ids_dg, element_ids_dgfv,
          fstar1_L_threaded, fstar1_R_threaded, fstar2_L_threaded, fstar2_R_threaded)
end


function create_cache(mesh::Union{TreeMesh{2}, StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}}, equations,
                      volume_integral::VolumeIntegralPureLGLFiniteVolume, dg::DG, uEltype)

  A3dp1_x = Array{uEltype, 3}
  A3dp1_y = Array{uEltype, 3}

  fstar1_L_threaded = A3dp1_x[A3dp1_x(undef, nvariables(equations), nnodes(dg)+1, nnodes(dg)) for _ in 1:Threads.nthreads()]
  fstar1_R_threaded = A3dp1_x[A3dp1_x(undef, nvariables(equations), nnodes(dg)+1, nnodes(dg)) for _ in 1:Threads.nthreads()]
  fstar2_L_threaded = A3dp1_y[A3dp1_y(undef, nvariables(equations), nnodes(dg), nnodes(dg)+1) for _ in 1:Threads.nthreads()]
  fstar2_R_threaded = A3dp1_y[A3dp1_y(undef, nvariables(equations), nnodes(dg), nnodes(dg)+1) for _ in 1:Threads.nthreads()]

  return (; fstar1_L_threaded, fstar1_R_threaded, fstar2_L_threaded, fstar2_R_threaded)
end


function create_cache(mesh::TreeMesh{2}, equations,
                      volume_integral::VolumeIntegralShockCapturingSubcell, dg::DG, uEltype)

  cache1 = create_cache(mesh, equations,
                       VolumeIntegralPureLGLFiniteVolume(volume_integral.volume_flux_fv),
                       dg, uEltype)

  A3dp1_x = Array{uEltype, 3}
  A3dp1_y = Array{uEltype, 3}
  A3d = Array{uEltype, 3}

  fhat1_threaded = A3dp1_x[A3dp1_x(undef, nvariables(equations), nnodes(dg)+1, nnodes(dg)) for _ in 1:Threads.nthreads()]
  fhat2_threaded = A3dp1_y[A3dp1_y(undef, nvariables(equations), nnodes(dg), nnodes(dg)+1) for _ in 1:Threads.nthreads()]
  flux_temp_threaded = A3d[A3d(undef, nvariables(equations), nnodes(dg), nnodes(dg)) for _ in 1:Threads.nthreads()]

  cache2 = create_cache(mesh, equations, volume_integral.indicator, dg, uEltype)

  return (; cache1..., cache2..., fhat1_threaded, fhat2_threaded, flux_temp_threaded)
end

function create_cache(mesh::TreeMesh{2}, equations, indicator::IndicatorIDP, dg::DG, uEltype)
  ContainerFCT2D = Trixi.ContainerFCT2D{uEltype}(0, nvariables(equations), nnodes(dg))

  return (; ContainerFCT2D)
end

function create_cache(mesh::TreeMesh{2}, equations, indicator::IndicatorMCL, dg::DG, uEltype)
  ContainerMCL2D = Trixi.ContainerMCL2D{uEltype}(0, nvariables(equations), nnodes(dg))

  A3dp1_x = Array{uEltype, 3}
  A3dp1_y = Array{uEltype, 3}

  antidiffusive_flux1_threaded = A3dp1_x[A3dp1_x(undef, nvariables(equations), nnodes(dg)+1, nnodes(dg)) for _ in 1:Threads.nthreads()]
  antidiffusive_flux2_threaded = A3dp1_y[A3dp1_y(undef, nvariables(equations), nnodes(dg), nnodes(dg)+1) for _ in 1:Threads.nthreads()]

  return (; ContainerMCL2D,
          antidiffusive_flux1_threaded, antidiffusive_flux2_threaded)
end


# The methods below are specialized on the mortar type
# and called from the basic `create_cache` method at the top.
function create_cache(mesh::Union{TreeMesh{2}, StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                      equations, mortar_l2::LobattoLegendreMortarL2, uEltype)
  # TODO: Taal performance using different types
  MA2d = MArray{Tuple{nvariables(equations), nnodes(mortar_l2)}, uEltype, 2, nvariables(equations) * nnodes(mortar_l2)}
  fstar_upper_threaded = MA2d[MA2d(undef) for _ in 1:Threads.nthreads()]
  fstar_lower_threaded = MA2d[MA2d(undef) for _ in 1:Threads.nthreads()]

  # A2d = Array{uEltype, 2}
  # fstar_upper_threaded = [A2d(undef, nvariables(equations), nnodes(mortar_l2)) for _ in 1:Threads.nthreads()]
  # fstar_lower_threaded = [A2d(undef, nvariables(equations), nnodes(mortar_l2)) for _ in 1:Threads.nthreads()]

  (; fstar_upper_threaded, fstar_lower_threaded)
end


# TODO: Taal discuss/refactor timer, allowing users to pass a custom timer?

function rhs!(du, u, t,
              mesh::Union{TreeMesh{2}, P4estMesh{2}}, equations,
              initial_condition, boundary_conditions, source_terms,
              dg::DG, cache)
  # Reset du
  @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

  # Calculate volume integral
  @trixi_timeit timer() "volume integral" calc_volume_integral!(
    du, u, mesh,
    have_nonconservative_terms(equations), equations,
    dg.volume_integral, dg, cache)

  # Prolong solution to interfaces
  @trixi_timeit timer() "prolong2interfaces" prolong2interfaces!(
    cache, u, mesh, equations, dg.surface_integral, dg)

  # Calculate interface fluxes
  @trixi_timeit timer() "interface flux" calc_interface_flux!(
    cache.elements.surface_flux_values, mesh,
    have_nonconservative_terms(equations), equations,
    dg.surface_integral, dg, cache)

  # Prolong solution to boundaries
  @trixi_timeit timer() "prolong2boundaries" prolong2boundaries!(
    cache, u, mesh, equations, dg.surface_integral, dg)

  # Calculate boundary fluxes
  @trixi_timeit timer() "boundary flux" calc_boundary_flux!(
    cache, t, boundary_conditions, mesh, equations, dg.surface_integral, dg)

  # Prolong solution to mortars
  @trixi_timeit timer() "prolong2mortars" prolong2mortars!(
    cache, u, mesh, equations, dg.mortar, dg.surface_integral, dg)

  # Calculate mortar fluxes
  @trixi_timeit timer() "mortar flux" calc_mortar_flux!(
    cache.elements.surface_flux_values, mesh,
    have_nonconservative_terms(equations), equations,
    dg.mortar, dg.surface_integral, dg, cache)

  # Calculate surface integrals
  @trixi_timeit timer() "surface integral" calc_surface_integral!(
    du, u, mesh, equations, dg.surface_integral, dg, cache)

  # Apply Jacobian from mapping to reference element
  @trixi_timeit timer() "Jacobian" apply_jacobian!(
    du, mesh, equations, dg, cache)

  # Calculate source terms
  @trixi_timeit timer() "source terms" calc_sources!(
    du, u, t, source_terms, equations, dg, cache)

  return nothing
end


function calc_volume_integral!(du, u,
                               mesh::Union{TreeMesh{2}, StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralWeakForm,
                               dg::DGSEM, cache)

  @threaded for element in eachelement(dg, cache)
    weak_form_kernel!(du, u, element, mesh,
                      nonconservative_terms, equations,
                      dg, cache)
  end

  return nothing
end

@inline function weak_form_kernel!(du, u,
                                   element, mesh::TreeMesh{2},
                                   nonconservative_terms::Val{false}, equations,
                                   dg::DGSEM, cache, alpha=true)
  # true * [some floating point value] == [exactly the same floating point value]
  # This can (hopefully) be optimized away due to constant propagation.
  @unpack derivative_dhat = dg.basis

  # Calculate volume terms in one element
  for j in eachnode(dg), i in eachnode(dg)
    u_node = get_node_vars(u, equations, dg, i, j, element)

    flux1 = flux(u_node, 1, equations)
    for ii in eachnode(dg)
      multiply_add_to_node_vars!(du, alpha * derivative_dhat[ii, i], flux1, equations, dg, ii, j, element)
    end

    flux2 = flux(u_node, 2, equations)
    for jj in eachnode(dg)
      multiply_add_to_node_vars!(du, alpha * derivative_dhat[jj, j], flux2, equations, dg, i, jj, element)
    end
  end

  return nothing
end


# flux differencing volume integral. For curved meshes averaging of the
# mapping terms, stored in `cache.elements.contravariant_vectors`, is peeled apart
# from the evaluation of the physical fluxes in each Cartesian direction
function calc_volume_integral!(du, u,
                               mesh::Union{TreeMesh{2}, StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralFluxDifferencing,
                               dg::DGSEM, cache)
  @threaded for element in eachelement(dg, cache)
    split_form_kernel!(du, u, element, mesh,
                       nonconservative_terms, equations,
                       volume_integral.volume_flux, dg, cache)
  end
end

@inline function split_form_kernel!(du, u,
                                    element, mesh::TreeMesh{2},
                                    nonconservative_terms::Val{false}, equations,
                                    volume_flux, dg::DGSEM, cache, alpha=true)
  # true * [some floating point value] == [exactly the same floating point value]
  # This can (hopefully) be optimized away due to constant propagation.
  @unpack derivative_split = dg.basis

  # Calculate volume integral in one element
  for j in eachnode(dg), i in eachnode(dg)
    u_node = get_node_vars(u, equations, dg, i, j, element)

    # All diagonal entries of `derivative_split` are zero. Thus, we can skip
    # the computation of the diagonal terms. In addition, we use the symmetry
    # of the `volume_flux` to save half of the possible two-poitn flux
    # computations.

    # x direction
    for ii in (i+1):nnodes(dg)
      u_node_ii = get_node_vars(u, equations, dg, ii, j, element)
      flux1 = volume_flux(u_node, u_node_ii, 1, equations)
      multiply_add_to_node_vars!(du, alpha * derivative_split[i, ii], flux1, equations, dg, i,  j, element)
      multiply_add_to_node_vars!(du, alpha * derivative_split[ii, i], flux1, equations, dg, ii, j, element)
    end

    # y direction
    for jj in (j+1):nnodes(dg)
      u_node_jj = get_node_vars(u, equations, dg, i, jj, element)
      flux2 = volume_flux(u_node, u_node_jj, 2, equations)
      multiply_add_to_node_vars!(du, alpha * derivative_split[j, jj], flux2, equations, dg, i, j,  element)
      multiply_add_to_node_vars!(du, alpha * derivative_split[jj, j], flux2, equations, dg, i, jj, element)
    end
  end
end

@inline function split_form_kernel!(du, u,
                                    element, mesh::TreeMesh{2},
                                    nonconservative_terms::Val{true}, equations,
                                    volume_flux, dg::DGSEM, cache, alpha=true)
  # true * [some floating point value] == [exactly the same floating point value]
  # This can (hopefully) be optimized away due to constant propagation.
  @unpack derivative_split = dg.basis
  symmetric_flux, nonconservative_flux = volume_flux

  # Apply the symmetric flux as usual
  split_form_kernel!(du, u, element, mesh, Val(false), equations, symmetric_flux, dg, cache, alpha)

  # Calculate the remaining volume terms using the nonsymmetric generalized flux
  for j in eachnode(dg), i in eachnode(dg)
    u_node = get_node_vars(u, equations, dg, i, j, element)

    # The diagonal terms are zero since the diagonal of `derivative_split`
    # is zero. We ignore this for now.

    # x direction
    integral_contribution = zero(u_node)
    for ii in eachnode(dg)
      u_node_ii = get_node_vars(u, equations, dg, ii, j, element)
      noncons_flux1 = nonconservative_flux(u_node, u_node_ii, 1, equations)
      integral_contribution = integral_contribution + derivative_split[i, ii] * noncons_flux1
    end

    # y direction
    for jj in eachnode(dg)
      u_node_jj = get_node_vars(u, equations, dg, i, jj, element)
      noncons_flux2 = nonconservative_flux(u_node, u_node_jj, 2, equations)
      integral_contribution = integral_contribution + derivative_split[j, jj] * noncons_flux2
    end

    # The factor 0.5 cancels the factor 2 in the flux differencing form
    multiply_add_to_node_vars!(du, alpha * 0.5, integral_contribution, equations, dg, i, j, element)
  end
end


# TODO: Taal dimension agnostic
function calc_volume_integral!(du, u,
                               mesh::Union{TreeMesh{2}, StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralShockCapturingHG,
                               dg::DGSEM, cache)
  @unpack element_ids_dg, element_ids_dgfv = cache
  @unpack volume_flux_dg, volume_flux_fv, indicator = volume_integral

  # Calculate blending factors α: u = u_DG * (1 - α) + u_FV * α
  alpha = @trixi_timeit timer() "blending factors" indicator(u, mesh, equations, dg, cache)

  # Determine element ids for DG-only and blended DG-FV volume integral
  pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, dg, cache)

  # Loop over pure DG elements
  @trixi_timeit timer() "pure DG" @threaded for idx_element in eachindex(element_ids_dg)
    element = element_ids_dg[idx_element]
    split_form_kernel!(du, u, element, mesh,
                       nonconservative_terms, equations,
                       volume_flux_dg, dg, cache)
  end

  # Loop over blended DG-FV elements
  @trixi_timeit timer() "blended DG-FV" @threaded for idx_element in eachindex(element_ids_dgfv)
    element = element_ids_dgfv[idx_element]
    alpha_element = alpha[element]

    # Calculate DG volume integral contribution
    split_form_kernel!(du, u, element, mesh,
                       nonconservative_terms, equations,
                       volume_flux_dg, dg, cache, 1 - alpha_element)

    # Calculate FV volume integral contribution
    fv_kernel!(du, u, mesh, nonconservative_terms, equations, volume_flux_fv,
               dg, cache, element, alpha_element)
  end

  return nothing
end

# TODO: Taal dimension agnostic
function calc_volume_integral!(du, u,
                               mesh::TreeMesh{2},
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralPureLGLFiniteVolume,
                               dg::DGSEM, cache)
  @unpack volume_flux_fv = volume_integral

  # Calculate LGL FV volume integral
  @threaded for element in eachelement(dg, cache)
    fv_kernel!(du, u, mesh, nonconservative_terms, equations, volume_flux_fv,
               dg, cache, element, true)
  end

  return nothing
end


@inline function fv_kernel!(du, u,
                            mesh::Union{TreeMesh{2}, StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                            nonconservative_terms, equations,
                            volume_flux_fv, dg::DGSEM, cache, element, alpha=true)
  @unpack fstar1_L_threaded, fstar1_R_threaded, fstar2_L_threaded, fstar2_R_threaded = cache
  @unpack inverse_weights = dg.basis

  # Calculate FV two-point fluxes
  fstar1_L = fstar1_L_threaded[Threads.threadid()]
  fstar2_L = fstar2_L_threaded[Threads.threadid()]
  fstar1_R = fstar1_R_threaded[Threads.threadid()]
  fstar2_R = fstar2_R_threaded[Threads.threadid()]
  calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u, mesh,
               nonconservative_terms, equations, volume_flux_fv, dg, element, cache)

  # Calculate FV volume integral contribution
  for j in eachnode(dg), i in eachnode(dg)
    for v in eachvariable(equations)
      du[v, i, j, element] += ( alpha *
                                (inverse_weights[i] * (fstar1_L[v, i+1, j] - fstar1_R[v, i, j]) +
                                 inverse_weights[j] * (fstar2_L[v, i, j+1] - fstar2_R[v, i, j])) )
    end
  end

  return nothing
end



#     calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u_leftright,
#                  nonconservative_terms::Val{false}, equations,
#                  volume_flux_fv, dg, element)
#
# Calculate the finite volume fluxes inside the elements (**without non-conservative terms**).
#
# # Arguments
# - `fstar1_L::AbstractArray{<:Real, 3}`
# - `fstar1_R::AbstractArray{<:Real, 3}`
# - `fstar2_L::AbstractArray{<:Real, 3}`
# - `fstar2_R::AbstractArray{<:Real, 3}`
@inline function calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u::AbstractArray{<:Any,4},
                              mesh::TreeMesh{2}, nonconservative_terms::Val{false}, equations,
                              volume_flux_fv, dg::DGSEM, element, cache)

  fstar1_L[:, 1,            :] .= zero(eltype(fstar1_L))
  fstar1_L[:, nnodes(dg)+1, :] .= zero(eltype(fstar1_L))
  fstar1_R[:, 1,            :] .= zero(eltype(fstar1_R))
  fstar1_R[:, nnodes(dg)+1, :] .= zero(eltype(fstar1_R))

  for j in eachnode(dg), i in 2:nnodes(dg)
    u_ll = get_node_vars(u, equations, dg, i-1, j, element)
    u_rr = get_node_vars(u, equations, dg, i,   j, element)
    flux = volume_flux_fv(u_ll, u_rr, 1, equations) # orientation 1: x direction
    set_node_vars!(fstar1_L, flux, equations, dg, i, j)
    set_node_vars!(fstar1_R, flux, equations, dg, i, j)
  end

  fstar2_L[:, :, 1           ] .= zero(eltype(fstar2_L))
  fstar2_L[:, :, nnodes(dg)+1] .= zero(eltype(fstar2_L))
  fstar2_R[:, :, 1           ] .= zero(eltype(fstar2_R))
  fstar2_R[:, :, nnodes(dg)+1] .= zero(eltype(fstar2_R))

  for j in 2:nnodes(dg), i in eachnode(dg)
    u_ll = get_node_vars(u, equations, dg, i, j-1, element)
    u_rr = get_node_vars(u, equations, dg, i, j,   element)
    flux = volume_flux_fv(u_ll, u_rr, 2, equations) # orientation 2: y direction
    set_node_vars!(fstar2_L, flux, equations, dg, i, j)
    set_node_vars!(fstar2_R, flux, equations, dg, i, j)
  end

  return nothing
end

#     calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u_leftright,
#                  nonconservative_terms::Val{true}, equations,
#                  volume_flux_fv, dg, element)
#
# Calculate the finite volume fluxes inside the elements (**with non-conservative terms**).
#
# # Arguments
# - `fstar1_L::AbstractArray{<:Real, 3}`:
# - `fstar1_R::AbstractArray{<:Real, 3}`:
# - `fstar2_L::AbstractArray{<:Real, 3}`:
# - `fstar2_R::AbstractArray{<:Real, 3}`:
# - `u_leftright::AbstractArray{<:Real, 4}`
@inline function calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u::AbstractArray{<:Any,4},
                              mesh::TreeMesh{2}, nonconservative_terms::Val{true}, equations,
                              volume_flux_fv, dg::DGSEM, element, cache)
  volume_flux, nonconservative_flux = volume_flux_fv

  # Fluxes in x
  fstar1_L[:, 1,            :] .= zero(eltype(fstar1_L))
  fstar1_L[:, nnodes(dg)+1, :] .= zero(eltype(fstar1_L))
  fstar1_R[:, 1,            :] .= zero(eltype(fstar1_R))
  fstar1_R[:, nnodes(dg)+1, :] .= zero(eltype(fstar1_R))

  for j in eachnode(dg), i in 2:nnodes(dg)
    u_ll = get_node_vars(u, equations, dg, i-1, j, element)
    u_rr = get_node_vars(u, equations, dg, i,   j, element)

    # Compute conservative part
    f1 = volume_flux(u_ll, u_rr, 1, equations) # orientation 1: x direction

    # Compute nonconservative part
    # Note the factor 0.5 necessary for the nonconservative fluxes based on
    # the interpretation of global SBP operators coupled discontinuously via
    # central fluxes/SATs
    f1_L = f1 + 0.5 * nonconservative_flux(u_ll, u_rr, 1, equations)
    f1_R = f1 + 0.5 * nonconservative_flux(u_rr, u_ll, 1, equations)

    # Copy to temporary storage
    set_node_vars!(fstar1_L, f1_L, equations, dg, i, j)
    set_node_vars!(fstar1_R, f1_R, equations, dg, i, j)
  end

  # Fluxes in y
  fstar2_L[:, :, 1           ] .= zero(eltype(fstar2_L))
  fstar2_L[:, :, nnodes(dg)+1] .= zero(eltype(fstar2_L))
  fstar2_R[:, :, 1           ] .= zero(eltype(fstar2_R))
  fstar2_R[:, :, nnodes(dg)+1] .= zero(eltype(fstar2_R))

  # Compute inner fluxes
  for j in 2:nnodes(dg), i in eachnode(dg)
    u_ll = get_node_vars(u, equations, dg, i, j-1, element)
    u_rr = get_node_vars(u, equations, dg, i, j,   element)

    # Compute conservative part
    f2 = volume_flux(u_ll, u_rr, 2, equations) # orientation 2: y direction

    # Compute nonconservative part
    # Note the factor 0.5 necessary for the nonconservative fluxes based on
    # the interpretation of global SBP operators coupled discontinuously via
    # central fluxes/SATs
    f2_L = f2 + 0.5 * nonconservative_flux(u_ll, u_rr, 2, equations)
    f2_R = f2 + 0.5 * nonconservative_flux(u_rr, u_ll, 2, equations)

    # Copy to temporary storage
    set_node_vars!(fstar2_L, f2_L, equations, dg, i, j)
    set_node_vars!(fstar2_R, f2_R, equations, dg, i, j)
  end

  return nothing
end


function calc_volume_integral!(du, u,
                               mesh::TreeMesh{2},
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralShockCapturingSubcell,
                               dg::DGSEM, cache)
  # Calculate maximum wave speeds lambda
  # TODO:
  # Option one: (Right now) Calculate the lambdas 4 times each time step (before each RK stage and in callback) plus one time to init the callback
  #   1 In the stepsize callback to get the right time step
  #   Remove 1, the first time step cannot be calculated and the others are not accurate (with old lambdas)
  #   2 In the volume integral (here).
  #   Remove 2, the first entropy analysis of the analysis_callback doesn't work.
  #             And we get different result because otherwise the lambdas are only updated once in a RK step.
  #   -> 4 times per timestep is actually not that bad. (3 times would be optimal)
  # Option two: Calculate lambdas after each RK stage plus in the init_stepsize_callback.
  #   Problem: Entropy change at t=0 only works if the stepsize callback is listed before analysis callback (to calculate the lambdas before)
  @trixi_timeit timer() "calc_lambda!" calc_lambda!(u, mesh, equations, dg, cache, volume_integral.indicator)
  # Calculate bar states
  @trixi_timeit timer() "calc_bar_states!" calc_bar_states!(u, mesh, nonconservative_terms, equations, volume_integral.indicator, dg, cache)
  # Calculate boundaries
  @trixi_timeit timer() "calc_var_bounds!" calc_var_bounds!(u, mesh, nonconservative_terms, equations, volume_integral.indicator, dg, cache)

  @trixi_timeit timer() "subcell_limiting_kernel!" @threaded for element in eachelement(dg, cache)
    subcell_limiting_kernel!(du, u, element, mesh,
                             nonconservative_terms, equations,
                             volume_integral, volume_integral.indicator,
                             dg, cache)
  end
end

@inline function subcell_limiting_kernel!(du, u,
                                          element, mesh::TreeMesh{2},
                                          nonconservative_terms::Val{false}, equations,
                                          volume_integral, indicator::IndicatorIDP,
                                          dg::DGSEM, cache)
  @unpack inverse_weights = dg.basis
  @unpack volume_flux_dg, volume_flux_fv = volume_integral

  # high-order DG fluxes
  @unpack fhat1_threaded, fhat2_threaded = cache

  fhat1 = fhat1_threaded[Threads.threadid()]
  fhat2 = fhat2_threaded[Threads.threadid()]
  calcflux_fhat!(fhat1, fhat2, u, mesh,
      nonconservative_terms, equations, volume_flux_dg, dg, element, cache)

  # low-order FV fluxes
  @unpack fstar1_L_threaded, fstar1_R_threaded, fstar2_L_threaded, fstar2_R_threaded = cache

  fstar1_L = fstar1_L_threaded[Threads.threadid()]
  fstar2_L = fstar2_L_threaded[Threads.threadid()]
  fstar1_R = fstar1_R_threaded[Threads.threadid()]
  fstar2_R = fstar2_R_threaded[Threads.threadid()]
  calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u, mesh,
      nonconservative_terms, equations, volume_flux_fv, dg, element, cache)

  # Calculate antidiffusive flux
  @unpack antidiffusive_flux1, antidiffusive_flux2 = cache.ContainerFCT2D

  calcflux_antidiffusive!(antidiffusive_flux1, antidiffusive_flux2, fhat1, fhat2, fstar1_L, fstar2_L, u, mesh,
      nonconservative_terms, equations, indicator, dg, element, cache)

  # Calculate volume integral contribution of low-order FV flux
  for j in eachnode(dg), i in eachnode(dg)
    for v in eachvariable(equations)
      du[v, i, j, element] += inverse_weights[i] * (fstar1_L[v, i+1, j] - fstar1_R[v, i, j]) +
                              inverse_weights[j] * (fstar2_L[v, i, j+1] - fstar2_R[v, i, j])

    end
  end

  return nothing
end

@inline function subcell_limiting_kernel!(du, u,
                                          element, mesh::TreeMesh{2},
                                          nonconservative_terms::Val{false}, equations,
                                          volume_integral, indicator::IndicatorMCL,
                                          dg::DGSEM, cache)
  @unpack inverse_weights = dg.basis
  @unpack volume_flux_dg, volume_flux_fv = volume_integral

  # high-order DG fluxes
  @unpack fhat1_threaded, fhat2_threaded = cache
  fhat1 = fhat1_threaded[Threads.threadid()]
  fhat2 = fhat2_threaded[Threads.threadid()]
  calcflux_fhat!(fhat1, fhat2, u, mesh,
      nonconservative_terms, equations, volume_flux_dg, dg, element, cache)

  # low-order FV fluxes
  @unpack fstar1_L_threaded, fstar1_R_threaded, fstar2_L_threaded, fstar2_R_threaded = cache
  fstar1_L = fstar1_L_threaded[Threads.threadid()]
  fstar2_L = fstar2_L_threaded[Threads.threadid()]
  fstar1_R = fstar1_R_threaded[Threads.threadid()]
  fstar2_R = fstar2_R_threaded[Threads.threadid()]
  calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u, mesh,
      nonconservative_terms, equations, volume_flux_fv, dg, element, cache)

  # antidiffusive flux
  @unpack antidiffusive_flux1_threaded, antidiffusive_flux2_threaded = cache
  antidiffusive_flux1 = antidiffusive_flux1_threaded[Threads.threadid()]
  antidiffusive_flux2 = antidiffusive_flux2_threaded[Threads.threadid()]
  calcflux_antidiffusive!(antidiffusive_flux1, antidiffusive_flux2, fhat1, fhat2, fstar1_L, fstar2_L,
      u, mesh, nonconservative_terms, equations, indicator, dg, element, cache)

  # limited antidiffusive flux
  calcflux_antidiffusive_limited!(antidiffusive_flux1, antidiffusive_flux2,
      u, mesh, nonconservative_terms, equations, indicator, dg, element, cache)

  @unpack antidiffusive_flux1_limited, antidiffusive_flux2_limited = cache.ContainerMCL2D
  for j in eachnode(dg), i in eachnode(dg)
    for v in eachvariable(equations)
      du[v, i, j, element] += inverse_weights[i] * (fstar1_L[v, i+1, j] - fstar1_R[v, i, j]) +
                              inverse_weights[j] * (fstar2_L[v, i, j+1] - fstar2_R[v, i, j])

      du[v, i, j, element] += inverse_weights[i] * (antidiffusive_flux1_limited[v, i+1, j, element] - antidiffusive_flux1_limited[v, i, j, element]) +
                              inverse_weights[j] * (antidiffusive_flux2_limited[v, i, j+1, element] - antidiffusive_flux2_limited[v, i, j, element])
    end
  end

  return nothing
end


#     calcflux_fhat!(fhat1, fhat2, u, mesh,
#                    nonconservative_terms, equations, volume_flux_dg, dg, element, cache)
#
# Calculate the DG staggered volume fluxes `fhat` in subcell FV-form inside the element
# (**without non-conservative terms**).
#
# # Arguments
# - `fhat1::AbstractArray{<:Real, 3}`
# - `fhat2::AbstractArray{<:Real, 3}`
@inline function calcflux_fhat!(fhat1, fhat2, u,
                                mesh::TreeMesh{2}, nonconservative_terms::Val{false}, equations,
                                volume_flux, dg::DGSEM, element, cache)

  @unpack weights, derivative_split = dg.basis
  @unpack flux_temp_threaded = cache

  flux_temp = flux_temp_threaded[Threads.threadid()]

  # The FV-form fluxes are calculated in a recursive manner, i.e.:
  # fhat_(0,1)   = w_0 * FVol_0,
  # fhat_(j,j+1) = fhat_(j-1,j) + w_j * FVol_j,   for j=1,...,N-1,
  # with the split form volume fluxes FVol_j = -2 * sum_i=0^N D_ji f*_(j,i).

  # To use the symmetry of the `volume_flux`, the split form volume flux is precalculated
  # like in `calc_volume_integral!` for the `VolumeIntegralFluxDifferencing`
  # and saved in in `flux_temp`.

  # Split form volume flux in orientation 1: x direction
  flux_temp .= zero(eltype(flux_temp))

  for j in eachnode(dg), i in eachnode(dg)
    u_node = get_node_vars(u, equations, dg, i, j, element)

    # All diagonal entries of `derivative_split` are zero. Thus, we can skip
    # the computation of the diagonal terms. In addition, we use the symmetry
    # of the `volume_flux` to save half of the possible two-poitn flux
    # computations.
    for ii in (i+1):nnodes(dg)
      u_node_ii = get_node_vars(u, equations, dg, ii, j, element)
      flux1 = volume_flux(u_node, u_node_ii, 1, equations)
      multiply_add_to_node_vars!(flux_temp, derivative_split[i, ii], flux1, equations, dg, i,  j)
      multiply_add_to_node_vars!(flux_temp, derivative_split[ii, i], flux1, equations, dg, ii, j)
    end
  end

  # FV-form flux `fhat` in x direction
  fhat1[:, 1,            :] .= zero(eltype(fhat1))
  fhat1[:, nnodes(dg)+1, :] .= zero(eltype(fhat1))

  for j in eachnode(dg), i in 1:nnodes(dg)-1, v in eachvariable(equations)
    fhat1[v, i+1, j] = fhat1[v, i, j] + weights[i] * flux_temp[v, i, j]
  end

  # Split form volume flux in orientation 2: y direction
  flux_temp .= zero(eltype(flux_temp))

  for j in eachnode(dg), i in eachnode(dg)
    u_node = get_node_vars(u, equations, dg, i, j, element)
    for jj in (j+1):nnodes(dg)
      u_node_jj = get_node_vars(u, equations, dg, i, jj, element)
      flux2 = volume_flux(u_node, u_node_jj, 2, equations)
      multiply_add_to_node_vars!(flux_temp, derivative_split[j, jj], flux2, equations, dg, i, j)
      multiply_add_to_node_vars!(flux_temp, derivative_split[jj, j], flux2, equations, dg, i, jj)
    end
  end

  # FV-form flux `fhat` in y direction
  fhat2[:, :, 1           ] .= zero(eltype(fhat2))
  fhat2[:, :, nnodes(dg)+1] .= zero(eltype(fhat2))

  for j in 1:nnodes(dg)-1, i in eachnode(dg), v in eachvariable(equations)
    fhat2[v, i, j+1] = fhat2[v, i, j] + weights[j] * flux_temp[v, i, j]
  end

  return nothing
end

@inline function calcflux_antidiffusive!(antidiffusive_flux1, antidiffusive_flux2, fhat1, fhat2, fstar1, fstar2, u, mesh,
                                         nonconservative_terms, equations, indicator::IndicatorIDP, dg, element, cache)

  for j in eachnode(dg), i in 2:nnodes(dg)
    for v in eachvariable(equations)
      antidiffusive_flux1[v, i, j, element] = fhat1[v, i, j] - fstar1[v, i, j]
    end
  end
  for j in 2:nnodes(dg), i in eachnode(dg)
    for v in eachvariable(equations)
      antidiffusive_flux2[v, i, j, element] = fhat2[v, i, j] - fstar2[v, i, j]
    end
  end

  antidiffusive_flux1[:, 1,            :, element] .= zero(eltype(antidiffusive_flux1))
  antidiffusive_flux1[:, nnodes(dg)+1, :, element] .= zero(eltype(antidiffusive_flux1))

  antidiffusive_flux2[:, :, 1,            element] .= zero(eltype(antidiffusive_flux2))
  antidiffusive_flux2[:, :, nnodes(dg)+1, element] .= zero(eltype(antidiffusive_flux2))

  return nothing
end

@inline function calcflux_antidiffusive!(antidiffusive_flux1, antidiffusive_flux2, fhat1, fhat2, fstar1, fstar2, u, mesh,
                                         nonconservative_terms, equations, indicator::IndicatorMCL, dg, element, cache)

  for j in eachnode(dg), i in 2:nnodes(dg)
    for v in eachvariable(equations)
      antidiffusive_flux1[v, i, j] = fhat1[v, i, j] - fstar1[v, i, j]
    end
  end
  for j in 2:nnodes(dg), i in eachnode(dg)
    for v in eachvariable(equations)
      antidiffusive_flux2[v, i, j] = fhat2[v, i, j] - fstar2[v, i, j]
    end
  end

  # antidiffusive_flux1[:, 1,            :] .= zero(eltype(antidiffusive_flux1))
  # antidiffusive_flux1[:, nnodes(dg)+1, :] .= zero(eltype(antidiffusive_flux1))

  # antidiffusive_flux2[:, :, 1,          ] .= zero(eltype(antidiffusive_flux2))
  # antidiffusive_flux2[:, :, nnodes(dg)+1] .= zero(eltype(antidiffusive_flux2))

  return nothing
end

@inline function calc_bar_states!(u, mesh, nonconservative_terms, equations, indicator, dg, cache)

  return nothing
end

@inline function calc_bar_states!(u, mesh,
                                  nonconservative_terms, equations, indicator::IndicatorMCL, dg, cache)
  @unpack lambda1, lambda2, bar_states1, bar_states2 = indicator.cache.ContainerShockCapturingIndicator

  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in 2:nnodes(dg)
      u_node     = get_node_vars(u, equations, dg, i,   j, element)
      u_node_im1 = get_node_vars(u, equations, dg, i-1, j, element)

      flux1     = flux(u_node,     1, equations)
      flux1_im1 = flux(u_node_im1, 1, equations)

      for v in eachvariable(equations)
        bar_states1[v, i, j, element] = 0.5 * (u_node[v] + u_node_im1[v]) - 0.5 * (flux1[v] - flux1_im1[v]) / lambda1[i, j, element]
      end
    end
    # bar_states1[:, 1,            :, element] .= zero(eltype(bar_states1))
    # bar_states1[:, nnodes(dg)+1, :, element] .= zero(eltype(bar_states1))

    for j in 2:nnodes(dg), i in eachnode(dg)
      u_node     = get_node_vars(u, equations, dg, i, j  , element)
      u_node_jm1 = get_node_vars(u, equations, dg, i, j-1, element)

      flux2     = flux(u_node,     2, equations)
      flux2_jm1 = flux(u_node_jm1, 2, equations)

      for v in eachvariable(equations)
        bar_states2[v, i, j, element] = 0.5 * (u_node[v] + u_node_jm1[v]) - 0.5 * (flux2[v] - flux2_jm1[v]) / lambda2[i, j, element]
      end
    end
    # bar_states2[:, :,            1, element] .= zero(eltype(bar_states2))
    # bar_states2[:, :, nnodes(dg)+1, element] .= zero(eltype(bar_states2))
  end

  return nothing
end

@inline function calc_var_bounds!(u, mesh, nonconservative_terms, equations, indicator, dg, cache)

  return nothing
end

@inline function calc_var_bounds!(u, mesh, nonconservative_terms, equations, indicator::IndicatorMCL, dg, cache)
  @unpack var_min, var_max, bar_states1, bar_states2 = indicator.cache.ContainerShockCapturingIndicator

  # Note: Bar states and lambdas at the interfaces are not needed anywhere else. Calculating here without saving.

  @threaded for element in eachelement(dg, cache)
    for v in eachvariable(equations)
      var_min[v, :, :, element] .= typemax(eltype(var_min))
      var_max[v, :, :, element] .= typemin(eltype(var_max))
    end

    for j in eachnode(dg), i in 2:nnodes(dg)
      bar_state_rho = bar_states1[1, i, j, element]
      var_min[1, i-1, j, element] = min(var_min[1, i-1, j, element], bar_state_rho)
      var_max[1, i-1, j, element] = max(var_max[1, i-1, j, element], bar_state_rho)
      var_min[1, i  , j, element] = min(var_min[1, i  , j, element], bar_state_rho)
      var_max[1, i  , j, element] = max(var_max[1, i  , j, element], bar_state_rho)
      for v in 2:nvariables(equations)
        bar_state_phi = bar_states1[v, i, j, element] / bar_state_rho
        var_min[v, i-1, j, element] = min(var_min[v, i-1, j, element], bar_state_phi)
        var_max[v, i-1, j, element] = max(var_max[v, i-1, j, element], bar_state_phi)
        var_min[v, i  , j, element] = min(var_min[v, i  , j, element], bar_state_phi)
        var_max[v, i  , j, element] = max(var_max[v, i  , j, element], bar_state_phi)
      end
    end
    for j in 2:nnodes(dg), i in eachnode(dg)
      bar_state_rho = bar_states2[1, i, j, element]
      var_min[1, i, j-1, element] = min(var_min[1, i, j-1, element], bar_state_rho)
      var_max[1, i, j-1, element] = max(var_max[1, i, j-1, element], bar_state_rho)
      var_min[1, i, j  , element] = min(var_min[1, i, j,   element], bar_state_rho)
      var_max[1, i, j  , element] = max(var_max[1, i, j,   element], bar_state_rho)
      for v in 2:nvariables(equations)
        bar_state_phi = bar_states2[v, i, j, element] / bar_state_rho
        var_min[v, i, j-1, element] = min(var_min[v, i, j-1, element], bar_state_phi)
        var_max[v, i, j-1, element] = max(var_max[v, i, j-1, element], bar_state_phi)
        var_min[v, i, j  , element] = min(var_min[v, i, j,   element], bar_state_phi)
        var_max[v, i, j  , element] = max(var_max[v, i, j,   element], bar_state_phi)
      end
    end
  end

  for interface in eachinterface(dg, cache)
    # Get neighboring element ids
    left  = cache.interfaces.neighbor_ids[1, interface]
    right = cache.interfaces.neighbor_ids[2, interface]

    orientation = cache.interfaces.orientations[interface]

    for i in eachnode(dg)
      if orientation == 1
        index_left  = (nnodes(dg), i, left)
        index_right = (1,          i, right)
      else
        index_left  = (i, nnodes(dg), left)
        index_right = (i,          1, right)
      end

      u_left  = get_node_vars(u, equations, dg, index_left...)
      u_right = get_node_vars(u, equations, dg, index_right...)

      flux_left  = flux(u_left,  orientation, equations)
      flux_right = flux(u_right, orientation, equations)
      lambda = max_abs_speed_naive(u_left, u_right, orientation, equations)

      bar_state_rho = 0.5 * (u_left[1] + u_right[1]) - 0.5 * (flux_right[1] - flux_left[1]) / lambda
      var_min[1, index_left...]  = min(var_min[1, index_left...], bar_state_rho)
      var_max[1, index_left...]  = max(var_max[1, index_left...], bar_state_rho)
      var_min[1, index_right...] = min(var_min[1, index_right...], bar_state_rho)
      var_max[1, index_right...] = max(var_max[1, index_right...], bar_state_rho)
      for v in 2:nvariables(equations)
        bar_state_phi = 0.5 * (u_left[v] + u_right[v]) - 0.5 * (flux_right[v] - flux_left[v]) / lambda
        bar_state_phi = bar_state_phi / bar_state_rho
        var_min[v, index_left...]  = min(var_min[v, index_left...], bar_state_phi)
        var_max[v, index_left...]  = max(var_max[v, index_left...], bar_state_phi)
        var_min[v, index_right...] = min(var_min[v, index_right...], bar_state_phi)
        var_max[v, index_right...] = max(var_max[v, index_right...], bar_state_phi)
      end
    end
  end

  return nothing
end

@inline function calcflux_antidiffusive_limited!(antidiffusive_flux1, antidiffusive_flux2,
                                                 u, mesh, nonconservative_terms, equations, indicator, dg, element, cache)
  @unpack antidiffusive_flux1_limited, antidiffusive_flux2_limited = cache.ContainerMCL2D
  @unpack var_min, var_max, lambda1, lambda2, bar_states1, bar_states2 = indicator.cache.ContainerShockCapturingIndicator

  for j in eachnode(dg), i in 2:nnodes(dg)
    # Limit density
    # bar_state = bar_states1[1, i, j, element]
    # if antidiffusive_flux1[1, i, j] > 0
    #   antidiffusive_flux1_limited[1, i, j, element] = min(antidiffusive_flux1[1, i, j],
    #       lambda1[i, j, element] * min(rho_max[i, j, element] - bar_state, bar_state - rho_min[i-1, j, element]))
    # else
    #   antidiffusive_flux1_limited[1, i, j, element] = max(antidiffusive_flux1[1, i, j],
    #       lambda1[i, j, element] * max(rho_min[i, j, element] - bar_state, bar_state - rho_max[i-1, j, element]))
    # end

    # alternative density limiting
    lambda = lambda1[i, j, element]
    bar_state_rho = lambda * bar_states1[1, i, j, element]
    f_min = max(lambda * var_min[1, i, j, element] - bar_state_rho,
                bar_state_rho - lambda * var_max[1, i-1, j, element])
    f_max = min(lambda * var_max[1, i, j, element] - bar_state_rho,
                bar_state_rho - lambda * var_min[1, i-1, j, element])
    antidiffusive_flux1_limited[1, i, j, element] = max(f_min, min(antidiffusive_flux1[1, i, j], f_max))

    # Limit velocity and total energy
    for v in 2:nvariables(equations)
      bar_states_phi = lambda * bar_states1[v, i, j, element]

      rho_limited_i   = bar_state_rho + antidiffusive_flux1_limited[1, i, j, element]
      rho_limited_im1 = bar_state_rho - antidiffusive_flux1_limited[1, i, j, element]

      phi = bar_states_phi / bar_state_rho

      antidiffusive_flux1_limited[v, i, j, element] = rho_limited_i * phi - bar_states_phi

      g = antidiffusive_flux1[v, i, j] - antidiffusive_flux1_limited[v, i, j, element]

      g_min = max(rho_limited_i   * (var_min[v, i, j, element] - phi),
                  rho_limited_im1 * (phi - var_max[v, i-1, j, element]))
      g_max = min(rho_limited_i   * (var_max[v, i, j, element] - phi),
                  rho_limited_im1 * (phi - var_min[v, i-1, j, element]))

      antidiffusive_flux1_limited[v, i, j, element] += max(g_min, min(g, g_max))
    end
  end

  for j in 2:nnodes(dg), i in eachnode(dg)
    # Limit density
    # bar_state = bar_states2[1, i, j, element]
    # if antidiffusive_flux2[1, i, j] > 0
    #   antidiffusive_flux2_limited[1, i, j, element] = min(antidiffusive_flux2[1, i, j],
    #       lambda2[i, j, element] * min(rho_max[i, j, element] - bar_state, bar_state - rho_min[i, j-1, element]))
    # else
    #   antidiffusive_flux2_limited[1, i, j, element] = max(antidiffusive_flux2[1, i, j],
    #       lambda2[i, j, element] * max(rho_min[i, j, element] - bar_state, bar_state - rho_max[i, j-1, element]))
    # end

    # alternative density limiting
    lambda = lambda2[i, j, element]
    bar_state_rho = lambda * bar_states2[1, i, j, element]
    f_min = max(lambda * var_min[1, i, j, element] - bar_state_rho,
                bar_state_rho - lambda * var_max[1, i, j-1, element])
    f_max = min(lambda * var_max[1, i, j, element] - bar_state_rho,
                bar_state_rho - lambda * var_min[1, i, j-1, element])
    antidiffusive_flux2_limited[1, i, j, element] = max(f_min, min(antidiffusive_flux2[1, i, j], f_max))

    # Limit velocity and total energy
    for v in 2:nvariables(equations)
      bar_state_phi = lambda * bar_states2[v, i, j, element]

      rho_limited_j   = bar_state_rho + antidiffusive_flux2_limited[1, i, j, element]
      rho_limited_jm1 = bar_state_rho - antidiffusive_flux2_limited[1, i, j, element]

      phi = bar_state_phi / bar_state_rho

      antidiffusive_flux2_limited[v, i, j, element] = rho_limited_j * phi - bar_state_phi

      g = antidiffusive_flux2[v, i, j] - antidiffusive_flux2_limited[v, i, j, element]

      g_min = max(rho_limited_j   * (var_min[v, i, j, element] - phi),
                  rho_limited_jm1 * (phi - var_max[v, i, j-1, element]))
      g_max = min(rho_limited_j   * (var_max[v, i, j, element] - phi),
                  rho_limited_jm1 * (phi - var_min[v, i, j-1, element]))

      antidiffusive_flux2_limited[v, i, j, element] += max(g_min, min(g, g_max))
    end
  end

  # Limit pressure
  if indicator.IDPPressureTVD
    for j in eachnode(dg), i in 2:nnodes(dg)
      bar_state_velocity = bar_states1[2, i, j, element]^2 + bar_states1[3, i, j, element]^2
      flux_velocity = antidiffusive_flux1_limited[2, i, j, element]^2 + antidiffusive_flux1_limited[3, i, j, element]^2

      Q = lambda1[i, j, element]^2 * (bar_states1[1, i, j, element] * bar_states1[4, i, j, element] -
                                      0.5 * bar_state_velocity)
      R_max = sqrt(bar_state_velocity * flux_velocity) +
              abs(bar_states1[1, i, j, element] * antidiffusive_flux1_limited[4, i, j, element]) +
              abs(bar_states1[4, i, j, element] * antidiffusive_flux1_limited[1, i, j, element])
      R_max *= lambda1[i, j, element]
      R_max += max(0, 0.5 * flux_velocity -
                      antidiffusive_flux1_limited[4, i, j, element] * antidiffusive_flux1_limited[1, i, j, element])

      if R_max > Q
        for v in eachvariable(equations)
          antidiffusive_flux1_limited[v, i, j, element] *= Q / R_max
        end
      end
    end

    for j in 2:nnodes(dg), i in eachnode(dg)
      bar_state_velocity = bar_states2[2, i, j, element]^2 + bar_states2[3, i, j, element]^2
      flux_velocity = antidiffusive_flux2_limited[2, i, j, element]^2 + antidiffusive_flux2_limited[3, i, j, element]^2

      Q = lambda2[i, j, element]^2 * (bar_states2[1, i, j, element] * bar_states2[4, i, j, element] -
                                      0.5 * bar_state_velocity)
      R_max = sqrt(bar_state_velocity * flux_velocity) +
              abs(bar_states2[1, i, j, element] * antidiffusive_flux2_limited[4, i, j, element]) +
              abs(bar_states2[4, i, j, element] * antidiffusive_flux2_limited[1, i, j, element])
      R_max *= lambda2[i, j, element]
      R_max += max(0, 0.5 * flux_velocity -
                      antidiffusive_flux2_limited[4, i, j, element] * antidiffusive_flux2_limited[1, i, j, element])

      if R_max > Q
        for v in eachvariable(equations)
          antidiffusive_flux2_limited[v, i, j, element] *= Q / R_max
        end
      end
    end
  end

  antidiffusive_flux1_limited[:,            1, :, element] .= zero(eltype(antidiffusive_flux1_limited))
  antidiffusive_flux1_limited[:, nnodes(dg)+1, :, element] .= zero(eltype(antidiffusive_flux1_limited))

  antidiffusive_flux2_limited[:, :,            1, element] .= zero(eltype(antidiffusive_flux2_limited))
  antidiffusive_flux2_limited[:, :, nnodes(dg)+1, element] .= zero(eltype(antidiffusive_flux2_limited))

  return nothing
end

@inline function calc_lambda!(u::AbstractArray{<:Any,4}, mesh, equations, dg, cache, indicator)

  return nothing
end

@inline function calc_lambda!(u::AbstractArray{<:Any,4}, mesh, equations, dg, cache, indicator::IndicatorMCL)
  @unpack lambda1, lambda2 = indicator.cache.ContainerShockCapturingIndicator

  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in 2:nnodes(dg)
      u_node     = get_node_vars(u, equations, dg, i,   j, element)
      u_node_im1 = get_node_vars(u, equations, dg, i-1, j, element)
      lambda1[i, j, element] = max_abs_speed_naive(u_node_im1, u_node, 1, equations)
    end
    lambda1[1,            :, element] .= zero(eltype(lambda1))
    lambda1[nnodes(dg)+1, :, element] .= zero(eltype(lambda1))

    for j in 2:nnodes(dg), i in eachnode(dg)
      u_node     = get_node_vars(u, equations, dg, i,   j, element)
      u_node_jm1 = get_node_vars(u, equations, dg, i, j-1, element)
      lambda2[i, j, element] = max_abs_speed_naive(u_node_jm1, u_node, 2, equations)
    end
    lambda2[:,            1, element] .= zero(eltype(lambda2))
    lambda2[:, nnodes(dg)+1, element] .= zero(eltype(lambda2))
  end

  return nothing
end

@inline function antidiffusive_stage!(u_ode, u_old_ode, dt, semi, indicator::IndicatorIDP)
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
  @unpack inverse_weights = solver.basis
  @unpack antidiffusive_flux1, antidiffusive_flux2 = cache.ContainerFCT2D

  u_old = wrap_array(u_old_ode, mesh, equations, solver, cache)
  u     = wrap_array(u_ode,     mesh, equations, solver, cache)

  @trixi_timeit timer() "alpha calculation" semi.solver.volume_integral.indicator(u, u_old, mesh, equations, solver, dt, cache)
  @unpack alpha1, alpha2 = semi.solver.volume_integral.indicator.cache.ContainerShockCapturingIndicator

  @threaded for element in eachelement(solver, cache)
    inverse_jacobian = -cache.elements.inverse_jacobian[element]

    # Calculate volume integral contribution
    # Note: antidiffusive_flux1[v, i, xi, element] = antidiffusive_flux2[v, xi, i, element] = 0 for all i in 1:nnodes and xi in {1, nnodes+1}
    for j in eachnode(solver), i in eachnode(solver)
      alpha_flux1     = (1.0 - alpha1[i,   j, element]) * get_node_vars(antidiffusive_flux1, equations, solver, i,   j, element)
      alpha_flux1_ip1 = (1.0 - alpha1[i+1, j, element]) * get_node_vars(antidiffusive_flux1, equations, solver, i+1, j, element)
      alpha_flux2     = (1.0 - alpha2[i,   j, element]) * get_node_vars(antidiffusive_flux2, equations, solver, i,   j, element)
      alpha_flux2_jp1 = (1.0 - alpha2[i, j+1, element]) * get_node_vars(antidiffusive_flux2, equations, solver, i, j+1, element)

      for v in eachvariable(equations)
        u[v, i, j, element] += dt * inverse_jacobian * (inverse_weights[i] * (alpha_flux1_ip1[v] - alpha_flux1[v]) +
                                                        inverse_weights[j] * (alpha_flux2_jp1[v] - alpha_flux2[v]) )
      end
    end
  end

  return nothing
end

@inline function antidiffusive_stage!(u_ode, u_old_ode, dt, semi, indicator::IndicatorMCL)

  return nothing
end

# 2d, IndicatorIDP
@inline function IDP_checkBounds(u::AbstractArray{<:Any,4}, mesh, equations, solver, cache, indicator::IndicatorIDP)
  @unpack IDPDensityTVD, IDPPressureTVD, IDPPositivity, IDPSpecEntropy, IDPMathEntropy = solver.volume_integral.indicator
  @unpack var_bounds = solver.volume_integral.indicator.cache.ContainerShockCapturingIndicator
  @unpack idp_bounds_delta_threaded = solver.volume_integral.indicator.cache

  @threaded for element in eachelement(solver, cache)
    idp_bounds_delta = idp_bounds_delta_threaded[Threads.threadid()]
    for j in eachnode(solver), i in eachnode(solver)
      counter = 0
      if IDPDensityTVD
        counter += 1 # rho_min
        idp_bounds_delta[counter] = max(idp_bounds_delta[counter], var_bounds[1][i, j, element] - u[1, i, j, element])
        counter += 1 # rho_max
        idp_bounds_delta[counter] = max(idp_bounds_delta[counter], u[1, i, j, element] - var_bounds[2][i, j, element])
      end
      if IDPPressureTVD
        p = pressure(get_node_vars(u, equations, solver, i, j, element), equations)
        counter += 1 # p_min
        idp_bounds_delta[counter] = max(idp_bounds_delta[counter], var_bounds[counter][i, j, element] - p)
        counter += 1 # p_max
        idp_bounds_delta[counter] = max(idp_bounds_delta[counter], p - var_bounds[counter][i, j, element])
      end
      if IDPPositivity && !IDPDensityTVD
        counter += 1 # rho_min
        idp_bounds_delta[counter] = max(idp_bounds_delta[counter], var_bounds[counter][i, j, element] - u[1, i, j, element])
      end
      if IDPPositivity && !IDPPressureTVD
        p = pressure(get_node_vars(u, equations, solver, i, j, element), equations)
        counter += 1 # p_min
        idp_bounds_delta[counter] = max(idp_bounds_delta[counter], var_bounds[counter][i, j, element] - p)
      end
      if IDPSpecEntropy
        s = entropy_spec(get_node_vars(u, equations, solver, i, j, element), equations)
        counter += 1 # s_min
        idp_bounds_delta[counter] = max(idp_bounds_delta[counter], var_bounds[counter][i, j, element] - s)
      end
      if IDPMathEntropy
        s = entropy_math(get_node_vars(u, equations, solver, i, j, element), equations)
        counter += 1 # s_max
        idp_bounds_delta[counter] = max(idp_bounds_delta[counter], s - var_bounds[counter][i, j, element])
      end
    end
  end

  return nothing
end

# 2d, IndicatorMCL
@inline function IDP_checkBounds(u::AbstractArray{<:Any,4}, mesh, equations, solver, cache, indicator::IndicatorMCL)
  @unpack var_min, var_max, bar_states1, bar_states2, lambda1, lambda2 = solver.volume_integral.indicator.cache.ContainerShockCapturingIndicator
  @unpack idp_bounds_delta_threaded = solver.volume_integral.indicator.cache
  @unpack antidiffusive_flux1_limited, antidiffusive_flux2_limited = cache.ContainerMCL2D

  @threaded for element in eachelement(solver, cache)
    idp_bounds_delta = idp_bounds_delta_threaded[Threads.threadid()]

    # Density
    err_lower_bound = zero(eltype(u))
    err_upper_bound = zero(eltype(u))
    for j in eachnode(solver), i in eachnode(solver)
      var_min_local = var_min[1, i, j, element]
      var_max_local = var_max[1, i, j, element]

      # -x
      if i > 1
        var_limited = bar_states1[1, i,   j, element] + antidiffusive_flux1_limited[1, i,   j, element] / lambda1[i,   j, element]
        err_lower_bound = max(err_lower_bound, var_min_local - var_limited)
        err_upper_bound = max(err_upper_bound, var_limited - var_max_local)
      end
      # +x
      if i < nnodes(solver)
        var_limited = bar_states1[1, i+1, j, element] - antidiffusive_flux1_limited[1, i+1, j, element] / lambda1[i+1, j, element]
        err_lower_bound = max(err_lower_bound, var_min_local - var_limited)
        err_upper_bound = max(err_upper_bound, var_limited - var_max_local)
      end
      # -y
      if j > 1
        var_limited = bar_states2[1, i,   j, element] + antidiffusive_flux2_limited[1, i,   j, element] / lambda2[i,   j, element]
        err_lower_bound = max(err_lower_bound, var_min_local - var_limited)
        err_upper_bound = max(err_upper_bound, var_limited - var_max_local)
      end
      # +y
      if j < nnodes(solver)
        var_limited = bar_states2[1, i, j+1, element] - antidiffusive_flux2_limited[1, i, j+1, element] / lambda2[i, j+1, element]
        err_lower_bound = max(err_lower_bound, var_min_local - var_limited)
        err_upper_bound = max(err_upper_bound, var_limited - var_max_local)
      end
    end
    idp_bounds_delta[1] = max(idp_bounds_delta[1], err_lower_bound)
    idp_bounds_delta[2] = max(idp_bounds_delta[2], err_upper_bound)

    # Velocity and total energy
    for v in 2:nvariables(equations)
      err_lower_bound = zero(eltype(u))
      err_upper_bound = zero(eltype(u))
      for j in eachnode(solver), i in eachnode(solver)
        var_min_local = var_min[v, i, j, element]
        var_max_local = var_max[v, i, j, element]

        # -x
        if i > 1
          rho_limited = bar_states1[1, i,   j, element] + antidiffusive_flux1_limited[1, i,   j, element] / lambda1[i,   j, element]
          var_limited = bar_states1[v, i,   j, element] + antidiffusive_flux1_limited[v, i,   j, element] / lambda1[i,   j, element]
          err_lower_bound = max(err_lower_bound, rho_limited * var_min_local - var_limited)
          err_upper_bound = max(err_upper_bound, var_limited - rho_limited * var_max_local)
        end
        # +x
        if i < nnodes(solver)
          rho_limited = bar_states1[1, i+1, j, element] - antidiffusive_flux1_limited[1, i+1, j, element] / lambda1[i+1, j, element]
          var_limited = bar_states1[v, i+1, j, element] - antidiffusive_flux1_limited[v, i+1, j, element] / lambda1[i+1, j, element]
          err_lower_bound = max(err_lower_bound, rho_limited * var_min_local - var_limited)
          err_upper_bound = max(err_upper_bound, var_limited - rho_limited * var_max_local)
        end
        # -y
        if j > 1
          rho_limited = bar_states2[1, i,   j, element] + antidiffusive_flux2_limited[1, i,   j, element] / lambda2[i,   j, element]
          var_limited = bar_states2[v, i,   j, element] + antidiffusive_flux2_limited[v, i,   j, element] / lambda2[i,   j, element]
          err_lower_bound = max(err_lower_bound, rho_limited * var_min_local - var_limited)
          err_upper_bound = max(err_upper_bound, var_limited - rho_limited * var_max_local)
        end
        # +y
        if j < nnodes(solver)
          rho_limited = bar_states2[1, i, j+1, element] - antidiffusive_flux2_limited[1, i, j+1, element] / lambda2[i, j+1, element]
          var_limited = bar_states2[v, i, j+1, element] - antidiffusive_flux2_limited[v, i, j+1, element] / lambda2[i, j+1, element]
          err_lower_bound = max(err_lower_bound, rho_limited * var_min_local - var_limited)
          err_upper_bound = max(err_upper_bound, var_limited - rho_limited * var_max_local)
        end
      end
      idp_bounds_delta[2*v-1] = max(idp_bounds_delta[2*v-1], err_lower_bound)
      idp_bounds_delta[2*v  ] = max(idp_bounds_delta[2*v  ], err_upper_bound)
    end
  end

  return nothing
end


function prolong2interfaces!(cache, u,
                             mesh::TreeMesh{2}, equations, surface_integral, dg::DG)
  @unpack interfaces = cache
  @unpack orientations = interfaces

  @threaded for interface in eachinterface(dg, cache)
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

function calc_interface_flux!(surface_flux_values,
                              mesh::TreeMesh{2},
                              nonconservative_terms::Val{false}, equations,
                              surface_integral, dg::DG, cache)
  @unpack surface_flux = surface_integral
  @unpack u, neighbor_ids, orientations = cache.interfaces

  @threaded for interface in eachinterface(dg, cache)
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
      for v in eachvariable(equations)
        surface_flux_values[v, i, left_direction,  left_id]  = flux[v]
        surface_flux_values[v, i, right_direction, right_id] = flux[v]
      end
    end
  end

  return nothing
end

function calc_interface_flux!(surface_flux_values,
                              mesh::TreeMesh{2},
                              nonconservative_terms::Val{true}, equations,
                              surface_integral, dg::DG, cache)
  surface_flux, nonconservative_flux = surface_integral.surface_flux
  @unpack u, neighbor_ids, orientations = cache.interfaces

  @threaded for interface in eachinterface(dg, cache)
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
      orientation = orientations[interface]
      u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, interface)
      flux = surface_flux(u_ll, u_rr, orientation, equations)

      # Compute both nonconservative fluxes
      noncons_left  = nonconservative_flux(u_ll, u_rr, orientation, equations)
      noncons_right = nonconservative_flux(u_rr, u_ll, orientation, equations)

      # Copy flux to left and right element storage
      for v in eachvariable(equations)
        # Note the factor 0.5 necessary for the nonconservative fluxes based on
        # the interpretation of global SBP operators coupled discontinuously via
        # central fluxes/SATs
        surface_flux_values[v, i, left_direction,  left_id]  = flux[v] + 0.5 * noncons_left[v]
        surface_flux_values[v, i, right_direction, right_id] = flux[v] + 0.5 * noncons_right[v]
      end
    end
  end

  return nothing
end


function prolong2boundaries!(cache, u,
                             mesh::TreeMesh{2}, equations, surface_integral, dg::DG)
  @unpack boundaries = cache
  @unpack orientations, neighbor_sides = boundaries

  @threaded for boundary in eachboundary(dg, cache)
    element = boundaries.neighbor_ids[boundary]

    if orientations[boundary] == 1
      # boundary in x-direction
      if neighbor_sides[boundary] == 1
        # element in -x direction of boundary
        for l in eachnode(dg), v in eachvariable(equations)
          boundaries.u[1, v, l, boundary] = u[v, nnodes(dg), l, element]
        end
      else # Element in +x direction of boundary
        for l in eachnode(dg), v in eachvariable(equations)
          boundaries.u[2, v, l, boundary] = u[v, 1,          l, element]
        end
      end
    else # if orientations[boundary] == 2
      # boundary in y-direction
      if neighbor_sides[boundary] == 1
        # element in -y direction of boundary
        for l in eachnode(dg), v in eachvariable(equations)
          boundaries.u[1, v, l, boundary] = u[v, l, nnodes(dg), element]
        end
      else
        # element in +y direction of boundary
        for l in eachnode(dg), v in eachvariable(equations)
          boundaries.u[2, v, l, boundary] = u[v, l, 1,          element]
        end
      end
    end
  end

  return nothing
end

# TODO: Taal dimension agnostic
function calc_boundary_flux!(cache, t, boundary_condition::BoundaryConditionPeriodic,
                             mesh::TreeMesh{2}, equations, surface_integral, dg::DG)
  @assert isempty(eachboundary(dg, cache))
end

function calc_boundary_flux!(cache, t, boundary_conditions::NamedTuple,
                             mesh::TreeMesh{2}, equations, surface_integral, dg::DG)
  @unpack surface_flux_values = cache.elements
  @unpack n_boundaries_per_direction = cache.boundaries

  # Calculate indices
  lasts = accumulate(+, n_boundaries_per_direction)
  firsts = lasts - n_boundaries_per_direction .+ 1

  # Calc boundary fluxes in each direction
  calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[1],
                                   equations, surface_integral, dg, cache,
                                   1, firsts[1], lasts[1])
  calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[2],
                                   equations, surface_integral, dg, cache,
                                   2, firsts[2], lasts[2])
  calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[3],
                                   equations, surface_integral, dg, cache,
                                   3, firsts[3], lasts[3])
  calc_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions[4],
                                   equations, surface_integral, dg, cache,
                                   4, firsts[4], lasts[4])
end

function calc_boundary_flux_by_direction!(surface_flux_values::AbstractArray{<:Any,4}, t,
                                          boundary_condition, equations,
                                          surface_integral ,dg::DG, cache,
                                          direction, first_boundary, last_boundary)
  @unpack surface_flux = surface_integral
  @unpack u, neighbor_ids, neighbor_sides, node_coordinates, orientations = cache.boundaries

  @threaded for boundary in first_boundary:last_boundary
    # Get neighboring element
    neighbor = neighbor_ids[boundary]

    for i in eachnode(dg)
      # Get boundary flux
      u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, boundary)
      if neighbor_sides[boundary] == 1 # Element is on the left, boundary on the right
        u_inner = u_ll
      else # Element is on the right, boundary on the left
        u_inner = u_rr
      end
      x = get_node_coords(node_coordinates, equations, dg, i, boundary)
      flux = boundary_condition(u_inner, orientations[boundary], direction, x, t, surface_flux,
                                equations)

      # Copy flux to left and right element storage
      for v in eachvariable(equations)
        surface_flux_values[v, i, direction, neighbor] = flux[v]
      end
    end
  end

  return nothing
end


function prolong2mortars!(cache, u,
                          mesh::TreeMesh{2}, equations,
                          mortar_l2::LobattoLegendreMortarL2, surface_integral, dg::DGSEM)

  @threaded for mortar in eachmortar(dg, cache)

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
        element_solutions_to_mortars!(cache.mortars, mortar_l2, leftright, mortar, u_large)
      else
        # L2 mortars in y-direction
        u_large = view(u, :, :, nnodes(dg), large_element)
        element_solutions_to_mortars!(cache.mortars, mortar_l2, leftright, mortar, u_large)
      end
    else # large_sides[mortar] == 2 -> large element on right side
      leftright = 2
      if cache.mortars.orientations[mortar] == 1
        # L2 mortars in x-direction
        u_large = view(u, :, 1, :, large_element)
        element_solutions_to_mortars!(cache.mortars, mortar_l2, leftright, mortar, u_large)
      else
        # L2 mortars in y-direction
        u_large = view(u, :, :, 1, large_element)
        element_solutions_to_mortars!(cache.mortars, mortar_l2, leftright, mortar, u_large)
      end
    end
  end

  return nothing
end

@inline function element_solutions_to_mortars!(mortars, mortar_l2::LobattoLegendreMortarL2, leftright, mortar,
                                               u_large::AbstractArray{<:Any,2})
  multiply_dimensionwise!(view(mortars.u_upper, leftright, :, :, mortar), mortar_l2.forward_upper, u_large)
  multiply_dimensionwise!(view(mortars.u_lower, leftright, :, :, mortar), mortar_l2.forward_lower, u_large)
  return nothing
end


function calc_mortar_flux!(surface_flux_values,
                           mesh::TreeMesh{2},
                           nonconservative_terms::Val{false}, equations,
                           mortar_l2::LobattoLegendreMortarL2,
                           surface_integral, dg::DG, cache)
  @unpack surface_flux = surface_integral
  @unpack u_lower, u_upper, orientations = cache.mortars
  @unpack fstar_upper_threaded, fstar_lower_threaded = cache

  @threaded for mortar in eachmortar(dg, cache)
    # Choose thread-specific pre-allocated container
    fstar_upper = fstar_upper_threaded[Threads.threadid()]
    fstar_lower = fstar_lower_threaded[Threads.threadid()]

    # Calculate fluxes
    orientation = orientations[mortar]
    calc_fstar!(fstar_upper, equations, surface_flux, dg, u_upper, mortar, orientation)
    calc_fstar!(fstar_lower, equations, surface_flux, dg, u_lower, mortar, orientation)

    mortar_fluxes_to_elements!(surface_flux_values,
                               mesh, equations, mortar_l2, dg, cache,
                               mortar, fstar_upper, fstar_lower)
  end

  return nothing
end

function calc_mortar_flux!(surface_flux_values,
                           mesh::TreeMesh{2},
                           nonconservative_terms::Val{true}, equations,
                           mortar_l2::LobattoLegendreMortarL2,
                           surface_integral, dg::DG, cache)
  surface_flux, nonconservative_flux = surface_integral.surface_flux
  @unpack u_lower, u_upper, orientations, large_sides = cache.mortars
  @unpack fstar_upper_threaded, fstar_lower_threaded = cache

  @threaded for mortar in eachmortar(dg, cache)
    # Choose thread-specific pre-allocated container
    fstar_upper = fstar_upper_threaded[Threads.threadid()]
    fstar_lower = fstar_lower_threaded[Threads.threadid()]

    # Calculate fluxes
    orientation = orientations[mortar]
    calc_fstar!(fstar_upper, equations, surface_flux, dg, u_upper, mortar, orientation)
    calc_fstar!(fstar_lower, equations, surface_flux, dg, u_lower, mortar, orientation)

    # Add nonconservative fluxes.
    # These need to be adapted on the geometry (left/right) since the order of
    # the arguments matters, based on the global SBP operator intepretation.
    # The same interpretation (global SBP operators coupled discontinuously via
    # central fluxes/SATs) explains why we need the factor 0.5.
    # Alternatively, you can also follow the argumentation of Bohm et al. 2018
    # ("nonconservative diamond flux")
    if large_sides[mortar] == 1 # -> small elements on right side
      for i in eachnode(dg)
        # Pull the left and right solutions
        u_upper_ll, u_upper_rr = get_surface_node_vars(u_upper, equations, dg, i, mortar)
        u_lower_ll, u_lower_rr = get_surface_node_vars(u_lower, equations, dg, i, mortar)
        # Call pointwise nonconservative term
        noncons_upper = nonconservative_flux(u_upper_ll, u_upper_rr, orientation, equations)
        noncons_lower = nonconservative_flux(u_lower_ll, u_lower_rr, orientation, equations)
        # Add to primary and secondary temporay storage
        multiply_add_to_node_vars!(fstar_upper, 0.5, noncons_upper, equations, dg, i)
        multiply_add_to_node_vars!(fstar_lower, 0.5, noncons_lower, equations, dg, i)
      end
    else # large_sides[mortar] == 2 -> small elements on the left
      for i in eachnode(dg)
        # Pull the left and right solutions
        u_upper_ll, u_upper_rr = get_surface_node_vars(u_upper, equations, dg, i, mortar)
        u_lower_ll, u_lower_rr = get_surface_node_vars(u_lower, equations, dg, i, mortar)
        # Call pointwise nonconservative term
        noncons_upper = nonconservative_flux(u_upper_rr, u_upper_ll, orientation, equations)
        noncons_lower = nonconservative_flux(u_lower_rr, u_lower_ll, orientation, equations)
        # Add to primary and secondary temporay storage
        multiply_add_to_node_vars!(fstar_upper, 0.5, noncons_upper, equations, dg, i)
        multiply_add_to_node_vars!(fstar_lower, 0.5, noncons_lower, equations, dg, i)
      end
    end

    mortar_fluxes_to_elements!(surface_flux_values,
                               mesh, equations, mortar_l2, dg, cache,
                               mortar, fstar_upper, fstar_lower)
  end

  return nothing
end


@inline function calc_fstar!(destination::AbstractArray{<:Any,2}, equations,
                             surface_flux, dg::DGSEM,
                             u_interfaces, interface, orientation)

  for i in eachnode(dg)
    # Call pointwise two-point numerical flux function
    u_ll, u_rr = get_surface_node_vars(u_interfaces, equations, dg, i, interface)
    flux = surface_flux(u_ll, u_rr, orientation, equations)

    # Copy flux to left and right element storage
    set_node_vars!(destination, flux, equations, dg, i)
  end

  return nothing
end

@inline function mortar_fluxes_to_elements!(surface_flux_values,
                                            mesh::TreeMesh{2}, equations,
                                            mortar_l2::LobattoLegendreMortarL2,
                                            dg::DGSEM, cache,
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

  # TODO: Taal performance
  # for v in eachvariable(equations)
  #   # The code below is semantically equivalent to
  #   # surface_flux_values[v, :, direction, large_element] .=
  #   #   (mortar_l2.reverse_upper * fstar_upper[v, :] + mortar_l2.reverse_lower * fstar_lower[v, :])
  #   # but faster and does not allocate.
  #   # Note that `true * some_float == some_float` in Julia, i.e. `true` acts as
  #   # a universal `one`. Hence, the second `mul!` means "add the matrix-vector
  #   # product to the current value of the destination".
  #   @views mul!(surface_flux_values[v, :, direction, large_element],
  #               mortar_l2.reverse_upper, fstar_upper[v, :])
  #   @views mul!(surface_flux_values[v, :, direction, large_element],
  #               mortar_l2.reverse_lower,  fstar_lower[v, :], true, true)
  # end
  # The code above could be replaced by the following code. However, the relative efficiency
  # depends on the types of fstar_upper/fstar_lower and dg.l2mortar_reverse_upper.
  # Using StaticArrays for both makes the code above faster for common test cases.
  multiply_dimensionwise!(
    view(surface_flux_values, :, :, direction, large_element), mortar_l2.reverse_upper, fstar_upper,
                                                               mortar_l2.reverse_lower, fstar_lower)

  return nothing
end


function calc_surface_integral!(du, u, mesh::Union{TreeMesh{2}, StructuredMesh{2}},
                                equations, surface_integral::SurfaceIntegralWeakForm,
                                dg::DG, cache)
  @unpack boundary_interpolation = dg.basis
  @unpack surface_flux_values = cache.elements

  # Note that all fluxes have been computed with outward-pointing normal vectors.
  # Access the factors only once before beginning the loop to increase performance.
  # We also use explicit assignments instead of `+=` to let `@muladd` turn these
  # into FMAs (see comment at the top of the file).
  factor_1 = boundary_interpolation[1,          1]
  factor_2 = boundary_interpolation[nnodes(dg), 2]
  @threaded for element in eachelement(dg, cache)
    for l in eachnode(dg)
      for v in eachvariable(equations)
        # surface at -x
        du[v, 1,          l, element] = (
          du[v, 1,          l, element] - surface_flux_values[v, l, 1, element] * factor_1)

        # surface at +x
        du[v, nnodes(dg), l, element] = (
          du[v, nnodes(dg), l, element] + surface_flux_values[v, l, 2, element] * factor_2)

        # surface at -y
        du[v, l, 1,          element] = (
          du[v, l, 1,          element] - surface_flux_values[v, l, 3, element] * factor_1)

        # surface at +y
        du[v, l, nnodes(dg), element] = (
          du[v, l, nnodes(dg), element] + surface_flux_values[v, l, 4, element] * factor_2)
      end
    end
  end

  return nothing
end


function apply_jacobian!(du, mesh::TreeMesh{2},
                         equations, dg::DG, cache)

  @threaded for element in eachelement(dg, cache)
    factor = -cache.elements.inverse_jacobian[element]

    for j in eachnode(dg), i in eachnode(dg)
      for v in eachvariable(equations)
        du[v, i, j, element] *= factor
      end
    end
  end

  return nothing
end


# TODO: Taal dimension agnostic
function calc_sources!(du, u, t, source_terms::Nothing,
                       equations::AbstractEquations{2}, dg::DG, cache)
  return nothing
end

function calc_sources!(du, u, t, source_terms,
                       equations::AbstractEquations{2}, dg::DG, cache)

  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, j, element)
      x_local = get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, element)
      du_local = source_terms(u_local, x_local, t, equations)
      add_to_node_vars!(du, du_local, equations, dg, i, j, element)
    end
  end

  return nothing
end


end # @muladd
