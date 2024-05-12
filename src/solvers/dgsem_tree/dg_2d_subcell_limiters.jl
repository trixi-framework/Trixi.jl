# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function create_cache(mesh::Union{TreeMesh{2}, StructuredMesh{2}, P4estMesh{2}},
                      equations, volume_integral::VolumeIntegralSubcellLimiting,
                      dg::DG, uEltype)
    cache = create_cache(mesh, equations,
                         VolumeIntegralPureLGLFiniteVolume(volume_integral.volume_flux_fv),
                         dg, uEltype)
    if volume_integral.limiter.smoothness_indicator
        element_ids_dg = Int[]
        element_ids_dgfv = Int[]
        cache = (; cache..., element_ids_dg, element_ids_dgfv)
    end

    A3dp1_x = Array{uEltype, 3}
    A3dp1_y = Array{uEltype, 3}
    A3d = Array{uEltype, 3}
    A4d = Array{uEltype, 4}

    fhat1_L_threaded = A3dp1_x[A3dp1_x(undef, nvariables(equations), nnodes(dg) + 1,
                                       nnodes(dg)) for _ in 1:Threads.nthreads()]
    fhat2_L_threaded = A3dp1_y[A3dp1_y(undef, nvariables(equations), nnodes(dg),
                                       nnodes(dg) + 1) for _ in 1:Threads.nthreads()]
    fhat1_R_threaded = A3dp1_x[A3dp1_x(undef, nvariables(equations), nnodes(dg) + 1,
                                       nnodes(dg)) for _ in 1:Threads.nthreads()]
    fhat2_R_threaded = A3dp1_y[A3dp1_y(undef, nvariables(equations), nnodes(dg),
                                       nnodes(dg) + 1) for _ in 1:Threads.nthreads()]
    flux_temp_threaded = A3d[A3d(undef, nvariables(equations), nnodes(dg), nnodes(dg))
                             for _ in 1:Threads.nthreads()]
    fhat_temp_threaded = A3d[A3d(undef, nvariables(equations), nnodes(dg),
                                 nnodes(dg))
                             for _ in 1:Threads.nthreads()]
    antidiffusive_fluxes = Trixi.ContainerAntidiffusiveFlux2D{uEltype}(0,
                                                                       nvariables(equations),
                                                                       nnodes(dg))

    if have_nonconservative_terms(equations) == true
        flux_nonconservative_temp_threaded = A4d[A4d(undef, nvariables(equations),
                                                     n_nonconservative_terms(equations),
                                                     nnodes(dg), nnodes(dg))
                                                 for _ in 1:Threads.nthreads()]
        fhat_nonconservative_temp_threaded = A4d[A4d(undef, nvariables(equations),
                                                     n_nonconservative_terms(equations),
                                                     nnodes(dg), nnodes(dg))
                                                 for _ in 1:Threads.nthreads()]
        phi_threaded = A4d[A4d(undef, nvariables(equations),
                               n_nonconservative_terms(equations),
                               nnodes(dg), nnodes(dg))
                           for _ in 1:Threads.nthreads()]
        cache = (; cache..., flux_nonconservative_temp_threaded,
                 fhat_nonconservative_temp_threaded, phi_threaded)
    end

    return (; cache..., antidiffusive_fluxes,
            fhat1_L_threaded, fhat2_L_threaded, fhat1_R_threaded, fhat2_R_threaded,
            flux_temp_threaded, fhat_temp_threaded)
end

function calc_volume_integral!(du, u,
                               mesh::Union{TreeMesh{2}, StructuredMesh{2},
                                           P4estMesh{2}},
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralSubcellLimiting,
                               dg::DGSEM, cache, t, boundary_conditions)
    @unpack limiter = volume_integral

    # Calculate lambdas and bar states
    @trixi_timeit timer() "calc_lambdas_bar_states!" calc_lambdas_bar_states!(u, t,
                                                                              mesh,
                                                                              nonconservative_terms,
                                                                              equations,
                                                                              limiter,
                                                                              dg, cache,
                                                                              boundary_conditions)
    # Calculate boundaries
    @trixi_timeit timer() "calc_variable_bounds!" calc_variable_bounds!(u, mesh,
                                                                        nonconservative_terms,
                                                                        equations,
                                                                        limiter, dg,
                                                                        cache)

    if limiter.smoothness_indicator
        (; element_ids_dg, element_ids_dgfv) = cache
        # Calculate element-wise blending factors α
        alpha_element = @trixi_timeit timer() "element-wise blending factors" limiter.IndicatorHG(u,
                                                                                                  mesh,
                                                                                                  equations,
                                                                                                  dg,
                                                                                                  cache)

        # Determine element ids for DG-only and subcell-wise blended DG-FV volume integral
        pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha_element,
                                      dg, cache)

        # Loop over pure DG elements
        @trixi_timeit timer() "pure DG" @threaded for idx_element in eachindex(element_ids_dg)
            element = element_ids_dg[idx_element]
            flux_differencing_kernel!(du, u, element, mesh,
                                      nonconservative_terms, equations,
                                      volume_integral.volume_flux_dg, dg, cache)
        end

        # Loop over blended DG-FV elements
        @trixi_timeit timer() "subcell-wise blended DG-FV" @threaded for idx_element in eachindex(element_ids_dgfv)
            element = element_ids_dgfv[idx_element]
            subcell_limiting_kernel!(du, u, element, mesh,
                                     nonconservative_terms, equations,
                                     volume_integral, limiter,
                                     dg, cache)
        end
    else # limiter.smoothness_indicator == false
        # Loop over all elements
        @trixi_timeit timer() "subcell-wise blended DG-FV" @threaded for element in eachelement(dg,
                                                                                                cache)
            subcell_limiting_kernel!(du, u, element, mesh,
                                     nonconservative_terms, equations,
                                     volume_integral, limiter,
                                     dg, cache)
        end
    end
end

@inline function subcell_limiting_kernel!(du, u, element,
                                          mesh::Union{TreeMesh{2}, StructuredMesh{2},
                                                      P4estMesh{2}},
                                          nonconservative_terms, equations,
                                          volume_integral, limiter::SubcellLimiterIDP,
                                          dg::DGSEM, cache)
    @unpack inverse_weights = dg.basis
    @unpack volume_flux_dg, volume_flux_fv = volume_integral

    # high-order DG fluxes
    @unpack fhat1_L_threaded, fhat1_R_threaded, fhat2_L_threaded, fhat2_R_threaded = cache

    fhat1_L = fhat1_L_threaded[Threads.threadid()]
    fhat1_R = fhat1_R_threaded[Threads.threadid()]
    fhat2_L = fhat2_L_threaded[Threads.threadid()]
    fhat2_R = fhat2_R_threaded[Threads.threadid()]
    calcflux_fhat!(fhat1_L, fhat1_R, fhat2_L, fhat2_R, u, mesh,
                   nonconservative_terms, equations, volume_flux_dg, dg, element,
                   cache)

    # low-order FV fluxes
    @unpack fstar1_L_threaded, fstar1_R_threaded, fstar2_L_threaded, fstar2_R_threaded = cache

    fstar1_L = fstar1_L_threaded[Threads.threadid()]
    fstar2_L = fstar2_L_threaded[Threads.threadid()]
    fstar1_R = fstar1_R_threaded[Threads.threadid()]
    fstar2_R = fstar2_R_threaded[Threads.threadid()]
    calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u, mesh,
                 nonconservative_terms, equations, volume_flux_fv, dg, element,
                 cache)

    # antidiffusive flux
    calcflux_antidiffusive!(fhat1_L, fhat1_R, fhat2_L, fhat2_R,
                            fstar1_L, fstar1_R, fstar2_L, fstar2_R,
                            u, mesh, nonconservative_terms, equations, limiter, dg,
                            element, cache)

    # Calculate volume integral contribution of low-order FV flux
    for j in eachnode(dg), i in eachnode(dg)
        for v in eachvariable(equations)
            du[v, i, j, element] += inverse_weights[i] *
                                    (fstar1_L[v, i + 1, j] - fstar1_R[v, i, j]) +
                                    inverse_weights[j] *
                                    (fstar2_L[v, i, j + 1] - fstar2_R[v, i, j])
        end
    end

    return nothing
end

@inline function subcell_limiting_kernel!(du, u,
                                          element,
                                          mesh::Union{TreeMesh{2}, StructuredMesh{2},
                                                      P4estMesh{2}},
                                          nonconservative_terms::False, equations,
                                          volume_integral, limiter::SubcellLimiterMCL,
                                          dg::DGSEM, cache)
    (; inverse_weights) = dg.basis
    (; volume_flux_dg, volume_flux_fv) = volume_integral

    # high-order DG fluxes
    (; fhat1_L_threaded, fhat1_R_threaded, fhat2_L_threaded, fhat2_R_threaded) = cache
    fhat1_L = fhat1_L_threaded[Threads.threadid()]
    fhat1_R = fhat1_R_threaded[Threads.threadid()]
    fhat2_L = fhat2_L_threaded[Threads.threadid()]
    fhat2_R = fhat2_R_threaded[Threads.threadid()]
    calcflux_fhat!(fhat1_L, fhat1_R, fhat2_L, fhat2_R, u, mesh,
                   nonconservative_terms, equations, volume_flux_dg, dg, element,
                   cache)

    # low-order FV fluxes
    (; fstar1_L_threaded, fstar1_R_threaded, fstar2_L_threaded, fstar2_R_threaded) = cache
    fstar1_L = fstar1_L_threaded[Threads.threadid()]
    fstar2_L = fstar2_L_threaded[Threads.threadid()]
    fstar1_R = fstar1_R_threaded[Threads.threadid()]
    fstar2_R = fstar2_R_threaded[Threads.threadid()]
    calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u, mesh,
                 nonconservative_terms, equations, volume_flux_fv, dg, element,
                 cache)

    # antidiffusive flux
    calcflux_antidiffusive!(fhat1_L, fhat1_R, fhat2_L, fhat2_R,
                            fstar1_L, fstar1_R, fstar2_L, fstar2_R,
                            u, mesh, nonconservative_terms, equations, limiter, dg,
                            element, cache)

    # limit antidiffusive flux
    calcflux_antidiffusive_limited!(u, mesh, nonconservative_terms, equations,
                                    limiter, dg, element, cache,
                                    fstar1_L, fstar2_L)

    (; antidiffusive_flux1_L, antidiffusive_flux2_L, antidiffusive_flux1_R, antidiffusive_flux2_R) = cache.antidiffusive_fluxes
    for j in eachnode(dg), i in eachnode(dg)
        for v in eachvariable(equations)
            du[v, i, j, element] += inverse_weights[i] *
                                    (fstar1_L[v, i + 1, j] - fstar1_R[v, i, j]) +
                                    inverse_weights[j] *
                                    (fstar2_L[v, i, j + 1] - fstar2_R[v, i, j])

            du[v, i, j, element] += inverse_weights[i] *
                                    (-antidiffusive_flux1_L[v, i + 1, j, element] +
                                     antidiffusive_flux1_R[v, i, j, element]) +
                                    inverse_weights[j] *
                                    (-antidiffusive_flux2_L[v, i, j + 1, element] +
                                     antidiffusive_flux2_R[v, i, j, element])
        end
    end

    return nothing
end

# Calculate the DG staggered volume fluxes `fhat` in subcell FV-form inside the element
# (**without non-conservative terms**).
#
# See also `flux_differencing_kernel!`.
@inline function calcflux_fhat!(fhat1_L, fhat1_R, fhat2_L, fhat2_R, u,
                                mesh::TreeMesh{2}, nonconservative_terms::False,
                                equations,
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
        # of the `volume_flux` to save half of the possible two-point flux
        # computations.
        for ii in (i + 1):nnodes(dg)
            u_node_ii = get_node_vars(u, equations, dg, ii, j, element)
            flux1 = volume_flux(u_node, u_node_ii, 1, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[i, ii], flux1,
                                       equations, dg, i, j)
            multiply_add_to_node_vars!(flux_temp, derivative_split[ii, i], flux1,
                                       equations, dg, ii, j)
        end
    end

    # FV-form flux `fhat` in x direction
    fhat1_L[:, 1, :] .= zero(eltype(fhat1_L))
    fhat1_L[:, nnodes(dg) + 1, :] .= zero(eltype(fhat1_L))
    fhat1_R[:, 1, :] .= zero(eltype(fhat1_R))
    fhat1_R[:, nnodes(dg) + 1, :] .= zero(eltype(fhat1_R))

    for j in eachnode(dg), i in 1:(nnodes(dg) - 1), v in eachvariable(equations)
        fhat1_L[v, i + 1, j] = fhat1_L[v, i, j] + weights[i] * flux_temp[v, i, j]
        fhat1_R[v, i + 1, j] = fhat1_L[v, i + 1, j]
    end

    # Split form volume flux in orientation 2: y direction
    flux_temp .= zero(eltype(flux_temp))

    for j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, element)
        for jj in (j + 1):nnodes(dg)
            u_node_jj = get_node_vars(u, equations, dg, i, jj, element)
            flux2 = volume_flux(u_node, u_node_jj, 2, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[j, jj], flux2,
                                       equations, dg, i, j)
            multiply_add_to_node_vars!(flux_temp, derivative_split[jj, j], flux2,
                                       equations, dg, i, jj)
        end
    end

    # FV-form flux `fhat` in y direction
    fhat2_L[:, :, 1] .= zero(eltype(fhat2_L))
    fhat2_L[:, :, nnodes(dg) + 1] .= zero(eltype(fhat2_L))
    fhat2_R[:, :, 1] .= zero(eltype(fhat2_R))
    fhat2_R[:, :, nnodes(dg) + 1] .= zero(eltype(fhat2_R))

    for j in 1:(nnodes(dg) - 1), i in eachnode(dg), v in eachvariable(equations)
        fhat2_L[v, i, j + 1] = fhat2_L[v, i, j] + weights[j] * flux_temp[v, i, j]
        fhat2_R[v, i, j + 1] = fhat2_L[v, i, j + 1]
    end

    return nothing
end

# Calculate the DG staggered volume fluxes `fhat` in subcell FV-form inside the element
# (**with non-conservative terms**).
#
# See also `flux_differencing_kernel!`.
#
# The calculation of the non-conservative staggered "fluxes" requires non-conservative
# terms that can be written as a product of local and a symmetric contributions. See, e.g.,
#
# - Rueda-Ramírez, Gassner (2023). A Flux-Differencing Formula for Split-Form Summation By Parts
#   Discretizations of Non-Conservative Systems. https://arxiv.org/pdf/2211.14009.pdf.
#
@inline function calcflux_fhat!(fhat1_L, fhat1_R, fhat2_L, fhat2_R, u,
                                mesh::TreeMesh{2}, nonconservative_terms::True,
                                equations,
                                volume_flux, dg::DGSEM, element, cache)
    @unpack weights, derivative_split = dg.basis
    @unpack flux_temp_threaded, flux_nonconservative_temp_threaded = cache
    @unpack fhat_temp_threaded, fhat_nonconservative_temp_threaded, phi_threaded = cache

    volume_flux_cons, volume_flux_noncons = volume_flux

    flux_temp = flux_temp_threaded[Threads.threadid()]
    flux_noncons_temp = flux_nonconservative_temp_threaded[Threads.threadid()]

    fhat_temp = fhat_temp_threaded[Threads.threadid()]
    fhat_noncons_temp = fhat_nonconservative_temp_threaded[Threads.threadid()]
    phi = phi_threaded[Threads.threadid()]

    # The FV-form fluxes are calculated in a recursive manner, i.e.:
    # fhat_(0,1)   = w_0 * FVol_0,
    # fhat_(j,j+1) = fhat_(j-1,j) + w_j * FVol_j,   for j=1,...,N-1,
    # with the split form volume fluxes FVol_j = -2 * sum_i=0^N D_ji f*_(j,i).

    # To use the symmetry of the `volume_flux`, the split form volume flux is precalculated
    # like in `calc_volume_integral!` for the `VolumeIntegralFluxDifferencing`
    # and saved in in `flux_temp`.

    # Split form volume flux in orientation 1: x direction
    flux_temp .= zero(eltype(flux_temp))
    flux_noncons_temp .= zero(eltype(flux_noncons_temp))

    for j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, element)

        # All diagonal entries of `derivative_split` are zero. Thus, we can skip
        # the computation of the diagonal terms. In addition, we use the symmetry
        # of `volume_flux_cons` and `volume_flux_noncons` to save half of the possible two-point flux
        # computations.
        for ii in (i + 1):nnodes(dg)
            u_node_ii = get_node_vars(u, equations, dg, ii, j, element)
            flux1 = volume_flux_cons(u_node, u_node_ii, 1, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[i, ii], flux1,
                                       equations, dg, i, j)
            multiply_add_to_node_vars!(flux_temp, derivative_split[ii, i], flux1,
                                       equations, dg, ii, j)
            for noncons in 1:n_nonconservative_terms(equations)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux1_noncons = volume_flux_noncons(u_node, u_node_ii, 1, equations,
                                                    NonConservativeSymmetric(), noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5 * derivative_split[i, ii],
                                           flux1_noncons,
                                           equations, dg, noncons, i, j)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5 * derivative_split[ii, i],
                                           flux1_noncons,
                                           equations, dg, noncons, ii, j)
            end
        end
    end

    # FV-form flux `fhat` in x direction
    fhat1_L[:, 1, :] .= zero(eltype(fhat1_L))
    fhat1_L[:, nnodes(dg) + 1, :] .= zero(eltype(fhat1_L))
    fhat1_R[:, 1, :] .= zero(eltype(fhat1_R))
    fhat1_R[:, nnodes(dg) + 1, :] .= zero(eltype(fhat1_R))

    fhat_temp[:, 1, :] .= zero(eltype(fhat1_L))
    fhat_noncons_temp[:, :, 1, :] .= zero(eltype(fhat1_L))

    # Compute local contribution to non-conservative flux
    for j in eachnode(dg), i in eachnode(dg)
        u_local = get_node_vars(u, equations, dg, i, j, element)
        for noncons in 1:n_nonconservative_terms(equations)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, 1, equations,
                                               NonConservativeLocal(), noncons),
                           equations, dg, noncons, i, j)
        end
    end

    for j in eachnode(dg), i in 1:(nnodes(dg) - 1)
        # Conservative part
        for v in eachvariable(equations)
            value = fhat_temp[v, i, j] + weights[i] * flux_temp[v, i, j]
            fhat_temp[v, i + 1, j] = value
            fhat1_L[v, i + 1, j] = value
            fhat1_R[v, i + 1, j] = value
        end
        # Nonconservative part
        for noncons in 1:n_nonconservative_terms(equations),
            v in eachvariable(equations)

            value = fhat_noncons_temp[v, noncons, i, j] +
                    weights[i] * flux_noncons_temp[v, noncons, i, j]
            fhat_noncons_temp[v, noncons, i + 1, j] = value

            fhat1_L[v, i + 1, j] = fhat1_L[v, i + 1, j] + phi[v, noncons, i, j] * value
            fhat1_R[v, i + 1, j] = fhat1_R[v, i + 1, j] +
                                   phi[v, noncons, i + 1, j] * value
        end
    end

    # Split form volume flux in orientation 2: y direction
    flux_temp .= zero(eltype(flux_temp))
    flux_noncons_temp .= zero(eltype(flux_noncons_temp))

    for j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, element)
        for jj in (j + 1):nnodes(dg)
            u_node_jj = get_node_vars(u, equations, dg, i, jj, element)
            flux2 = volume_flux_cons(u_node, u_node_jj, 2, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[j, jj], flux2,
                                       equations, dg, i, j)
            multiply_add_to_node_vars!(flux_temp, derivative_split[jj, j], flux2,
                                       equations, dg, i, jj)
            for noncons in 1:n_nonconservative_terms(equations)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux2_noncons = volume_flux_noncons(u_node, u_node_jj, 2, equations,
                                                    NonConservativeSymmetric(), noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5 * derivative_split[j, jj],
                                           flux2_noncons,
                                           equations, dg, noncons, i, j)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5 * derivative_split[jj, j],
                                           flux2_noncons,
                                           equations, dg, noncons, i, jj)
            end
        end
    end

    # FV-form flux `fhat` in y direction
    fhat2_L[:, :, 1] .= zero(eltype(fhat2_L))
    fhat2_L[:, :, nnodes(dg) + 1] .= zero(eltype(fhat2_L))
    fhat2_R[:, :, 1] .= zero(eltype(fhat2_R))
    fhat2_R[:, :, nnodes(dg) + 1] .= zero(eltype(fhat2_R))

    fhat_temp[:, :, 1] .= zero(eltype(fhat1_L))
    fhat_noncons_temp[:, :, :, 1] .= zero(eltype(fhat1_L))

    # Compute local contribution to non-conservative flux
    for j in eachnode(dg), i in eachnode(dg)
        u_local = get_node_vars(u, equations, dg, i, j, element)
        for noncons in 1:n_nonconservative_terms(equations)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, 2, equations,
                                               NonConservativeLocal(), noncons),
                           equations, dg, noncons, i, j)
        end
    end

    for j in 1:(nnodes(dg) - 1), i in eachnode(dg)
        # Conservative part
        for v in eachvariable(equations)
            value = fhat_temp[v, i, j] + weights[j] * flux_temp[v, i, j]
            fhat_temp[v, i, j + 1] = value
            fhat2_L[v, i, j + 1] = value
            fhat2_R[v, i, j + 1] = value
        end
        # Nonconservative part
        for noncons in 1:n_nonconservative_terms(equations),
            v in eachvariable(equations)

            value = fhat_noncons_temp[v, noncons, i, j] +
                    weights[j] * flux_noncons_temp[v, noncons, i, j]
            fhat_noncons_temp[v, noncons, i, j + 1] = value

            fhat2_L[v, i, j + 1] = fhat2_L[v, i, j + 1] + phi[v, noncons, i, j] * value
            fhat2_R[v, i, j + 1] = fhat2_R[v, i, j + 1] +
                                   phi[v, noncons, i, j + 1] * value
        end
    end

    return nothing
end

# Calculate the antidiffusive flux `antidiffusive_flux` as the subtraction between `fhat` and `fstar` for conservative systems.
@inline function calcflux_antidiffusive!(fhat1_L, fhat1_R, fhat2_L, fhat2_R,
                                         fstar1_L, fstar1_R, fstar2_L, fstar2_R,
                                         u, mesh,
                                         nonconservative_terms::False, equations,
                                         limiter::SubcellLimiterIDP, dg, element, cache)
    @unpack antidiffusive_flux1_L, antidiffusive_flux2_L, antidiffusive_flux1_R, antidiffusive_flux2_R = cache.antidiffusive_fluxes

    for j in eachnode(dg), i in 2:nnodes(dg)
        for v in eachvariable(equations)
            antidiffusive_flux1_L[v, i, j, element] = fhat1_L[v, i, j] -
                                                      fstar1_L[v, i, j]
            antidiffusive_flux1_R[v, i, j, element] = antidiffusive_flux1_L[v, i, j,
                                                                            element]
        end
    end
    for j in 2:nnodes(dg), i in eachnode(dg)
        for v in eachvariable(equations)
            antidiffusive_flux2_L[v, i, j, element] = fhat2_L[v, i, j] -
                                                      fstar2_L[v, i, j]
            antidiffusive_flux2_R[v, i, j, element] = antidiffusive_flux2_L[v, i, j,
                                                                            element]
        end
    end

    antidiffusive_flux1_L[:, 1, :, element] .= zero(eltype(antidiffusive_flux1_L))
    antidiffusive_flux1_L[:, nnodes(dg) + 1, :, element] .= zero(eltype(antidiffusive_flux1_L))
    antidiffusive_flux1_R[:, 1, :, element] .= zero(eltype(antidiffusive_flux1_R))
    antidiffusive_flux1_R[:, nnodes(dg) + 1, :, element] .= zero(eltype(antidiffusive_flux1_R))

    antidiffusive_flux2_L[:, :, 1, element] .= zero(eltype(antidiffusive_flux2_L))
    antidiffusive_flux2_L[:, :, nnodes(dg) + 1, element] .= zero(eltype(antidiffusive_flux2_L))
    antidiffusive_flux2_R[:, :, 1, element] .= zero(eltype(antidiffusive_flux2_R))
    antidiffusive_flux2_R[:, :, nnodes(dg) + 1, element] .= zero(eltype(antidiffusive_flux2_R))

    return nothing
end

# Calculate the antidiffusive flux `antidiffusive_flux` as the subtraction between `fhat` and `fstar` for conservative systems.
@inline function calcflux_antidiffusive!(fhat1_L, fhat1_R, fhat2_L, fhat2_R,
                                         fstar1_L, fstar1_R, fstar2_L, fstar2_R,
                                         u, mesh,
                                         nonconservative_terms::True, equations,
                                         limiter::SubcellLimiterIDP, dg, element, cache)
    @unpack antidiffusive_flux1_L, antidiffusive_flux2_L, antidiffusive_flux1_R, antidiffusive_flux2_R = cache.antidiffusive_fluxes

    for j in eachnode(dg), i in 2:nnodes(dg)
        for v in eachvariable(equations)
            antidiffusive_flux1_L[v, i, j, element] = fhat1_L[v, i, j] -
                                                      fstar1_L[v, i, j]
            antidiffusive_flux1_R[v, i, j, element] = fhat1_R[v, i, j] -
                                                      fstar1_R[v, i, j]
        end
    end
    for j in 2:nnodes(dg), i in eachnode(dg)
        for v in eachvariable(equations)
            antidiffusive_flux2_L[v, i, j, element] = fhat2_L[v, i, j] -
                                                      fstar2_L[v, i, j]
            antidiffusive_flux2_R[v, i, j, element] = fhat2_R[v, i, j] -
                                                      fstar2_R[v, i, j]
        end
    end

    antidiffusive_flux1_L[:, 1, :, element] .= zero(eltype(antidiffusive_flux1_L))
    antidiffusive_flux1_L[:, nnodes(dg) + 1, :, element] .= zero(eltype(antidiffusive_flux1_L))
    antidiffusive_flux1_R[:, 1, :, element] .= zero(eltype(antidiffusive_flux1_R))
    antidiffusive_flux1_R[:, nnodes(dg) + 1, :, element] .= zero(eltype(antidiffusive_flux1_R))

    antidiffusive_flux2_L[:, :, 1, element] .= zero(eltype(antidiffusive_flux2_L))
    antidiffusive_flux2_L[:, :, nnodes(dg) + 1, element] .= zero(eltype(antidiffusive_flux2_L))
    antidiffusive_flux2_R[:, :, 1, element] .= zero(eltype(antidiffusive_flux2_R))
    antidiffusive_flux2_R[:, :, nnodes(dg) + 1, element] .= zero(eltype(antidiffusive_flux2_R))

    return nothing
end

@inline function calcflux_antidiffusive!(fhat1_L, fhat1_R, fhat2_L, fhat2_R,
                                         fstar1_L, fstar1_R, fstar2_L, fstar2_R,
                                         u, mesh,
                                         nonconservative_terms::False, equations,
                                         limiter::SubcellLimiterMCL, dg, element, cache)
    (; antidiffusive_flux1_L, antidiffusive_flux2_L, antidiffusive_flux1_R, antidiffusive_flux2_R) = cache.antidiffusive_fluxes

    for j in eachnode(dg), i in 2:nnodes(dg)
        for v in eachvariable(equations)
            antidiffusive_flux1_L[v, i, j, element] = -(fhat1_L[v, i, j] -
                                                        fstar1_L[v, i, j])
            antidiffusive_flux1_R[v, i, j, element] = antidiffusive_flux1_L[v, i, j,
                                                                            element]
        end
    end
    for j in 2:nnodes(dg), i in eachnode(dg)
        for v in eachvariable(equations)
            antidiffusive_flux2_L[v, i, j, element] = -(fhat2_L[v, i, j] -
                                                        fstar2_L[v, i, j])
            antidiffusive_flux2_R[v, i, j, element] = antidiffusive_flux2_L[v, i, j,
                                                                            element]
        end
    end

    antidiffusive_flux1_L[:, 1, :, element] .= zero(eltype(antidiffusive_flux1_L))
    antidiffusive_flux1_L[:, nnodes(dg) + 1, :, element] .= zero(eltype(antidiffusive_flux1_L))
    antidiffusive_flux1_R[:, 1, :, element] .= zero(eltype(antidiffusive_flux1_R))
    antidiffusive_flux1_R[:, nnodes(dg) + 1, :, element] .= zero(eltype(antidiffusive_flux1_R))

    antidiffusive_flux2_L[:, :, 1, element] .= zero(eltype(antidiffusive_flux2_L))
    antidiffusive_flux2_L[:, :, nnodes(dg) + 1, element] .= zero(eltype(antidiffusive_flux2_L))
    antidiffusive_flux2_R[:, :, 1, element] .= zero(eltype(antidiffusive_flux2_R))
    antidiffusive_flux2_R[:, :, nnodes(dg) + 1, element] .= zero(eltype(antidiffusive_flux2_R))

    return nothing
end

@inline function calcflux_antidiffusive!(fhat1_L, fhat1_R, fhat2_L, fhat2_R,
                                         fstar1_L, fstar1_R, fstar2_L, fstar2_R,
                                         u, mesh,
                                         nonconservative_terms::True, equations,
                                         limiter::SubcellLimiterMCL, dg, element, cache)
    (; antidiffusive_flux1_L, antidiffusive_flux2_L, antidiffusive_flux1_R, antidiffusive_flux2_R) = cache.antidiffusive_fluxes

    for j in eachnode(dg), i in 2:nnodes(dg)
        for v in eachvariable(equations)
            antidiffusive_flux1_L[v, i, j, element] = -(fhat1_L[v, i, j] -
                                                        fstar1_L[v, i, j])
            antidiffusive_flux1_R[v, i, j, element] = -(fhat1_R[v, i, j] -
                                                        fstar1_R[v, i, j])
        end
    end
    for j in 2:nnodes(dg), i in eachnode(dg)
        for v in eachvariable(equations)
            antidiffusive_flux2_L[v, i, j, element] = -(fhat2_L[v, i, j] -
                                                        fstar2_L[v, i, j])
            antidiffusive_flux2_R[v, i, j, element] = -(fhat2_R[v, i, j] -
                                                        fstar2_R[v, i, j])
        end
    end

    antidiffusive_flux1_L[:, 1, :, element] .= zero(eltype(antidiffusive_flux1_L))
    antidiffusive_flux1_L[:, nnodes(dg) + 1, :, element] .= zero(eltype(antidiffusive_flux1_L))
    antidiffusive_flux1_R[:, 1, :, element] .= zero(eltype(antidiffusive_flux1_R))
    antidiffusive_flux1_R[:, nnodes(dg) + 1, :, element] .= zero(eltype(antidiffusive_flux1_R))

    antidiffusive_flux2_L[:, :, 1, element] .= zero(eltype(antidiffusive_flux2_L))
    antidiffusive_flux2_L[:, :, nnodes(dg) + 1, element] .= zero(eltype(antidiffusive_flux2_L))
    antidiffusive_flux2_R[:, :, 1, element] .= zero(eltype(antidiffusive_flux2_R))
    antidiffusive_flux2_R[:, :, nnodes(dg) + 1, element] .= zero(eltype(antidiffusive_flux2_R))

    return nothing
end

@inline function calc_lambdas_bar_states!(u, t, mesh::TreeMesh{2},
                                          nonconservative_terms, equations, limiter,
                                          dg, cache, boundary_conditions;
                                          calc_bar_states = true)
    if limiter isa SubcellLimiterIDP && !limiter.bar_states
        return nothing
    end
    (; lambda1, lambda2, bar_states1, bar_states2) = limiter.cache.container_bar_states

    # Calc lambdas and bar states inside elements
    @threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in 2:nnodes(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            u_node_im1 = get_node_vars(u, equations, dg, i - 1, j, element)
            lambda1[i, j, element] = max_abs_speed_naive(u_node_im1, u_node, 1,
                                                         equations)

            !calc_bar_states && continue

            flux1 = flux(u_node, 1, equations)
            flux1_im1 = flux(u_node_im1, 1, equations)
            for v in eachvariable(equations)
                bar_states1[v, i, j, element] = 0.5 * (u_node[v] + u_node_im1[v]) -
                                                0.5 * (flux1[v] - flux1_im1[v]) /
                                                lambda1[i, j, element]
            end
        end

        for j in 2:nnodes(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            u_node_jm1 = get_node_vars(u, equations, dg, i, j - 1, element)
            lambda2[i, j, element] = max_abs_speed_naive(u_node_jm1, u_node, 2,
                                                         equations)

            !calc_bar_states && continue

            flux2 = flux(u_node, 2, equations)
            flux2_jm1 = flux(u_node_jm1, 2, equations)
            for v in eachvariable(equations)
                bar_states2[v, i, j, element] = 0.5 * (u_node[v] + u_node_jm1[v]) -
                                                0.5 * (flux2[v] - flux2_jm1[v]) /
                                                lambda2[i, j, element]
            end
        end
    end

    # Calc lambdas and bar states at interfaces and periodic boundaries
    @threaded for interface in eachinterface(dg, cache)
        # Get neighboring element ids
        left_id = cache.interfaces.neighbor_ids[1, interface]
        right_id = cache.interfaces.neighbor_ids[2, interface]

        orientation = cache.interfaces.orientations[interface]

        if orientation == 1
            for j in eachnode(dg)
                u_left = get_node_vars(u, equations, dg, nnodes(dg), j, left_id)
                u_right = get_node_vars(u, equations, dg, 1, j, right_id)
                lambda = max_abs_speed_naive(u_left, u_right, orientation, equations)

                lambda1[nnodes(dg) + 1, j, left_id] = lambda
                lambda1[1, j, right_id] = lambda

                !calc_bar_states && continue

                flux_left = flux(u_left, orientation, equations)
                flux_right = flux(u_right, orientation, equations)
                bar_state = 0.5 * (u_left + u_right) -
                            0.5 * (flux_right - flux_left) / lambda
                for v in eachvariable(equations)
                    bar_states1[v, nnodes(dg) + 1, j, left_id] = bar_state[v]
                    bar_states1[v, 1, j, right_id] = bar_state[v]
                end
            end
        else # orientation == 2
            for i in eachnode(dg)
                u_left = get_node_vars(u, equations, dg, i, nnodes(dg), left_id)
                u_right = get_node_vars(u, equations, dg, i, 1, right_id)
                lambda = max_abs_speed_naive(u_left, u_right, orientation, equations)

                lambda2[i, nnodes(dg) + 1, left_id] = lambda
                lambda2[i, 1, right_id] = lambda

                !calc_bar_states && continue

                flux_left = flux(u_left, orientation, equations)
                flux_right = flux(u_right, orientation, equations)
                bar_state = 0.5 * (u_left + u_right) -
                            0.5 * (flux_right - flux_left) / lambda
                for v in eachvariable(equations)
                    bar_states2[v, i, nnodes(dg) + 1, left_id] = bar_state[v]
                    bar_states2[v, i, 1, right_id] = bar_state[v]
                end
            end
        end
    end

    # Calc lambdas and bar states at physical boundaries
    @threaded for boundary in eachboundary(dg, cache)
        element = cache.boundaries.neighbor_ids[boundary]
        orientation = cache.boundaries.orientations[boundary]
        neighbor_side = cache.boundaries.neighbor_sides[boundary]

        if orientation == 1
            if neighbor_side == 2 # Element is on the right, boundary on the left
                for j in eachnode(dg)
                    u_inner = get_node_vars(u, equations, dg, 1, j, element)
                    u_outer = get_boundary_outer_state(u_inner, cache, t,
                                                       boundary_conditions[1],
                                                       orientation, 1,
                                                       mesh, equations, dg,
                                                       1, j, element)
                    lambda1[1, j, element] = max_abs_speed_naive(u_inner, u_outer,
                                                                 orientation, equations)

                    !calc_bar_states && continue

                    flux_inner = flux(u_inner, orientation, equations)
                    flux_outer = flux(u_outer, orientation, equations)
                    bar_state = 0.5 * (u_inner + u_outer) -
                                0.5 * (flux_inner - flux_outer) / lambda1[1, j, element]
                    for v in eachvariable(equations)
                        bar_states1[v, 1, j, element] = bar_state[v]
                    end
                end
            else # Element is on the left, boundary on the right
                for j in eachnode(dg)
                    u_inner = get_node_vars(u, equations, dg, nnodes(dg), j, element)
                    u_outer = get_boundary_outer_state(u_inner, cache, t,
                                                       boundary_conditions[2],
                                                       orientation, 2,
                                                       mesh, equations, dg,
                                                       nnodes(dg), j, element)
                    lambda1[nnodes(dg) + 1, j, element] = max_abs_speed_naive(u_inner,
                                                                              u_outer,
                                                                              orientation,
                                                                              equations)

                    !calc_bar_states && continue

                    flux_inner = flux(u_inner, orientation, equations)
                    flux_outer = flux(u_outer, orientation, equations)
                    bar_state = 0.5 * (u_inner + u_outer) -
                                0.5 * (flux_outer - flux_inner) /
                                lambda1[nnodes(dg) + 1, j, element]
                    for v in eachvariable(equations)
                        bar_states1[v, nnodes(dg) + 1, j, element] = bar_state[v]
                    end
                end
            end
        else # orientation == 2
            if neighbor_side == 2 # Element is on the right, boundary on the left
                for i in eachnode(dg)
                    u_inner = get_node_vars(u, equations, dg, i, 1, element)
                    u_outer = get_boundary_outer_state(u_inner, cache, t,
                                                       boundary_conditions[3],
                                                       orientation, 3,
                                                       mesh, equations, dg,
                                                       i, 1, element)
                    lambda2[i, 1, element] = max_abs_speed_naive(u_inner, u_outer,
                                                                 orientation, equations)

                    !calc_bar_states && continue

                    flux_inner = flux(u_inner, orientation, equations)
                    flux_outer = flux(u_outer, orientation, equations)
                    bar_state = 0.5 * (u_inner + u_outer) -
                                0.5 * (flux_inner - flux_outer) / lambda2[i, 1, element]
                    for v in eachvariable(equations)
                        bar_states2[v, i, 1, element] = bar_state[v]
                    end
                end
            else # Element is on the left, boundary on the right
                for i in eachnode(dg)
                    u_inner = get_node_vars(u, equations, dg, i, nnodes(dg), element)
                    u_outer = get_boundary_outer_state(u_inner, cache, t,
                                                       boundary_conditions[4],
                                                       orientation, 4,
                                                       mesh, equations, dg,
                                                       i, nnodes(dg), element)
                    lambda2[i, nnodes(dg) + 1, element] = max_abs_speed_naive(u_inner,
                                                                              u_outer,
                                                                              orientation,
                                                                              equations)

                    !calc_bar_states && continue

                    flux_inner = flux(u_inner, orientation, equations)
                    flux_outer = flux(u_outer, orientation, equations)
                    bar_state = 0.5 * (u_inner + u_outer) -
                                0.5 * (flux_outer - flux_inner) /
                                lambda2[i, nnodes(dg) + 1, element]
                    for v in eachvariable(equations)
                        bar_states2[v, i, nnodes(dg) + 1, element] = bar_state[v]
                    end
                end
            end
        end
    end

    return nothing
end

@inline function calc_variable_bounds!(u, mesh, nonconservative_terms, equations,
                                       limiter::SubcellLimiterIDP, dg, cache)
    if !limiter.bar_states
        return nothing
    end
    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    (; bar_states1, bar_states2) = limiter.cache.container_bar_states

    # Local two-sided limiting for conservative variables
    if limiter.local_twosided
        for v in limiter.local_twosided_variables_cons
            v_string = string(v)
            var_min = variable_bounds[Symbol(v_string, "_min")]
            var_max = variable_bounds[Symbol(v_string, "_max")]
            @threaded for element in eachelement(dg, cache)
                for j in eachnode(dg), i in eachnode(dg)
                    var_min[i, j, element] = typemax(eltype(var_min))
                    var_max[i, j, element] = typemin(eltype(var_max))
                end
                for j in eachnode(dg), i in eachnode(dg)
                    var_min[i, j, element] = min(var_min[i, j, element],
                                                 u[v, i, j, element])
                    var_max[i, j, element] = max(var_max[i, j, element],
                                                 u[v, i, j, element])
                    # TODO: Add source term!
                    # - xi direction
                    var_min[i, j, element] = min(var_min[i, j, element],
                                                 bar_states1[v, i, j, element])
                    var_max[i, j, element] = max(var_max[i, j, element],
                                                 bar_states1[v, i, j, element])
                    # + xi direction
                    var_min[i, j, element] = min(var_min[i, j, element],
                                                 bar_states1[v, i + 1, j, element])
                    var_max[i, j, element] = max(var_max[i, j, element],
                                                 bar_states1[v, i + 1, j, element])
                    # - eta direction
                    var_min[i, j, element] = min(var_min[i, j, element],
                                                 bar_states2[v, i, j, element])
                    var_max[i, j, element] = max(var_max[i, j, element],
                                                 bar_states2[v, i, j, element])
                    # + eta direction
                    var_min[i, j, element] = min(var_min[i, j, element],
                                                 bar_states2[v, i, j + 1, element])
                    var_max[i, j, element] = max(var_max[i, j, element],
                                                 bar_states2[v, i, j + 1, element])
                end
            end
        end
    end
    # Local two-sided limiting for non-linear variables
    if limiter.local_onesided
        for (variable, min_or_max) in limiter.local_onesided_variables_nonlinear
            var_minmax = variable_bounds[Symbol(string(variable), "_",
                                                string(min_or_max))]
            @threaded for element in eachelement(dg, cache)
                for j in eachnode(dg), i in eachnode(dg)
                    if min_or_max === max
                        var_minmax[i, j, element] = typemin(eltype(var_minmax))
                    else
                        var_minmax[i, j, element] = typemax(eltype(var_minmax))
                    end
                end
                # FV solution at node (i, j)
                for j in eachnode(dg), i in eachnode(dg)
                    var = variable(get_node_vars(u, equations, dg, i, j, element),
                                   equations)
                    var_minmax[i, j, element] = min_or_max(var_minmax[i, j, element],
                                                           var)
                    # TODO: Add source term!
                end
                # xi direction: subcell face between (i-1, j) and (i, j)
                for j in eachnode(dg), i in 1:(nnodes(dg) + 1)
                    var = variable(get_node_vars(bar_states1, equations, dg, i, j,
                                                 element), equations)
                    if i <= nnodes(dg)
                        var_minmax[i, j, element] = min_or_max(var_minmax[i, j,
                                                                          element], var)
                    end
                    if i > 1
                        var_minmax[i - 1, j, element] = min_or_max(var_minmax[i - 1, j,
                                                                              element],
                                                                   var)
                    end
                end
                # eta direction: subcell face between (i, j-1) and (i, j)
                for j in 1:(nnodes(dg) + 1), i in eachnode(dg)
                    var = variable(get_node_vars(bar_states2, equations, dg, i, j,
                                                 element), equations)
                    if j <= nnodes(dg)
                        var_minmax[i, j, element] = min_or_max(var_minmax[i, j,
                                                                          element], var)
                    end
                    if j > 1
                        var_minmax[i, j - 1, element] = min_or_max(var_minmax[i, j - 1,
                                                                              element],
                                                                   var)
                    end
                end
            end
        end
    end

    return nothing
end

@inline function calc_variable_bounds!(u, mesh, nonconservative_terms, equations,
                                       limiter::SubcellLimiterMCL, dg, cache)
    (; var_min, var_max) = limiter.cache.subcell_limiter_coefficients
    (; bar_states1, bar_states2) = limiter.cache.container_bar_states

    @threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg), v in eachvariable(equations)
            var_min[v, i, j, element] = typemax(eltype(var_min))
            var_max[v, i, j, element] = typemin(eltype(var_max))
        end

        if limiter.density_limiter
            for j in eachnode(dg), i in eachnode(dg)
                # Previous solution
                var_min[1, i, j, element] = min(var_min[1, i, j, element],
                                                u[1, i, j, element])
                var_max[1, i, j, element] = max(var_max[1, i, j, element],
                                                u[1, i, j, element])
                # - xi direction
                bar_state_rho = bar_states1[1, i, j, element]
                var_min[1, i, j, element] = min(var_min[1, i, j, element],
                                                bar_state_rho)
                var_max[1, i, j, element] = max(var_max[1, i, j, element],
                                                bar_state_rho)
                # + xi direction
                bar_state_rho = bar_states1[1, i + 1, j, element]
                var_min[1, i, j, element] = min(var_min[1, i, j, element],
                                                bar_state_rho)
                var_max[1, i, j, element] = max(var_max[1, i, j, element],
                                                bar_state_rho)
                # - eta direction
                bar_state_rho = bar_states2[1, i, j, element]
                var_min[1, i, j, element] = min(var_min[1, i, j, element],
                                                bar_state_rho)
                var_max[1, i, j, element] = max(var_max[1, i, j, element],
                                                bar_state_rho)
                # + eta direction
                bar_state_rho = bar_states2[1, i, j + 1, element]
                var_min[1, i, j, element] = min(var_min[1, i, j, element],
                                                bar_state_rho)
                var_max[1, i, j, element] = max(var_max[1, i, j, element],
                                                bar_state_rho)
            end
        end #limiter.density_limiter

        if limiter.sequential_limiter
            for j in eachnode(dg), i in eachnode(dg)
                # Previous solution
                for v in 2:nvariables(equations)
                    phi = u[v, i, j, element] / u[1, i, j, element]
                    var_min[v, i, j, element] = min(var_min[v, i, j, element], phi)
                    var_max[v, i, j, element] = max(var_max[v, i, j, element], phi)
                end
                # - xi direction
                bar_state_rho = bar_states1[1, i, j, element]
                for v in 2:nvariables(equations)
                    bar_state_phi = bar_states1[v, i, j, element] / bar_state_rho
                    var_min[v, i, j, element] = min(var_min[v, i, j, element],
                                                    bar_state_phi)
                    var_max[v, i, j, element] = max(var_max[v, i, j, element],
                                                    bar_state_phi)
                end
                # + xi direction
                bar_state_rho = bar_states1[1, i + 1, j, element]
                for v in 2:nvariables(equations)
                    bar_state_phi = bar_states1[v, i + 1, j, element] / bar_state_rho
                    var_min[v, i, j, element] = min(var_min[v, i, j, element],
                                                    bar_state_phi)
                    var_max[v, i, j, element] = max(var_max[v, i, j, element],
                                                    bar_state_phi)
                end
                # - eta direction
                bar_state_rho = bar_states2[1, i, j, element]
                for v in 2:nvariables(equations)
                    bar_state_phi = bar_states2[v, i, j, element] / bar_state_rho
                    var_min[v, i, j, element] = min(var_min[v, i, j, element],
                                                    bar_state_phi)
                    var_max[v, i, j, element] = max(var_max[v, i, j, element],
                                                    bar_state_phi)
                end
                # + eta direction
                bar_state_rho = bar_states2[1, i, j + 1, element]
                for v in 2:nvariables(equations)
                    bar_state_phi = bar_states2[v, i, j + 1, element] / bar_state_rho
                    var_min[v, i, j, element] = min(var_min[v, i, j, element],
                                                    bar_state_phi)
                    var_max[v, i, j, element] = max(var_max[v, i, j, element],
                                                    bar_state_phi)
                end
            end
        elseif limiter.conservative_limiter
            for j in eachnode(dg), i in eachnode(dg)
                # Previous solution
                for v in 2:nvariables(equations)
                    var_min[v, i, j, element] = min(var_min[v, i, j, element],
                                                    u[v, i, j, element])
                    var_max[v, i, j, element] = max(var_max[v, i, j, element],
                                                    u[v, i, j, element])
                end
                # - xi direction
                for v in 2:nvariables(equations)
                    bar_state_rho = bar_states1[v, i, j, element]
                    var_min[v, i, j, element] = min(var_min[v, i, j, element],
                                                    bar_state_rho)
                    var_max[v, i, j, element] = max(var_max[v, i, j, element],
                                                    bar_state_rho)
                end
                # + xi direction
                for v in 2:nvariables(equations)
                    bar_state_rho = bar_states1[v, i + 1, j, element]
                    var_min[v, i, j, element] = min(var_min[v, i, j, element],
                                                    bar_state_rho)
                    var_max[v, i, j, element] = max(var_max[v, i, j, element],
                                                    bar_state_rho)
                end
                # - eta direction
                for v in 2:nvariables(equations)
                    bar_state_rho = bar_states2[v, i, j, element]
                    var_min[v, i, j, element] = min(var_min[v, i, j, element],
                                                    bar_state_rho)
                    var_max[v, i, j, element] = max(var_max[v, i, j, element],
                                                    bar_state_rho)
                end
                # + eta direction
                for v in 2:nvariables(equations)
                    bar_state_rho = bar_states2[v, i, j + 1, element]
                    var_min[v, i, j, element] = min(var_min[v, i, j, element],
                                                    bar_state_rho)
                    var_max[v, i, j, element] = max(var_max[v, i, j, element],
                                                    bar_state_rho)
                end
            end
        end
    end

    return nothing
end

@inline function calcflux_antidiffusive_limited!(u, mesh, nonconservative_terms::False,
                                                 equations, limiter, dg, element,
                                                 cache,
                                                 fstar1, fstar2)
    (; antidiffusive_flux1_L, antidiffusive_flux2_L, antidiffusive_flux1_R, antidiffusive_flux2_R) = cache.antidiffusive_fluxes
    (; var_min, var_max) = limiter.cache.subcell_limiter_coefficients
    (; bar_states1, bar_states2, lambda1, lambda2) = limiter.cache.container_bar_states

    if limiter.Plotting
        (; alpha, alpha_pressure, alpha_entropy, alpha_mean,
        alpha_mean_pressure, alpha_mean_entropy) = limiter.cache.subcell_limiter_coefficients
        for j in eachnode(dg), i in eachnode(dg)
            alpha_mean[:, i, j, element] .= zero(eltype(alpha_mean))
            alpha[:, i, j, element] .= one(eltype(alpha))
            if limiter.positivity_limiter_pressure
                alpha_mean_pressure[i, j, element] = zero(eltype(alpha_mean_pressure))
                alpha_pressure[i, j, element] = one(eltype(alpha_pressure))
            end
            if limiter.entropy_limiter_semidiscrete
                alpha_mean_entropy[i, j, element] = zero(eltype(alpha_mean_entropy))
                alpha_entropy[i, j, element] = one(eltype(alpha_entropy))
            end
        end
    end

    # The antidiffuse flux can have very small absolute values. This can lead to values of f_min which are zero up to machine accuracy.
    # To avoid further calculations with these values, we replace them by 0.
    # It can also happen that the limited flux changes its sign (for instance to -1e-13).
    # This does not really make sense in theory and causes problems for the visualization.
    # Therefore we make sure that the flux keeps its sign during limiting.

    # Density limiter
    if limiter.density_limiter
        for j in eachnode(dg), i in 2:nnodes(dg)
            lambda = lambda1[i, j, element]
            bar_state_rho = bar_states1[1, i, j, element]

            # Limit density
            if antidiffusive_flux1_L[1, i, j, element] > 0
                f_max = lambda * min(var_max[1, i - 1, j, element] - bar_state_rho,
                            bar_state_rho - var_min[1, i, j, element])
                f_max = isapprox(f_max, 0.0, atol = eps()) ? 0.0 : f_max
                flux_limited = min(antidiffusive_flux1_L[1, i, j, element],
                                   max(f_max, 0.0))
            else
                f_min = lambda * max(var_min[1, i - 1, j, element] - bar_state_rho,
                            bar_state_rho - var_max[1, i, j, element])
                f_min = isapprox(f_min, 0.0, atol = eps()) ? 0.0 : f_min
                flux_limited = max(antidiffusive_flux1_L[1, i, j, element],
                                   min(f_min, 0.0))
            end

            if limiter.Plotting || limiter.density_coefficient_for_all
                if isapprox(antidiffusive_flux1_L[1, i, j, element], 0.0, atol = eps())
                    coefficient = 1.0 # flux_limited is zero as well
                else
                    coefficient = min(1,
                                      (flux_limited + sign(flux_limited) * eps()) /
                                      (antidiffusive_flux1_L[1, i, j, element] +
                                       sign(flux_limited) * eps()))
                end

                if limiter.Plotting
                    (; alpha, alpha_mean) = limiter.cache.subcell_limiter_coefficients
                    alpha[1, i - 1, j, element] = min(alpha[1, i - 1, j, element],
                                                      coefficient)
                    alpha[1, i, j, element] = min(alpha[1, i, j, element], coefficient)
                    alpha_mean[1, i - 1, j, element] += coefficient
                    alpha_mean[1, i, j, element] += coefficient
                end
            end
            antidiffusive_flux1_L[1, i, j, element] = flux_limited

            #Limit all quantities with the same alpha
            if limiter.density_coefficient_for_all
                for v in 2:nvariables(equations)
                    antidiffusive_flux1_L[v, i, j, element] = coefficient *
                                                              antidiffusive_flux1_L[v,
                                                                                    i,
                                                                                    j,
                                                                                    element]
                end
            end
        end

        for j in 2:nnodes(dg), i in eachnode(dg)
            lambda = lambda2[i, j, element]
            bar_state_rho = bar_states2[1, i, j, element]

            # Limit density
            if antidiffusive_flux2_L[1, i, j, element] > 0
                f_max = lambda * min(var_max[1, i, j - 1, element] - bar_state_rho,
                            bar_state_rho - var_min[1, i, j, element])
                f_max = isapprox(f_max, 0.0, atol = eps()) ? 0.0 : f_max
                flux_limited = min(antidiffusive_flux2_L[1, i, j, element],
                                   max(f_max, 0.0))
            else
                f_min = lambda * max(var_min[1, i, j - 1, element] - bar_state_rho,
                            bar_state_rho - var_max[1, i, j, element])
                f_min = isapprox(f_min, 0.0, atol = eps()) ? 0.0 : f_min
                flux_limited = max(antidiffusive_flux2_L[1, i, j, element],
                                   min(f_min, 0.0))
            end

            if limiter.Plotting || limiter.density_coefficient_for_all
                if isapprox(antidiffusive_flux2_L[1, i, j, element], 0.0, atol = eps())
                    coefficient = 1.0 # flux_limited is zero as well
                else
                    coefficient = min(1,
                                      (flux_limited + sign(flux_limited) * eps()) /
                                      (antidiffusive_flux2_L[1, i, j, element] +
                                       sign(flux_limited) * eps()))
                end

                if limiter.Plotting
                    (; alpha, alpha_mean) = limiter.cache.subcell_limiter_coefficients
                    alpha[1, i, j - 1, element] = min(alpha[1, i, j - 1, element],
                                                      coefficient)
                    alpha[1, i, j, element] = min(alpha[1, i, j, element], coefficient)
                    alpha_mean[1, i, j - 1, element] += coefficient
                    alpha_mean[1, i, j, element] += coefficient
                end
            end
            antidiffusive_flux2_L[1, i, j, element] = flux_limited

            #Limit all quantities with the same alpha
            if limiter.density_coefficient_for_all
                for v in 2:nvariables(equations)
                    antidiffusive_flux2_L[v, i, j, element] = coefficient *
                                                              antidiffusive_flux2_L[v,
                                                                                    i,
                                                                                    j,
                                                                                    element]
                end
            end
        end
    end # if limiter.density_limiter

    # Sequential limiter
    if limiter.sequential_limiter
        for j in eachnode(dg), i in 2:nnodes(dg)
            lambda = lambda1[i, j, element]
            bar_state_rho = bar_states1[1, i, j, element]

            # Limit velocity and total energy
            rho_limited_iim1 = lambda * bar_state_rho -
                               antidiffusive_flux1_L[1, i, j, element]
            rho_limited_im1i = lambda * bar_state_rho +
                               antidiffusive_flux1_L[1, i, j, element]
            for v in 2:nvariables(equations)
                bar_state_phi = bar_states1[v, i, j, element]

                phi = bar_state_phi / bar_state_rho

                g = antidiffusive_flux1_L[v, i, j, element] +
                    (lambda * bar_state_phi - rho_limited_im1i * phi)

                if g > 0
                    g_max = min(rho_limited_im1i *
                                (var_max[v, i - 1, j, element] - phi),
                                rho_limited_iim1 * (phi - var_min[v, i, j, element]))
                    g_max = isapprox(g_max, 0.0, atol = eps()) ? 0.0 : g_max
                    g_limited = min(g, max(g_max, 0.0))
                else
                    g_min = max(rho_limited_im1i *
                                (var_min[v, i - 1, j, element] - phi),
                                rho_limited_iim1 * (phi - var_max[v, i, j, element]))
                    g_min = isapprox(g_min, 0.0, atol = eps()) ? 0.0 : g_min
                    g_limited = max(g, min(g_min, 0.0))
                end
                if limiter.Plotting
                    if isapprox(g, 0.0, atol = eps())
                        coefficient = 1.0 # g_limited is zero as well
                    else
                        coefficient = min(1,
                                          (g_limited + sign(g_limited) * eps()) /
                                          (g + sign(g_limited) * eps()))
                    end
                    (; alpha, alpha_mean) = limiter.cache.subcell_limiter_coefficients
                    alpha[v, i - 1, j, element] = min(alpha[v, i - 1, j, element],
                                                      coefficient)
                    alpha[v, i, j, element] = min(alpha[v, i, j, element], coefficient)
                    alpha_mean[v, i - 1, j, element] += coefficient
                    alpha_mean[v, i, j, element] += coefficient
                end
                antidiffusive_flux1_L[v, i, j, element] = (rho_limited_im1i * phi -
                                                           lambda * bar_state_phi) +
                                                          g_limited
            end
        end

        for j in 2:nnodes(dg), i in eachnode(dg)
            lambda = lambda2[i, j, element]
            bar_state_rho = bar_states2[1, i, j, element]

            # Limit velocity and total energy
            rho_limited_jjm1 = lambda * bar_state_rho -
                               antidiffusive_flux2_L[1, i, j, element]
            rho_limited_jm1j = lambda * bar_state_rho +
                               antidiffusive_flux2_L[1, i, j, element]
            for v in 2:nvariables(equations)
                bar_state_phi = bar_states2[v, i, j, element]

                phi = bar_state_phi / bar_state_rho

                g = antidiffusive_flux2_L[v, i, j, element] +
                    (lambda * bar_state_phi - rho_limited_jm1j * phi)

                if g > 0
                    g_max = min(rho_limited_jm1j *
                                (var_max[v, i, j - 1, element] - phi),
                                rho_limited_jjm1 * (phi - var_min[v, i, j, element]))
                    g_max = isapprox(g_max, 0.0, atol = eps()) ? 0.0 : g_max
                    g_limited = min(g, max(g_max, 0.0))
                else
                    g_min = max(rho_limited_jm1j *
                                (var_min[v, i, j - 1, element] - phi),
                                rho_limited_jjm1 * (phi - var_max[v, i, j, element]))
                    g_min = isapprox(g_min, 0.0, atol = eps()) ? 0.0 : g_min
                    g_limited = max(g, min(g_min, 0.0))
                end
                if limiter.Plotting
                    if isapprox(g, 0.0, atol = eps())
                        coefficient = 1.0 # g_limited is zero as well
                    else
                        coefficient = min(1,
                                          (g_limited + sign(g_limited) * eps()) /
                                          (g + sign(g_limited) * eps()))
                    end
                    (; alpha, alpha_mean) = limiter.cache.subcell_limiter_coefficients
                    alpha[v, i, j - 1, element] = min(alpha[v, i, j - 1, element],
                                                      coefficient)
                    alpha[v, i, j, element] = min(alpha[v, i, j, element], coefficient)
                    alpha_mean[v, i, j - 1, element] += coefficient
                    alpha_mean[v, i, j, element] += coefficient
                end

                antidiffusive_flux2_L[v, i, j, element] = (rho_limited_jm1j * phi -
                                                           lambda * bar_state_phi) +
                                                          g_limited
            end
        end
        # Conservative limiter
    elseif limiter.conservative_limiter
        for j in eachnode(dg), i in 2:nnodes(dg)
            lambda = lambda1[i, j, element]
            for v in 2:nvariables(equations)
                bar_state_phi = bar_states1[v, i, j, element]
                # Limit density
                if antidiffusive_flux1_L[v, i, j, element] > 0
                    f_max = lambda * min(var_max[v, i - 1, j, element] - bar_state_phi,
                                bar_state_phi - var_min[v, i, j, element])
                    f_max = isapprox(f_max, 0.0, atol = eps()) ? 0.0 : f_max
                    flux_limited = min(antidiffusive_flux1_L[v, i, j, element],
                                       max(f_max, 0.0))
                else
                    f_min = lambda * max(var_min[v, i - 1, j, element] - bar_state_phi,
                                bar_state_phi - var_max[v, i, j, element])
                    f_min = isapprox(f_min, 0.0, atol = eps()) ? 0.0 : f_min
                    flux_limited = max(antidiffusive_flux1_L[v, i, j, element],
                                       min(f_min, 0.0))
                end

                if limiter.Plotting
                    if isapprox(antidiffusive_flux1_L[v, i, j, element], 0.0,
                                atol = eps())
                        coefficient = 1.0 # flux_limited is zero as well
                    else
                        coefficient = min(1,
                                          (flux_limited + sign(flux_limited) * eps()) /
                                          (antidiffusive_flux1_L[v, i, j, element] +
                                           sign(flux_limited) * eps()))
                    end
                    (; alpha, alpha_mean) = limiter.cache.subcell_limiter_coefficients
                    alpha[v, i - 1, j, element] = min(alpha[v, i - 1, j, element],
                                                      coefficient)
                    alpha[v, i, j, element] = min(alpha[v, i, j, element], coefficient)
                    alpha_mean[v, i - 1, j, element] += coefficient
                    alpha_mean[v, i, j, element] += coefficient
                end
                antidiffusive_flux1_L[v, i, j, element] = flux_limited
            end
        end

        for j in 2:nnodes(dg), i in eachnode(dg)
            lambda = lambda2[i, j, element]
            for v in 2:nvariables(equations)
                bar_state_phi = bar_states2[v, i, j, element]
                # Limit density
                if antidiffusive_flux2_L[v, i, j, element] > 0
                    f_max = lambda * min(var_max[v, i, j - 1, element] - bar_state_phi,
                                bar_state_phi - var_min[v, i, j, element])
                    f_max = isapprox(f_max, 0.0, atol = eps()) ? 0.0 : f_max
                    flux_limited = min(antidiffusive_flux2_L[v, i, j, element],
                                       max(f_max, 0.0))
                else
                    f_min = lambda * max(var_min[v, i, j - 1, element] - bar_state_phi,
                                bar_state_phi - var_max[v, i, j, element])
                    f_min = isapprox(f_min, 0.0, atol = eps()) ? 0.0 : f_min
                    flux_limited = max(antidiffusive_flux2_L[v, i, j, element],
                                       min(f_min, 0.0))
                end

                if limiter.Plotting
                    if isapprox(antidiffusive_flux2_L[v, i, j, element], 0.0,
                                atol = eps())
                        coefficient = 1.0 # flux_limited is zero as well
                    else
                        coefficient = min(1,
                                          (flux_limited + sign(flux_limited) * eps()) /
                                          (antidiffusive_flux2_L[v, i, j, element] +
                                           sign(flux_limited) * eps()))
                    end
                    (; alpha, alpha_mean) = limiter.cache.subcell_limiter_coefficients
                    alpha[v, i, j - 1, element] = min(alpha[v, i, j - 1, element],
                                                      coefficient)
                    alpha[v, i, j, element] = min(alpha[v, i, j, element], coefficient)
                    alpha_mean[v, i, j - 1, element] += coefficient
                    alpha_mean[v, i, j, element] += coefficient
                end
                antidiffusive_flux2_L[v, i, j, element] = flux_limited
            end
        end
    end # limiter.sequential_limiter and limiter.conservative_limiter

    # Density positivity limiter
    if limiter.positivity_limiter_density
        beta = limiter.positivity_limiter_correction_factor
        for j in eachnode(dg), i in 2:nnodes(dg)
            lambda = lambda1[i, j, element]
            bar_state_rho = bar_states1[1, i, j, element]
            # Limit density
            if antidiffusive_flux1_L[1, i, j, element] > 0
                f_max = (1 - beta) * lambda * bar_state_rho
                f_max = isapprox(f_max, 0.0, atol = eps()) ? 0.0 : f_max
                flux_limited = min(antidiffusive_flux1_L[1, i, j, element],
                                   max(f_max, 0.0))
            else
                f_min = -(1 - beta) * lambda * bar_state_rho
                f_min = isapprox(f_min, 0.0, atol = eps()) ? 0.0 : f_min
                flux_limited = max(antidiffusive_flux1_L[1, i, j, element],
                                   min(f_min, 0.0))
            end

            if limiter.Plotting || limiter.density_coefficient_for_all
                if isapprox(antidiffusive_flux1_L[1, i, j, element], 0.0, atol = eps())
                    coefficient = 1.0  # flux_limited is zero as well
                else
                    coefficient = flux_limited / antidiffusive_flux1_L[1, i, j, element]
                end

                if limiter.Plotting
                    (; alpha, alpha_mean) = limiter.cache.subcell_limiter_coefficients
                    alpha[1, i - 1, j, element] = min(alpha[1, i - 1, j, element],
                                                      coefficient)
                    alpha[1, i, j, element] = min(alpha[1, i, j, element], coefficient)
                    if !limiter.density_limiter
                        alpha_mean[1, i - 1, j, element] += coefficient
                        alpha_mean[1, i, j, element] += coefficient
                    end
                end
            end
            antidiffusive_flux1_L[1, i, j, element] = flux_limited

            #Limit all quantities with the same alpha
            if limiter.density_coefficient_for_all
                for v in 2:nvariables(equations)
                    antidiffusive_flux1_L[v, i, j, element] = coefficient *
                                                              antidiffusive_flux1_L[v,
                                                                                    i,
                                                                                    j,
                                                                                    element]
                end
            end
        end

        for j in 2:nnodes(dg), i in eachnode(dg)
            lambda = lambda2[i, j, element]
            bar_state_rho = bar_states2[1, i, j, element]
            # Limit density
            if antidiffusive_flux2_L[1, i, j, element] > 0
                f_max = (1 - beta) * lambda * bar_state_rho
                f_max = isapprox(f_max, 0.0, atol = eps()) ? 0.0 : f_max
                flux_limited = min(antidiffusive_flux2_L[1, i, j, element],
                                   max(f_max, 0.0))
            else
                f_min = -(1 - beta) * lambda * bar_state_rho
                f_min = isapprox(f_min, 0.0, atol = eps()) ? 0.0 : f_min
                flux_limited = max(antidiffusive_flux2_L[1, i, j, element],
                                   min(f_min, 0.0))
            end

            if limiter.Plotting || limiter.density_coefficient_for_all
                if isapprox(antidiffusive_flux2_L[1, i, j, element], 0.0, atol = eps())
                    coefficient = 1.0  # flux_limited is zero as well
                else
                    coefficient = flux_limited / antidiffusive_flux2_L[1, i, j, element]
                end

                if limiter.Plotting
                    (; alpha, alpha_mean) = limiter.cache.subcell_limiter_coefficients
                    alpha[1, i, j - 1, element] = min(alpha[1, i, j - 1, element],
                                                      coefficient)
                    alpha[1, i, j, element] = min(alpha[1, i, j, element], coefficient)
                    if !limiter.density_limiter
                        alpha_mean[1, i, j - 1, element] += coefficient
                        alpha_mean[1, i, j, element] += coefficient
                    end
                end
            end
            antidiffusive_flux2_L[1, i, j, element] = flux_limited

            #Limit all quantities with the same alpha
            if limiter.density_coefficient_for_all
                for v in 2:nvariables(equations)
                    antidiffusive_flux2_L[v, i, j, element] = coefficient *
                                                              antidiffusive_flux2_L[v,
                                                                                    i,
                                                                                    j,
                                                                                    element]
                end
            end
        end
    end #if limiter.positivity_limiter_density

    # Divide alpha_mean by number of additions
    if limiter.Plotting
        (; alpha_mean) = limiter.cache.subcell_limiter_coefficients
        # Interfaces contribute with 1.0
        if limiter.density_limiter || limiter.positivity_limiter_density
            for i in eachnode(dg)
                alpha_mean[1, i, 1, element] += 1.0
                alpha_mean[1, i, nnodes(dg), element] += 1.0
                alpha_mean[1, 1, i, element] += 1.0
                alpha_mean[1, nnodes(dg), i, element] += 1.0
            end
            for j in eachnode(dg), i in eachnode(dg)
                alpha_mean[1, i, j, element] /= 4
            end
        end
        if limiter.sequential_limiter || limiter.conservative_limiter
            for v in 2:nvariables(equations)
                for i in eachnode(dg)
                    alpha_mean[v, i, 1, element] += 1.0
                    alpha_mean[v, i, nnodes(dg), element] += 1.0
                    alpha_mean[v, 1, i, element] += 1.0
                    alpha_mean[v, nnodes(dg), i, element] += 1.0
                end
                for j in eachnode(dg), i in eachnode(dg)
                    alpha_mean[v, i, j, element] /= 4
                end
            end
        end
    end

    # Limit pressure à la Kuzmin
    if limiter.positivity_limiter_pressure
        (; alpha_pressure, alpha_mean_pressure) = limiter.cache.subcell_limiter_coefficients
        for j in eachnode(dg), i in 2:nnodes(dg)
            bar_state_velocity = bar_states1[2, i, j, element]^2 +
                                 bar_states1[3, i, j, element]^2
            flux_velocity = antidiffusive_flux1_L[2, i, j, element]^2 +
                            antidiffusive_flux1_L[3, i, j, element]^2

            Q = lambda1[i, j, element]^2 *
                (bar_states1[1, i, j, element] * bar_states1[4, i, j, element] -
                 0.5 * bar_state_velocity)

            if limiter.positivity_limiter_pressure_exact
                # exact calculation of max(R_ij, R_ji)
                R_max = lambda1[i, j, element] *
                        abs(bar_states1[2, i, j, element] *
                            antidiffusive_flux1_L[2, i, j, element] +
                            bar_states1[3, i, j, element] *
                            antidiffusive_flux1_L[3, i, j, element] -
                            bar_states1[1, i, j, element] *
                            antidiffusive_flux1_L[4, i, j, element] -
                            bar_states1[4, i, j, element] *
                            antidiffusive_flux1_L[1, i, j, element])
                R_max += max(0,
                             0.5 * flux_velocity -
                             antidiffusive_flux1_L[4, i, j, element] *
                             antidiffusive_flux1_L[1, i, j, element])
            else
                # approximation R_max
                R_max = lambda1[i, j, element] *
                        (sqrt(bar_state_velocity * flux_velocity) +
                         abs(bar_states1[1, i, j, element] *
                             antidiffusive_flux1_L[4, i, j, element]) +
                         abs(bar_states1[4, i, j, element] *
                             antidiffusive_flux1_L[1, i, j, element]))
                R_max += max(0,
                             0.5 * flux_velocity -
                             antidiffusive_flux1_L[4, i, j, element] *
                             antidiffusive_flux1_L[1, i, j, element])
            end
            alpha = 1 # Initialize alpha for plotting
            if R_max > Q
                alpha = Q / R_max
                for v in eachvariable(equations)
                    antidiffusive_flux1_L[v, i, j, element] *= alpha
                end
            end
            if limiter.Plotting
                alpha_pressure[i - 1, j, element] = min(alpha_pressure[i - 1, j,
                                                                       element], alpha)
                alpha_pressure[i, j, element] = min(alpha_pressure[i, j, element],
                                                    alpha)
                alpha_mean_pressure[i - 1, j, element] += alpha
                alpha_mean_pressure[i, j, element] += alpha
            end
        end

        for j in 2:nnodes(dg), i in eachnode(dg)
            bar_state_velocity = bar_states2[2, i, j, element]^2 +
                                 bar_states2[3, i, j, element]^2
            flux_velocity = antidiffusive_flux2_L[2, i, j, element]^2 +
                            antidiffusive_flux2_L[3, i, j, element]^2

            Q = lambda2[i, j, element]^2 *
                (bar_states2[1, i, j, element] * bar_states2[4, i, j, element] -
                 0.5 * bar_state_velocity)

            if limiter.positivity_limiter_pressure_exact
                # exact calculation of max(R_ij, R_ji)
                R_max = lambda2[i, j, element] *
                        abs(bar_states2[2, i, j, element] *
                            antidiffusive_flux2_L[2, i, j, element] +
                            bar_states2[3, i, j, element] *
                            antidiffusive_flux2_L[3, i, j, element] -
                            bar_states2[1, i, j, element] *
                            antidiffusive_flux2_L[4, i, j, element] -
                            bar_states2[4, i, j, element] *
                            antidiffusive_flux2_L[1, i, j, element])
                R_max += max(0,
                             0.5 * flux_velocity -
                             antidiffusive_flux2_L[4, i, j, element] *
                             antidiffusive_flux2_L[1, i, j, element])
            else
                # approximation R_max
                R_max = lambda2[i, j, element] *
                        (sqrt(bar_state_velocity * flux_velocity) +
                         abs(bar_states2[1, i, j, element] *
                             antidiffusive_flux2_L[4, i, j, element]) +
                         abs(bar_states2[4, i, j, element] *
                             antidiffusive_flux2_L[1, i, j, element]))
                R_max += max(0,
                             0.5 * flux_velocity -
                             antidiffusive_flux2_L[4, i, j, element] *
                             antidiffusive_flux2_L[1, i, j, element])
            end
            alpha = 1 # Initialize alpha for plotting
            if R_max > Q
                alpha = Q / R_max
                for v in eachvariable(equations)
                    antidiffusive_flux2_L[v, i, j, element] *= alpha
                end
            end
            if limiter.Plotting
                alpha_pressure[i, j - 1, element] = min(alpha_pressure[i, j - 1,
                                                                       element], alpha)
                alpha_pressure[i, j, element] = min(alpha_pressure[i, j, element],
                                                    alpha)
                alpha_mean_pressure[i, j - 1, element] += alpha
                alpha_mean_pressure[i, j, element] += alpha
            end
        end
        if limiter.Plotting
            (; alpha_mean_pressure) = limiter.cache.subcell_limiter_coefficients
            # Interfaces contribute with 1.0
            for i in eachnode(dg)
                alpha_mean_pressure[i, 1, element] += 1.0
                alpha_mean_pressure[i, nnodes(dg), element] += 1.0
                alpha_mean_pressure[1, i, element] += 1.0
                alpha_mean_pressure[nnodes(dg), i, element] += 1.0
            end
            for j in eachnode(dg), i in eachnode(dg)
                alpha_mean_pressure[i, j, element] /= 4
            end
        end
    end

    # Limit entropy
    # TODO: This is a very inefficient function. We compute the entropy four times at each node.
    # TODO: For now, this only works for Cartesian meshes.
    if limiter.entropy_limiter_semidiscrete
        for j in eachnode(dg), i in 2:nnodes(dg)
            antidiffusive_flux_local = get_node_vars(antidiffusive_flux1_L, equations,
                                                     dg,
                                                     i, j, element)
            u_local = get_node_vars(u, equations, dg, i, j, element)
            u_local_m1 = get_node_vars(u, equations, dg, i - 1, j, element)

            # Using mathematic entropy
            v_local = cons2entropy(u_local, equations)
            v_local_m1 = cons2entropy(u_local_m1, equations)

            q_local = u_local[2] / u_local[1] * entropy(u_local, equations)
            q_local_m1 = u_local_m1[2] / u_local_m1[1] * entropy(u_local_m1, equations)

            f_local = flux(u_local, 1, equations)
            f_local_m1 = flux(u_local_m1, 1, equations)

            psi_local = dot(v_local, f_local) - q_local
            psi_local_m1 = dot(v_local_m1, f_local_m1) - q_local_m1

            delta_v = v_local - v_local_m1
            delta_psi = psi_local - psi_local_m1

            entProd_FV = dot(delta_v, view(fstar1, :, i, j)) - delta_psi
            delta_entProd = dot(delta_v, antidiffusive_flux_local)

            alpha = 1 # Initialize alpha for plotting
            if (entProd_FV + delta_entProd > 0.0) && (delta_entProd != 0.0)
                alpha = min(1.0,
                            (abs(entProd_FV) + eps()) / (abs(delta_entProd) + eps()))
                for v in eachvariable(equations)
                    antidiffusive_flux1_L[v, i, j, element] = alpha *
                                                              antidiffusive_flux1_L[v,
                                                                                    i,
                                                                                    j,
                                                                                    element]
                end
            end
            if limiter.Plotting
                (; alpha_entropy, alpha_mean_entropy) = limiter.cache.subcell_limiter_coefficients
                alpha_entropy[i - 1, j, element] = min(alpha_entropy[i - 1, j, element],
                                                       alpha)
                alpha_entropy[i, j, element] = min(alpha_entropy[i, j, element], alpha)
                alpha_mean_entropy[i - 1, j, element] += alpha
                alpha_mean_entropy[i, j, element] += alpha
            end
        end

        for j in 2:nnodes(dg), i in eachnode(dg)
            antidiffusive_flux_local = get_node_vars(antidiffusive_flux2_L, equations,
                                                     dg,
                                                     i, j, element)
            u_local = get_node_vars(u, equations, dg, i, j, element)
            u_local_m1 = get_node_vars(u, equations, dg, i, j - 1, element)

            # Using mathematic entropy
            v_local = cons2entropy(u_local, equations)
            v_local_m1 = cons2entropy(u_local_m1, equations)

            q_local = u_local[3] / u_local[1] * entropy(u_local, equations)
            q_local_m1 = u_local_m1[3] / u_local_m1[1] * entropy(u_local_m1, equations)

            f_local = flux(u_local, 2, equations)
            f_local_m1 = flux(u_local_m1, 2, equations)

            psi_local = dot(v_local, f_local) - q_local
            psi_local_m1 = dot(v_local_m1, f_local_m1) - q_local_m1

            delta_v = v_local - v_local_m1
            delta_psi = psi_local - psi_local_m1

            entProd_FV = dot(delta_v, view(fstar2, :, i, j)) - delta_psi
            delta_entProd = dot(delta_v, antidiffusive_flux_local)

            alpha = 1 # Initialize alpha for plotting
            if (entProd_FV + delta_entProd > 0.0) && (delta_entProd != 0.0)
                alpha = min(1.0,
                            (abs(entProd_FV) + eps()) / (abs(delta_entProd) + eps()))
                for v in eachvariable(equations)
                    antidiffusive_flux2_L[v, i, j, element] = alpha *
                                                              antidiffusive_flux2_L[v,
                                                                                    i,
                                                                                    j,
                                                                                    element]
                end
            end
            if limiter.Plotting
                (; alpha_entropy, alpha_mean_entropy) = limiter.cache.subcell_limiter_coefficients
                alpha_entropy[i, j - 1, element] = min(alpha_entropy[i, j - 1, element],
                                                       alpha)
                alpha_entropy[i, j, element] = min(alpha_entropy[i, j, element], alpha)
                alpha_mean_entropy[i, j - 1, element] += alpha
                alpha_mean_entropy[i, j, element] += alpha
            end
        end
        if limiter.Plotting
            (; alpha_mean_entropy) = limiter.cache.subcell_limiter_coefficients
            # Interfaces contribute with 1.0
            for i in eachnode(dg)
                alpha_mean_entropy[i, 1, element] += 1.0
                alpha_mean_entropy[i, nnodes(dg), element] += 1.0
                alpha_mean_entropy[1, i, element] += 1.0
                alpha_mean_entropy[nnodes(dg), i, element] += 1.0
            end
            for j in eachnode(dg), i in eachnode(dg)
                alpha_mean_entropy[i, j, element] /= 4
            end
        end
    end

    # Copy antidiffusive fluxes left to antidifussive fluxes right
    for j in eachnode(dg), i in 2:nnodes(dg), v in eachvariable(equations)
        antidiffusive_flux1_R[v, i, j, element] = antidiffusive_flux1_L[v, i, j,
                                                                        element]
    end
    for j in 2:nnodes(dg), i in eachnode(dg), v in eachvariable(equations)
        antidiffusive_flux2_R[v, i, j, element] = antidiffusive_flux2_L[v, i, j,
                                                                        element]
    end

    return nothing
end

@inline function get_boundary_outer_state(u_inner, cache, t,
                                          boundary_condition::typeof(boundary_condition_slip_wall),
                                          orientation::Integer, direction,
                                          mesh, equations::CompressibleEulerEquations2D,
                                          dg, indices...)
    return SVector(u_inner[1], -u_inner[2], -u_inner[3], u_inner[4])
end

@inline function get_boundary_outer_state(u_inner, cache, t,
                                          boundary_condition::typeof(boundary_condition_slip_wall),
                                          normal_direction::AbstractVector, direction,
                                          mesh, equations::CompressibleEulerEquations2D,
                                          dg, indices...)
    factor = (normal_direction[1] * u_inner[2] + normal_direction[2] * u_inner[3])
    u_normal = (factor / sum(normal_direction .^ 2)) * normal_direction

    return SVector(u_inner[1],
                   u_inner[2] - 2.0 * u_normal[1],
                   u_inner[3] - 2.0 * u_normal[2],
                   u_inner[4])
end

@inline function get_boundary_outer_state(u_inner, cache, t,
                                          boundary_condition::BoundaryConditionDirichlet,
                                          orientation_or_normal, direction, mesh,
                                          equations, dg, indices...)
    (; node_coordinates) = cache.elements

    x = get_node_coords(node_coordinates, equations, dg, indices...)
    u_outer = boundary_condition.boundary_value_function(x, t, equations)

    return u_outer
end

@inline function get_boundary_outer_state(u_inner, cache, t,
                                          boundary_condition::BoundaryConditionCharacteristic,
                                          orientation_or_normal, direction,
                                          mesh::Union{TreeMesh, StructuredMesh},
                                          equations,
                                          dg, indices...)
    (; node_coordinates) = cache.elements

    x = get_node_coords(node_coordinates, equations, dg, indices...)
    u_outer = boundary_condition.boundary_value_function(boundary_condition.outer_boundary_value_function,
                                                         u_inner, orientation_or_normal,
                                                         direction, x, t, equations)

    return u_outer
end

@inline function get_boundary_outer_state(u_inner, cache, t,
                                          boundary_condition::BoundaryConditionCharacteristic,
                                          normal_direction::AbstractVector, direction,
                                          mesh::P4estMesh, equations, dg, indices...)
    (; node_coordinates) = cache.elements

    x = get_node_coords(node_coordinates, equations, dg, indices...)

    u_outer = boundary_condition.boundary_value_function(boundary_condition.outer_boundary_value_function,
                                                         u_inner,
                                                         normal_direction,
                                                         x, t, equations)

    return u_outer
end
end # @muladd
