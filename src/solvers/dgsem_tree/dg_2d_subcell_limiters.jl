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

# The methods below are specialized on the mortar type
# and called from the basic `create_cache` method at the top.
function create_cache(mesh::TreeMesh{2},
                      equations, mortar_idp::LobattoLegendreMortarIDP, uEltype)
    # TODO: Taal performance using different types

    # TODO: What do I really need?
    MA2d = MArray{Tuple{nvariables(equations), nnodes(mortar_idp)}, uEltype, 2,
                  nvariables(equations) * nnodes(mortar_idp)}
    fstar_primary_upper_threaded = MA2d[MA2d(undef) for _ in 1:Threads.nthreads()]
    fstar_primary_lower_threaded = MA2d[MA2d(undef) for _ in 1:Threads.nthreads()]
    fstar_secondary_upper_threaded = MA2d[MA2d(undef) for _ in 1:Threads.nthreads()]
    fstar_secondary_lower_threaded = MA2d[MA2d(undef) for _ in 1:Threads.nthreads()]

    # A2d = Array{uEltype, 2}
    # fstar_upper_threaded = [A2d(undef, nvariables(equations), nnodes(mortar_idp)) for _ in 1:Threads.nthreads()]
    # fstar_lower_threaded = [A2d(undef, nvariables(equations), nnodes(mortar_idp)) for _ in 1:Threads.nthreads()]

    (; fstar_primary_upper_threaded, fstar_primary_lower_threaded,
     fstar_secondary_upper_threaded, fstar_secondary_lower_threaded)
end

function calc_volume_integral!(du, u,
                               mesh::Union{TreeMesh{2}, StructuredMesh{2},
                                           P4estMesh{2}},
                               nonconservative_terms, equations,
                               volume_integral::VolumeIntegralSubcellLimiting,
                               dg::DGSEM, cache)
    @unpack limiter = volume_integral

    @threaded for element in eachelement(dg, cache)
        subcell_limiting_kernel!(du, u, element, mesh,
                                 nonconservative_terms, equations,
                                 volume_integral, limiter,
                                 dg, cache)
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
                                           0.5f0 * derivative_split[i, ii],
                                           flux1_noncons,
                                           equations, dg, noncons, i, j)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5f0 * derivative_split[ii, i],
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
                                         u,
                                         mesh::Union{TreeMesh{2}, StructuredMesh{2},
                                                     P4estMesh{2}},
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
                                         u,
                                         mesh::Union{TreeMesh{2}, StructuredMesh{2},
                                                     P4estMesh{2}},
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

function prolong2mortars!(cache, u,
                          mesh::TreeMesh{2}, equations,
                          mortar_l2::LobattoLegendreMortarIDP,
                          dg::DGSEM)
    @threaded for mortar in eachmortar(dg, cache)
        large_element = cache.mortars.neighbor_ids[3, mortar]
        upper_element = cache.mortars.neighbor_ids[2, mortar]
        lower_element = cache.mortars.neighbor_ids[1, mortar]

        # Copy solutions
        if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
            if cache.mortars.orientations[mortar] == 1
                # IDP mortars in x-direction
                for l in eachnode(dg)
                    for v in eachvariable(equations)
                        cache.mortars.u_upper[v, l, mortar] = u[v, 1, l, upper_element]
                        cache.mortars.u_lower[v, l, mortar] = u[v, 1, l, lower_element]
                        cache.mortars.u_large[v, l, mortar] = u[v, nnodes(dg), l,
                                                                large_element]
                    end
                end
            else
                # IDP mortars in y-direction
                for l in eachnode(dg)
                    for v in eachvariable(equations)
                        cache.mortars.u_upper[v, l, mortar] = u[v, l, 1, upper_element]
                        cache.mortars.u_lower[v, l, mortar] = u[v, l, 1, lower_element]
                        cache.mortars.u_large[v, l, mortar] = u[v, l, nnodes(dg),
                                                                large_element]
                    end
                end
            end
        else # large_sides[mortar] == 2 -> small elements on left side
            if cache.mortars.orientations[mortar] == 1
                # IDP mortars in x-direction
                for l in eachnode(dg)
                    for v in eachvariable(equations)
                        cache.mortars.u_upper[v, l, mortar] = u[v, nnodes(dg), l,
                                                                upper_element]
                        cache.mortars.u_lower[v, l, mortar] = u[v, nnodes(dg), l,
                                                                lower_element]
                        cache.mortars.u_large[v, l, mortar] = u[v, 1, l, large_element]
                    end
                end
            else
                # IDP mortars in y-direction
                for l in eachnode(dg)
                    for v in eachvariable(equations)
                        cache.mortars.u_upper[v, l, mortar] = u[v, l, nnodes(dg),
                                                                upper_element]
                        cache.mortars.u_lower[v, l, mortar] = u[v, l, nnodes(dg),
                                                                lower_element]
                        cache.mortars.u_large[v, l, mortar] = u[v, l, 1, large_element]
                    end
                end
            end
        end
    end

    return nothing
end

function calc_mortar_flux!(surface_flux_values,
                           mesh::TreeMesh{2},
                           nonconservative_terms::False, equations,
                           mortar_idp::LobattoLegendreMortarIDP,
                           surface_integral, dg::DG, cache)
    @unpack surface_flux = surface_integral
    @unpack u_lower, u_upper, u_large, orientations = cache.mortars
    @unpack (fstar_primary_upper_threaded, fstar_primary_lower_threaded,
    fstar_secondary_upper_threaded, fstar_secondary_lower_threaded) = cache
    (; weights) = dg.basis

    for mortar in eachmortar(dg, cache)    #=@threaded =#
        large_element = cache.mortars.neighbor_ids[3, mortar]
        upper_element = cache.mortars.neighbor_ids[2, mortar]
        lower_element = cache.mortars.neighbor_ids[1, mortar]

        # Choose thread-specific pre-allocated container
        # fstar_primary_upper = fstar_primary_upper_threaded[Threads.threadid()]
        # fstar_primary_lower = fstar_primary_lower_threaded[Threads.threadid()]
        # fstar_secondary_upper = fstar_secondary_upper_threaded[Threads.threadid()]
        # fstar_secondary_lower = fstar_secondary_lower_threaded[Threads.threadid()]

        # Calculate fluxes
        orientation = orientations[mortar]

        if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
            if orientation == 1
                # L2 mortars in x-direction
                direction_small = 1
                direction_large = 2
            else
                # L2 mortars in y-direction
                direction_small = 3
                direction_large = 4
            end

            surface_flux_values[:, :, direction_small, lower_element] .= zero(eltype(surface_flux_values))
            surface_flux_values[:, :, direction_small, upper_element] .= zero(eltype(surface_flux_values))
            surface_flux_values[:, :, direction_large, large_element] .= zero(eltype(surface_flux_values))
            # Lower element
            for i in eachnode(dg)
                # Call pointwise two-point numerical flux function
                u_lower_local = get_node_vars(u_lower, equations, dg, i, mortar) # u_rr

                # flux part
                f_lower = flux(u_lower_local, orientation, equations) # f_rr
                for j in eachnode(dg)
                    u_large_local = get_node_vars(u_large, equations, dg, j, mortar) # u_ll
                    f_large = flux(u_large_local, orientation, equations) # f_ll

                    lambda_max_local = max_abs_speed_naive(u_large_local, u_lower_local,
                                                           orientation, equations)

                    own_part = 0.5f0 * (f_lower - lambda_max_local * u_lower_local)
                    other_part = 0.5f0 * (f_large + lambda_max_local * u_large_local)
                    local_term = 0.5f0 * weights[j] * (own_part + other_part)

                    add_to_node_vars!(surface_flux_values, local_term, equations, dg,
                                      i, direction_small, lower_element)
                end
            end
            # Upper element
            for i in eachnode(dg)
                # Call pointwise two-point numerical flux function
                u_upper_local = get_node_vars(u_upper, equations, dg, i, mortar) # u_rr

                # flux part
                f_upper = flux(u_upper_local, orientation, equations) # f_rr
                for j in eachnode(dg)
                    u_large_local = get_node_vars(u_large, equations, dg, j, mortar) # u_ll
                    f_large = flux(u_large_local, orientation, equations) # f_ll

                    lambda_max_local = max_abs_speed_naive(u_large_local, u_upper_local,
                                                           orientation, equations)

                    own_part = 0.5f0 * (f_upper - lambda_max_local * u_upper_local)
                    other_part = 0.5f0 * (f_large + lambda_max_local * u_large_local)
                    local_term = 0.5f0 * weights[j] * (own_part + other_part)

                    add_to_node_vars!(surface_flux_values, local_term, equations, dg,
                                      i, direction_small, upper_element)
                end
            end
            # Large element
            for i in eachnode(dg)
                # Call pointwise two-point numerical flux function
                u_large_local = get_node_vars(u_large, equations, dg, i, mortar) # u_ll

                # flux part
                f_large = flux(u_large_local, orientation, equations) # f_ll
                for j in eachnode(dg)
                    # lower
                    u_lower_local = get_node_vars(u_lower, equations, dg, j, mortar) # u_rr
                    f_lower = flux(u_lower_local, orientation, equations) # f_rr

                    lambda_max_local = max_abs_speed_naive(u_large_local, u_lower_local,
                                                           orientation, equations)

                    own_part = 0.5f0 * (f_large + lambda_max_local * u_large_local)
                    other_part = 0.5f0 * (f_lower - lambda_max_local * u_lower_local)
                    local_term = 0.25f0 * weights[j] * (own_part + other_part)

                    add_to_node_vars!(surface_flux_values, local_term, equations, dg,
                                      i, direction_large, large_element)

                    # upper
                    u_upper_local = get_node_vars(u_upper, equations, dg, j, mortar) # u_rr
                    f_upper = flux(u_upper_local, orientation, equations) # f_rr

                    lambda_max_local = max_abs_speed_naive(u_large_local, u_upper_local,
                                                           orientation, equations)

                    own_part = 0.5f0 * (f_large + lambda_max_local * u_large_local)
                    other_part = 0.5f0 * (f_upper - lambda_max_local * u_upper_local)
                    local_term = 0.25f0 * weights[j] * (own_part + other_part)

                    add_to_node_vars!(surface_flux_values, local_term, equations, dg,
                                      i, direction_large, large_element)
                end
            end
        else # large_sides[mortar] == 2 -> small elements on left side
            if orientation == 1
                # L2 mortars in x-direction
                direction_small = 2
                direction_large = 1
            else
                # L2 mortars in y-direction
                direction_small = 4
                direction_large = 3
            end

            surface_flux_values[:, :, direction_small, lower_element] .= zero(eltype(surface_flux_values))
            surface_flux_values[:, :, direction_small, upper_element] .= zero(eltype(surface_flux_values))
            surface_flux_values[:, :, direction_large, large_element] .= zero(eltype(surface_flux_values))
            # Lower element
            for i in eachnode(dg)
                # Call pointwise two-point numerical flux function
                u_lower_local = get_node_vars(u_lower, equations, dg, i, mortar) # u_ll

                # flux part
                f_lower = flux(u_lower_local, orientation, equations)
                for j in eachnode(dg)
                    u_large_local = get_node_vars(u_large, equations, dg, j, mortar) # u_rr
                    f_large = flux(u_large_local, orientation, equations)

                    lambda_max_local = max_abs_speed_naive(u_lower_local, u_large_local,
                                                           orientation, equations)

                    own_part = 0.5f0 * (f_lower + lambda_max_local * u_lower_local)
                    other_part = 0.5f0 * (f_large - lambda_max_local * u_large_local)
                    local_term = 0.5f0 * weights[j] * (own_part + other_part)

                    add_to_node_vars!(surface_flux_values, local_term, equations, dg,
                                      i, direction_small, lower_element)
                end
            end
            # Upper element
            for i in eachnode(dg)
                # Call pointwise two-point numerical flux function
                u_upper_local = get_node_vars(u_upper, equations, dg, i, mortar) # u_ll

                # flux part
                f_upper = flux(u_upper_local, orientation, equations)
                for j in eachnode(dg)
                    u_large_local = get_node_vars(u_large, equations, dg, j, mortar) # u_rr
                    f_large = flux(u_large_local, orientation, equations)

                    lambda_max_local = max_abs_speed_naive(u_upper_local, u_large_local,
                                                           orientation, equations)

                    own_part = 0.5f0 * (f_upper + lambda_max_local * u_upper_local)
                    other_part = 0.5f0 * (f_large - lambda_max_local * u_large_local)
                    local_term = 0.5f0 * weights[j] * (own_part + other_part)

                    add_to_node_vars!(surface_flux_values, local_term, equations, dg,
                                      i, direction_small, upper_element)
                end
            end
            # Large element
            for i in eachnode(dg)
                # Call pointwise two-point numerical flux function
                u_large_local = get_node_vars(u_large, equations, dg, i, mortar) # u_rr

                # flux part
                f_large = flux(u_large_local, orientation, equations)
                for j in eachnode(dg)
                    # lower
                    u_lower_local = get_node_vars(u_lower, equations, dg, j, mortar) # u_ll
                    f_lower = flux(u_lower_local, orientation, equations)

                    lambda_max_local = max_abs_speed_naive(u_lower_local, u_large_local,
                                                           orientation, equations)

                    own_part = 0.5f0 * (f_large - lambda_max_local * u_large_local)
                    other_part = 0.5f0 * (f_lower + lambda_max_local * u_lower_local)
                    local_term = 0.25f0 * weights[j] * (own_part + other_part)

                    add_to_node_vars!(surface_flux_values, local_term, equations, dg,
                                      i, direction_large, large_element)

                    # upper
                    u_upper_local = get_node_vars(u_upper, equations, dg, j, mortar) # u_ll
                    f_upper = flux(u_upper_local, orientation, equations)

                    lambda_max_local = max_abs_speed_naive(u_upper_local, u_large_local,
                                                           orientation, equations)

                    own_part = 0.5f0 * (f_large - lambda_max_local * u_large_local)
                    other_part = 0.5f0 * (f_upper + lambda_max_local * u_upper_local)
                    local_term = 0.25f0 * weights[j] * (own_part + other_part)

                    add_to_node_vars!(surface_flux_values, local_term, equations, dg,
                                      i, direction_large, large_element)
                end
            end
        end
    end

    # #=@threaded =#for mortar in eachmortar(dg, cache)
    #     # Choose thread-specific pre-allocated container
    #     # fstar_primary_upper = fstar_primary_upper_threaded[Threads.threadid()]
    #     # fstar_primary_lower = fstar_primary_lower_threaded[Threads.threadid()]
    #     # fstar_secondary_upper = fstar_secondary_upper_threaded[Threads.threadid()]
    #     # fstar_secondary_lower = fstar_secondary_lower_threaded[Threads.threadid()]

    #     # Calculate fluxes
    #     orientation = orientations[mortar]
    #     calc_fstar_IDP_AMR!(surface_flux_values, equations, surface_flux, dg, u_lower, u_upper, u_large, mortar,
    #                 orientation)
    #     # calc_fstar_IDP_AMR!(fstar_primary_lower, equations, surface_flux, dg, u_lower, mortar,
    #     #             orientation)
    #     # calc_fstar_IDP_AMR!(fstar_secondary_upper, equations, surface_flux, dg, u_upper, mortar,
    #     #             orientation)
    #     # calc_fstar_IDP_AMR!(fstar_secondary_lower, equations, surface_flux, dg, u_lower, mortar,
    #     #             orientation)

    #     # mortar_fluxes_to_elements!(surface_flux_values,
    #     #                            mesh, equations, mortar_idp, dg, cache,
    #     #                            mortar, fstar_primary_upper, fstar_primary_lower,
    #     #                            fstar_secondary_upper, fstar_secondary_lower)
    # end

    return nothing
end

"""
    get_boundary_outer_state(u_inner, t,
                             boundary_condition::BoundaryConditionDirichlet,
                             orientation_or_normal, direction,
                             equations, dg, cache, indices...)
For subcell limiting, the calculation of local bounds for non-periodic domains requires the boundary
outer state. This function returns the boundary value  for [`BoundaryConditionDirichlet`](@ref) at
time `t` and for node with spatial indices `indices` at the boundary with `orientation_or_normal`
and `direction`.

Should be used together with [`TreeMesh`](@ref) or [`StructuredMesh`](@ref).

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
@inline function get_boundary_outer_state(u_inner, t,
                                          boundary_condition::BoundaryConditionDirichlet,
                                          orientation_or_normal, direction,
                                          equations, dg, cache, indices...)
    (; node_coordinates) = cache.elements

    x = get_node_coords(node_coordinates, equations, dg, indices...)
    u_outer = boundary_condition.boundary_value_function(x, t, equations)

    return u_outer
end
end # @muladd
