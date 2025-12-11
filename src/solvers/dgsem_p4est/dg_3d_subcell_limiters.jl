# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function create_cache(mesh::P4estMesh{3},
                      equations, volume_integral::VolumeIntegralSubcellLimiting,
                      dg::DG, cache_containers, uEltype)
    cache = create_cache(mesh, equations,
                         VolumeIntegralPureLGLFiniteVolume(volume_integral.volume_flux_fv),
                         dg, uEltype)

    A4d = Array{uEltype, 4}

    fhat1_L_threaded = A4d[A4d(undef, nvariables(equations),
                               nnodes(dg) + 1, nnodes(dg), nnodes(dg))
                           for _ in 1:Threads.maxthreadid()]
    fhat1_R_threaded = A4d[A4d(undef, nvariables(equations),
                               nnodes(dg) + 1, nnodes(dg), nnodes(dg))
                           for _ in 1:Threads.maxthreadid()]
    fhat2_L_threaded = A4d[A4d(undef, nvariables(equations),
                               nnodes(dg), nnodes(dg) + 1, nnodes(dg))
                           for _ in 1:Threads.maxthreadid()]
    fhat2_R_threaded = A4d[A4d(undef, nvariables(equations),
                               nnodes(dg), nnodes(dg) + 1, nnodes(dg))
                           for _ in 1:Threads.maxthreadid()]
    fhat3_L_threaded = A4d[A4d(undef, nvariables(equations),
                               nnodes(dg), nnodes(dg), nnodes(dg) + 1)
                           for _ in 1:Threads.maxthreadid()]
    fhat3_R_threaded = A4d[A4d(undef, nvariables(equations),
                               nnodes(dg), nnodes(dg), nnodes(dg) + 1)
                           for _ in 1:Threads.maxthreadid()]

    flux_temp_threaded = A4d[A4d(undef, nvariables(equations),
                                 nnodes(dg), nnodes(dg), nnodes(dg))
                             for _ in 1:Threads.maxthreadid()]
    fhat_temp_threaded = A4d[A4d(undef, nvariables(equations),
                                 nnodes(dg), nnodes(dg), nnodes(dg))
                             for _ in 1:Threads.maxthreadid()]

    @threaded for t in eachindex(fhat1_L_threaded)
        fhat1_L_threaded[t][:, 1, :, :] .= zero(uEltype)
        fhat1_R_threaded[t][:, 1, :, :] .= zero(uEltype)
        fhat1_L_threaded[t][:, nnodes(dg) + 1, :, :] .= zero(uEltype)
        fhat1_R_threaded[t][:, nnodes(dg) + 1, :, :] .= zero(uEltype)

        fhat2_L_threaded[t][:, :, 1, :] .= zero(uEltype)
        fhat2_R_threaded[t][:, :, 1, :] .= zero(uEltype)
        fhat2_L_threaded[t][:, :, nnodes(dg) + 1, :] .= zero(uEltype)
        fhat2_R_threaded[t][:, :, nnodes(dg) + 1, :] .= zero(uEltype)

        fhat3_L_threaded[t][:, :, :, 1] .= zero(uEltype)
        fhat3_R_threaded[t][:, :, :, 1] .= zero(uEltype)
        fhat3_L_threaded[t][:, :, :, nnodes(dg) + 1] .= zero(uEltype)
        fhat3_R_threaded[t][:, :, :, nnodes(dg) + 1] .= zero(uEltype)
    end

    antidiffusive_fluxes = ContainerAntidiffusiveFlux3D{uEltype}(0,
                                                                 nvariables(equations),
                                                                 nnodes(dg))

    if have_nonconservative_terms(equations) == true
        error("Unsupported system of equations with nonconservative terms")
    end

    return (; cache..., antidiffusive_fluxes,
            fhat1_L_threaded, fhat1_R_threaded,
            fhat2_L_threaded, fhat2_R_threaded,
            fhat3_L_threaded, fhat3_R_threaded,
            flux_temp_threaded, fhat_temp_threaded)
end

@inline function subcell_limiting_kernel!(du, u, element,
                                          mesh::P4estMesh{3},
                                          nonconservative_terms, equations,
                                          volume_integral, limiter::SubcellLimiterIDP,
                                          dg::DGSEM, cache)
    @unpack inverse_weights = dg.basis # Plays role of DG subcell sizes
    @unpack volume_flux_dg, volume_flux_fv = volume_integral

    # high-order DG fluxes
    @unpack fhat1_L_threaded, fhat1_R_threaded, fhat2_L_threaded, fhat2_R_threaded, fhat3_L_threaded, fhat3_R_threaded = cache

    fhat1_L = fhat1_L_threaded[Threads.threadid()]
    fhat1_R = fhat1_R_threaded[Threads.threadid()]
    fhat2_L = fhat2_L_threaded[Threads.threadid()]
    fhat2_R = fhat2_R_threaded[Threads.threadid()]
    fhat3_L = fhat3_L_threaded[Threads.threadid()]
    fhat3_R = fhat3_R_threaded[Threads.threadid()]
    calcflux_fhat!(fhat1_L, fhat1_R, fhat2_L, fhat2_R, fhat3_L, fhat3_R,
                   u, mesh, nonconservative_terms, equations, volume_flux_dg,
                   dg, element, cache)

    # low-order FV fluxes
    @unpack fstar1_L_threaded, fstar1_R_threaded, fstar2_L_threaded, fstar2_R_threaded, fstar3_L_threaded, fstar3_R_threaded = cache

    fstar1_L = fstar1_L_threaded[Threads.threadid()]
    fstar1_R = fstar1_R_threaded[Threads.threadid()]
    fstar2_L = fstar2_L_threaded[Threads.threadid()]
    fstar2_R = fstar2_R_threaded[Threads.threadid()]
    fstar3_L = fstar3_L_threaded[Threads.threadid()]
    fstar3_R = fstar3_R_threaded[Threads.threadid()]
    calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, fstar3_L, fstar3_R,
                 u, mesh, nonconservative_terms, equations, volume_flux_fv,
                 dg, element, cache)

    # antidiffusive flux
    calcflux_antidiffusive!(fhat1_L, fhat1_R, fhat2_L, fhat2_R, fhat3_L, fhat3_R,
                            fstar1_L, fstar1_R, fstar2_L, fstar2_R, fstar3_L, fstar3_R,
                            u, mesh, nonconservative_terms, equations, limiter,
                            dg, element, cache)

    # Calculate volume integral contribution of low-order FV flux
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        for v in eachvariable(equations)
            du[v, i, j, k, element] += inverse_weights[i] *
                                       (fstar1_L[v, i + 1, j, k] - fstar1_R[v, i, j, k]) +
                                       inverse_weights[j] *
                                       (fstar2_L[v, i, j + 1, k] - fstar2_R[v, i, j, k]) +
                                       inverse_weights[k] *
                                       (fstar3_L[v, i, j, k + 1] - fstar3_R[v, i, j, k])
        end
    end

    return nothing
end

# Calculate the DG staggered volume fluxes `fhat` in subcell FV-form inside the element
# (**without non-conservative terms**).
#
# See also `flux_differencing_kernel!`.
@inline function calcflux_fhat!(fhat1_L, fhat1_R, fhat2_L, fhat2_R, fhat3_L, fhat3_R,
                                u, mesh::P4estMesh{3},
                                nonconservative_terms::False, equations,
                                volume_flux, dg::DGSEM, element, cache)
    (; contravariant_vectors) = cache.elements
    (; weights, derivative_split) = dg.basis
    (; flux_temp_threaded) = cache

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

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)

        # pull the contravariant vectors in each coordinate direction
        Ja1_node = get_contravariant_vector(1, contravariant_vectors,
                                            i, j, k, element) # x direction

        # All diagonal entries of `derivative_split` are zero. Thus, we can skip
        # the computation of the diagonal terms. In addition, we use the symmetry
        # of the `volume_flux` to save half of the possible two-point flux
        # computations.

        # x direction
        for ii in (i + 1):nnodes(dg)
            u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
            # pull the contravariant vectors and compute the average
            Ja1_node_ii = get_contravariant_vector(1, contravariant_vectors,
                                                   ii, j, k, element)
            Ja1_avg = 0.5f0 * (Ja1_node + Ja1_node_ii)

            # compute the contravariant sharp flux in the direction of the averaged contravariant vector
            fluxtilde1 = volume_flux(u_node, u_node_ii, Ja1_avg, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[i, ii], fluxtilde1,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[ii, i], fluxtilde1,
                                       equations, dg, ii, j, k)
        end
    end

    # FV-form flux `fhat` in x direction
    for k in eachnode(dg), j in eachnode(dg), i in 1:(nnodes(dg) - 1),
        v in eachvariable(equations)

        fhat1_L[v, i + 1, j, k] = fhat1_L[v, i, j, k] +
                                  weights[i] * flux_temp[v, i, j, k]
        fhat1_R[v, i + 1, j, k] = fhat1_L[v, i + 1, j, k]
    end

    # Split form volume flux in orientation 2: y direction
    flux_temp .= zero(eltype(flux_temp))

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)

        # pull the contravariant vectors in each coordinate direction
        Ja2_node = get_contravariant_vector(2, contravariant_vectors,
                                            i, j, k, element)

        # y direction
        for jj in (j + 1):nnodes(dg)
            u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
            # pull the contravariant vectors and compute the average
            Ja2_node_jj = get_contravariant_vector(2, contravariant_vectors,
                                                   i, jj, k, element)
            Ja2_avg = 0.5f0 * (Ja2_node + Ja2_node_jj)
            # compute the contravariant sharp flux in the direction of the averaged contravariant vector
            fluxtilde2 = volume_flux(u_node, u_node_jj, Ja2_avg, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[j, jj], fluxtilde2,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[jj, j], fluxtilde2,
                                       equations, dg, i, jj, k)
        end
    end

    # FV-form flux `fhat` in y direction
    for k in eachnode(dg), j in 1:(nnodes(dg) - 1), i in eachnode(dg),
        v in eachvariable(equations)

        fhat2_L[v, i, j + 1, k] = fhat2_L[v, i, j, k] +
                                  weights[j] * flux_temp[v, i, j, k]
        fhat2_R[v, i, j + 1, k] = fhat2_L[v, i, j + 1, k]
    end

    # Split form volume flux in orientation 3: z direction
    flux_temp .= zero(eltype(flux_temp))

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)

        # pull the contravariant vectors in each coordinate direction
        Ja3_node = get_contravariant_vector(3, contravariant_vectors,
                                            i, j, k, element)

        # z direction
        for kk in (k + 1):nnodes(dg)
            u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
            # pull the contravariant vectors and compute the average
            Ja3_node_kk = get_contravariant_vector(3, contravariant_vectors,
                                                   i, j, kk, element)
            Ja3_avg = 0.5f0 * (Ja3_node + Ja3_node_kk)
            # compute the contravariant sharp flux in the direction of the averaged contravariant vector
            fluxtilde3 = volume_flux(u_node, u_node_kk, Ja3_avg, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[k, kk], fluxtilde3,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[kk, k], fluxtilde3,
                                       equations, dg, i, j, kk)
        end
    end

    # FV-form flux `fhat` in z direction
    for k in 1:(nnodes(dg) - 1), j in eachnode(dg), i in eachnode(dg),
        v in eachvariable(equations)

        fhat3_L[v, i, j, k + 1] = fhat3_L[v, i, j, k] +
                                  weights[k] * flux_temp[v, i, j, k]
        fhat3_R[v, i, j, k + 1] = fhat3_L[v, i, j, k + 1]
    end

    return nothing
end

# Calculate the antidiffusive flux `antidiffusive_flux` as the subtraction between `fhat` and `fstar` for conservative systems.
@inline function calcflux_antidiffusive!(fhat1_L, fhat1_R,
                                         fhat2_L, fhat2_R,
                                         fhat3_L, fhat3_R,
                                         fstar1_L, fstar1_R,
                                         fstar2_L, fstar2_R,
                                         fstar3_L, fstar3_R,
                                         u, mesh::P4estMesh{3},
                                         nonconservative_terms::False, equations,
                                         limiter::SubcellLimiterIDP, dg, element, cache)
    @unpack antidiffusive_flux1_L, antidiffusive_flux1_R, antidiffusive_flux2_L, antidiffusive_flux2_R, antidiffusive_flux3_L, antidiffusive_flux3_R = cache.antidiffusive_fluxes

    # Due to the use of LGL nodes, the DG staggered fluxes `fhat` and FV fluxes `fstar` are equal
    # on the element interfaces. So, they are not computed in the volume integral and set to zero
    # in their respective computation.
    # The antidiffusive fluxes are therefore zero on the element interfaces and don't need to be
    # computed either. They are set to zero directly after resizing the container.
    # This applies to the indices `i=1` and `i=nnodes(dg)+1` for `antidiffusive_flux1_L` and
    # `antidiffusive_flux1_R` and analogously for the other two directions.

    for k in eachnode(dg), j in eachnode(dg), i in 2:nnodes(dg)
        for v in eachvariable(equations)
            antidiffusive_flux1_L[v, i, j, k, element] = fhat1_L[v, i, j, k] -
                                                         fstar1_L[v, i, j, k]
            antidiffusive_flux1_R[v, i, j, k, element] = antidiffusive_flux1_L[v,
                                                                               i, j, k,
                                                                               element]
        end
    end
    for k in eachnode(dg), j in 2:nnodes(dg), i in eachnode(dg)
        for v in eachvariable(equations)
            antidiffusive_flux2_L[v, i, j, k, element] = fhat2_L[v, i, j, k] -
                                                         fstar2_L[v, i, j, k]
            antidiffusive_flux2_R[v, i, j, k, element] = antidiffusive_flux2_L[v,
                                                                               i, j, k,
                                                                               element]
        end
    end
    for k in 2:nnodes(dg), j in eachnode(dg), i in eachnode(dg)
        for v in eachvariable(equations)
            antidiffusive_flux3_L[v, i, j, k, element] = fhat3_L[v, i, j, k] -
                                                         fstar3_L[v, i, j, k]
            antidiffusive_flux3_R[v, i, j, k, element] = antidiffusive_flux3_L[v,
                                                                               i, j, k,
                                                                               element]
        end
    end

    return nothing
end
end # @muladd
