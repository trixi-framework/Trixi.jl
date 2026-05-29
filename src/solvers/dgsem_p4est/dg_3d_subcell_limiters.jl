# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Calculate the DG staggered volume fluxes `fhat` in subcell FV-form inside the element
# (**without non-conservative terms**).
#
# See also `flux_differencing_kernel!`.
@inline function calcflux_fhat!(fhat1_L, fhat1_R, fhat2_L, fhat2_R, fhat3_L, fhat3_R,
                                u, ::Type{<:P4estMesh{3}},
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
    for k in eachnode(dg), j in eachnode(dg), i in 1:(nnodes(dg) - 1)
        for v in eachvariable(equations)
            fhat1_L[v, i + 1, j, k] = fhat1_L[v, i, j, k] +
                                      weights[i] * flux_temp[v, i, j, k]
            fhat1_R[v, i + 1, j, k] = fhat1_L[v, i + 1, j, k]
        end
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
    for k in eachnode(dg), j in 1:(nnodes(dg) - 1), i in eachnode(dg)
        for v in eachvariable(equations)
            fhat2_L[v, i, j + 1, k] = fhat2_L[v, i, j, k] +
                                      weights[j] * flux_temp[v, i, j, k]
            fhat2_R[v, i, j + 1, k] = fhat2_L[v, i, j + 1, k]
        end
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
    for k in 1:(nnodes(dg) - 1), j in eachnode(dg), i in eachnode(dg)
        for v in eachvariable(equations)
            fhat3_L[v, i, j, k + 1] = fhat3_L[v, i, j, k] +
                                      weights[k] * flux_temp[v, i, j, k]
            fhat3_R[v, i, j, k + 1] = fhat3_L[v, i, j, k + 1]
        end
    end

    return nothing
end

# Calculate the DG staggered volume fluxes `fhat` in subcell FV-form inside the element
# (**with non-conservative terms in "local * symmetric" form**).
#
# See also `flux_differencing_kernel!`.
#
# The calculation of the non-conservative staggered "fluxes" requires non-conservative
# terms that can be written as a product of local and a symmetric contributions. See, e.g.,
#
# - Rueda-Ramírez, Gassner (2023). A Flux-Differencing Formula for Split-Form Summation By Parts
#   Discretizations of Non-Conservative Systems. https://arxiv.org/pdf/2211.14009.pdf.
#
@inline function calcflux_fhat!(fhat1_L, fhat1_R, fhat2_L, fhat2_R, fhat3_L, fhat3_R,
                                u, ::Type{<:P4estMesh{3}},
                                nonconservative_terms::True, equations,
                                volume_flux::Tuple{F_CONS, F_NONCONS}, dg::DGSEM,
                                element,
                                cache) where {
                                              F_CONS <: Function,
                                              F_NONCONS <:
                                              FluxNonConservative{NonConservativeSymmetric()}
                                              }
    (; contravariant_vectors) = cache.elements
    (; weights, derivative_split) = dg.basis
    (; flux_temp_threaded, flux_nonconservative_temp_threaded) = cache
    (; fhat_temp_threaded, fhat_nonconservative_temp_threaded, phi_threaded) = cache

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

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)

        # pull the contravariant vectors in each coordinate direction
        Ja1_node = get_contravariant_vector(1, contravariant_vectors,
                                            i, j, k, element) # x direction

        # All diagonal entries of `derivative_split` are zero. Thus, we can skip
        # the computation of the diagonal terms. In addition, we use the symmetry
        # of `volume_flux_cons` and `volume_flux_noncons` to save half of the possible two-point flux
        # computations.
        for ii in (i + 1):nnodes(dg)
            u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
            # pull the contravariant vectors and compute the average
            Ja1_node_ii = get_contravariant_vector(1, contravariant_vectors,
                                                   ii, j, k, element)
            Ja1_avg = 0.5f0 * (Ja1_node + Ja1_node_ii)

            # compute the contravariant sharp flux in the direction of the averaged contravariant vector
            fluxtilde1 = volume_flux_cons(u_node, u_node_ii, Ja1_avg, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[i, ii], fluxtilde1,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[ii, i], fluxtilde1,
                                       equations, dg, ii, j, k)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux1_noncons = volume_flux_noncons(u_node, u_node_ii, Ja1_avg,
                                                    equations,
                                                    NonConservativeSymmetric(), noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5f0 * derivative_split[i, ii],
                                           flux1_noncons,
                                           equations, dg, noncons, i, j, k)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5f0 * derivative_split[ii, i],
                                           flux1_noncons,
                                           equations, dg, noncons, ii, j, k)
            end
        end
    end

    # FV-form flux `fhat` in x direction
    fhat_temp[:, 1, :, :] .= zero(eltype(fhat1_L))
    fhat_noncons_temp[:, :, 1, :, :] .= zero(eltype(fhat1_L))

    # Compute local contribution to non-conservative flux
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_local = get_node_vars(u, equations, dg, i, j, k, element)
        # pull the local contravariant vector
        Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, Ja1_node, equations,
                                               NonConservativeLocal(), noncons),
                           equations, dg, noncons, i, j, k)
        end
    end

    for k in eachnode(dg), j in eachnode(dg), i in 1:(nnodes(dg) - 1)
        # Conservative part
        for v in eachvariable(equations)
            value = fhat_temp[v, i, j, k] + weights[i] * flux_temp[v, i, j, k]
            fhat_temp[v, i + 1, j, k] = value
            fhat1_L[v, i + 1, j, k] = value
            fhat1_R[v, i + 1, j, k] = value
        end
        # Nonconservative part
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons),
            v in eachvariable(equations)

            value = fhat_noncons_temp[v, noncons, i, j, k] +
                    weights[i] * flux_noncons_temp[v, noncons, i, j, k]
            fhat_noncons_temp[v, noncons, i + 1, j, k] = value

            fhat1_L[v, i + 1, j, k] = fhat1_L[v, i + 1, j, k] +
                                      phi[v, noncons, i, j, k] * value
            fhat1_R[v, i + 1, j, k] = fhat1_R[v, i + 1, j, k] +
                                      phi[v, noncons, i + 1, j, k] * value
        end
    end

    # Split form volume flux in orientation 2: y direction
    flux_temp .= zero(eltype(flux_temp))
    flux_noncons_temp .= zero(eltype(flux_noncons_temp))

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)

        # pull the contravariant vectors in each coordinate direction
        Ja2_node = get_contravariant_vector(2, contravariant_vectors,
                                            i, j, k, element)

        for jj in (j + 1):nnodes(dg)
            u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
            # pull the contravariant vectors and compute the average
            Ja2_node_jj = get_contravariant_vector(2, contravariant_vectors,
                                                   i, jj, k, element)
            Ja2_avg = 0.5f0 * (Ja2_node + Ja2_node_jj)
            # compute the contravariant sharp flux in the direction of the averaged contravariant vector
            fluxtilde2 = volume_flux_cons(u_node, u_node_jj, Ja2_avg, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[j, jj], fluxtilde2,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[jj, j], fluxtilde2,
                                       equations, dg, i, jj, k)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux2_noncons = volume_flux_noncons(u_node, u_node_jj, Ja2_avg,
                                                    equations,
                                                    NonConservativeSymmetric(), noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5f0 * derivative_split[j, jj],
                                           flux2_noncons,
                                           equations, dg, noncons, i, j, k)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5f0 * derivative_split[jj, j],
                                           flux2_noncons,
                                           equations, dg, noncons, i, jj, k)
            end
        end
    end

    # FV-form flux `fhat` in y direction
    fhat_temp[:, :, 1, :] .= zero(eltype(fhat1_L))
    fhat_noncons_temp[:, :, :, 1, :] .= zero(eltype(fhat1_L))

    # Compute local contribution to non-conservative flux
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_local = get_node_vars(u, equations, dg, i, j, k, element)
        # pull the local contravariant vector
        Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, Ja2_node, equations,
                                               NonConservativeLocal(), noncons),
                           equations, dg, noncons, i, j, k)
        end
    end

    for k in eachnode(dg), j in 1:(nnodes(dg) - 1), i in eachnode(dg)
        # Conservative part
        for v in eachvariable(equations)
            value = fhat_temp[v, i, j, k] + weights[j] * flux_temp[v, i, j, k]
            fhat_temp[v, i, j + 1, k] = value
            fhat2_L[v, i, j + 1, k] = value
            fhat2_R[v, i, j + 1, k] = value
        end
        # Nonconservative part
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons),
            v in eachvariable(equations)

            value = fhat_noncons_temp[v, noncons, i, j, k] +
                    weights[j] * flux_noncons_temp[v, noncons, i, j, k]
            fhat_noncons_temp[v, noncons, i, j + 1, k] = value

            fhat2_L[v, i, j + 1, k] = fhat2_L[v, i, j + 1, k] +
                                      phi[v, noncons, i, j, k] * value
            fhat2_R[v, i, j + 1, k] = fhat2_R[v, i, j + 1, k] +
                                      phi[v, noncons, i, j + 1, k] * value
        end
    end

    # Split form volume flux in orientation 3: z direction
    flux_temp .= zero(eltype(flux_temp))
    flux_noncons_temp .= zero(eltype(flux_noncons_temp))

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)

        # pull the contravariant vectors in each coordinate direction
        Ja3_node = get_contravariant_vector(3, contravariant_vectors,
                                            i, j, k, element)

        for kk in (k + 1):nnodes(dg)
            u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
            # pull the contravariant vectors and compute the average
            Ja3_node_kk = get_contravariant_vector(3, contravariant_vectors,
                                                   i, j, kk, element)
            Ja3_avg = 0.5f0 * (Ja3_node + Ja3_node_kk)
            # compute the contravariant sharp flux in the direction of the averaged contravariant vector
            fluxtilde3 = volume_flux_cons(u_node, u_node_kk, Ja3_avg, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[k, kk], fluxtilde3,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[kk, k], fluxtilde3,
                                       equations, dg, i, j, kk)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux3_noncons = volume_flux_noncons(u_node, u_node_kk, Ja3_avg,
                                                    equations,
                                                    NonConservativeSymmetric(), noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5f0 * derivative_split[k, kk],
                                           flux3_noncons,
                                           equations, dg, noncons, i, j, k)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5f0 * derivative_split[kk, k],
                                           flux3_noncons,
                                           equations, dg, noncons, i, j, kk)
            end
        end
    end

    # FV-form flux `fhat` in z direction
    fhat_temp[:, :, :, 1] .= zero(eltype(fhat1_L))
    fhat_noncons_temp[:, :, :, :, 1] .= zero(eltype(fhat1_L))

    # Compute local contribution to non-conservative flux
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_local = get_node_vars(u, equations, dg, i, j, k, element)
        # pull the local contravariant vector
        Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, Ja3_node, equations,
                                               NonConservativeLocal(), noncons),
                           equations, dg, noncons, i, j, k)
        end
    end

    for k in 1:(nnodes(dg) - 1), j in eachnode(dg), i in eachnode(dg)
        # Conservative part
        for v in eachvariable(equations)
            value = fhat_temp[v, i, j, k] + weights[k] * flux_temp[v, i, j, k]
            fhat_temp[v, i, j, k + 1] = value
            fhat3_L[v, i, j, k + 1] = value
            fhat3_R[v, i, j, k + 1] = value
        end
        # Nonconservative part
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons),
            v in eachvariable(equations)

            value = fhat_noncons_temp[v, noncons, i, j, k] +
                    weights[k] * flux_noncons_temp[v, noncons, i, j, k]
            fhat_noncons_temp[v, noncons, i, j, k + 1] = value

            fhat3_L[v, i, j, k + 1] = fhat3_L[v, i, j, k + 1] +
                                      phi[v, noncons, i, j, k] * value
            fhat3_R[v, i, j, k + 1] = fhat3_R[v, i, j, k + 1] +
                                      phi[v, noncons, i, j, k + 1] * value
        end
    end

    return nothing
end

# Calculate the DG staggered volume fluxes `fhat` in subcell FV-form inside the element
# (**with non-conservative terms in "local * jump" form**).
#
# See also `flux_differencing_kernel!`.
#
# The calculation of the non-conservative staggered "fluxes" requires non-conservative
# terms that can be written as a product of local and jump contributions.
@inline function calcflux_fhat!(fhat1_L, fhat1_R, fhat2_L, fhat2_R, fhat3_L, fhat3_R, u,
                                ::Type{<:Union{StructuredMesh{3}, P4estMesh{3}}},
                                have_nonconservative_terms::True, equations,
                                volume_flux::Tuple{F_CONS, F_NONCONS}, dg::DGSEM,
                                element,
                                cache) where {
                                              F_CONS <: Function,
                                              F_NONCONS <:
                                              FluxNonConservative{NonConservativeJump()}
                                              }
    (; contravariant_vectors) = cache.elements
    (; weights, derivative_split) = dg.basis
    (; flux_temp_threaded, flux_nonconservative_temp_threaded) = cache
    (; fhat_temp_threaded, fhat_nonconservative_temp_threaded, phi_threaded) = cache

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

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)

        # pull the contravariant vectors in each coordinate direction
        Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element) # x direction

        # All diagonal entries of `derivative_split` are zero. Thus, we can skip
        # the computation of the diagonal terms. In addition, we use the symmetry
        # of `volume_flux_cons` and `volume_flux_noncons` to save half of the possible two-point flux
        # computations.
        for ii in (i + 1):nnodes(dg)
            u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
            # pull the contravariant vectors and compute the average
            Ja1_node_ii = get_contravariant_vector(1, contravariant_vectors, ii, j, k,
                                                   element)
            Ja1_avg = 0.5f0 * (Ja1_node + Ja1_node_ii)

            # compute the contravariant sharp flux in the direction of the averaged contravariant vector
            fluxtilde1 = volume_flux_cons(u_node, u_node_ii, Ja1_avg, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[i, ii], fluxtilde1,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[ii, i], fluxtilde1,
                                       equations, dg, ii, j, k)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux1_noncons = volume_flux_noncons(u_node, u_node_ii, Ja1_avg,
                                                    equations,
                                                    NonConservativeJump(), noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5f0 * derivative_split[i, ii],
                                           flux1_noncons,
                                           equations, dg, noncons, i, j, k)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           -0.5f0 * derivative_split[ii, i],
                                           flux1_noncons,
                                           equations, dg, noncons, ii, j, k)
            end
        end
    end

    # FV-form flux `fhat` in x direction
    fhat_temp[:, 1, :, :] .= zero(eltype(fhat1_L))
    fhat_noncons_temp[:, :, 1, :, :] .= zero(eltype(fhat1_L))

    # Compute local contribution to non-conservative flux
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_local = get_node_vars(u, equations, dg, i, j, k, element)
        # pull the local contravariant vector
        Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, Ja1_node, equations,
                                               NonConservativeLocal(), noncons),
                           equations, dg, noncons, i, j, k)
        end
    end

    for k in eachnode(dg), j in eachnode(dg), i in 1:(nnodes(dg) - 1)
        # Conservative part
        for v in eachvariable(equations)
            value = fhat_temp[v, i, j, k] + weights[i] * flux_temp[v, i, j, k]
            fhat_temp[v, i + 1, j, k] = value
            fhat1_L[v, i + 1, j, k] = value
            fhat1_R[v, i + 1, j, k] = value
        end
        # Nonconservative part
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons),
            v in eachvariable(equations)

            value = fhat_noncons_temp[v, noncons, i, j, k] +
                    weights[i] * flux_noncons_temp[v, noncons, i, j, k]
            fhat_noncons_temp[v, noncons, i + 1, j, k] = value

            fhat1_L[v, i + 1, j, k] = fhat1_L[v, i + 1, j, k] +
                                      phi[v, noncons, i, j, k] * value
            fhat1_R[v, i + 1, j, k] = fhat1_R[v, i + 1, j, k] +
                                      phi[v, noncons, i + 1, j, k] * value
        end
    end

    # Apply correction term to the flux-differencing formula for nonconservative local * jump fluxes.
    for k in eachnode(dg), j in eachnode(dg)
        u_0 = get_node_vars(u, equations, dg, 1, j, k, element)
        Ja1_node_0 = get_contravariant_vector(1, contravariant_vectors, 1, j, k,
                                              element)

        for i in 2:(nnodes(dg) - 1)
            u_i = get_node_vars(u, equations, dg, i, j, k, element)
            Ja1_node_i = get_contravariant_vector(1, contravariant_vectors, i, j, k,
                                                  element)
            Ja1_avg = 0.5f0 * (Ja1_node_0 + Ja1_node_i)

            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                phi_jump = volume_flux_noncons(u_0, u_i, Ja1_avg, equations,
                                               NonConservativeJump(), noncons)

                for v in eachvariable(equations)
                    # The factor of 2 is missing on each term because Trixi multiplies all the non-cons terms with 0.5
                    fhat1_R[v, i, j, k] -= phi[v, noncons, i, j, k] * phi_jump[v]
                    fhat1_L[v, i + 1, j, k] -= phi[v, noncons, i, j, k] * phi_jump[v]
                end
            end
        end
        u_N = get_node_vars(u, equations, dg, nnodes(dg), j, k, element)
        Ja1_node_N = get_contravariant_vector(1, contravariant_vectors, nnodes(dg), j,
                                              k,
                                              element)
        Ja1_avg = 0.5f0 * (Ja1_node_0 + Ja1_node_N)

        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            phi_jump = volume_flux_noncons(u_0, u_N, Ja1_avg, equations,
                                           NonConservativeJump(), noncons)

            for v in eachvariable(equations)
                # The factor of 2 is missing because Trixi multiplies all the non-cons terms with 0.5
                fhat1_R[v, nnodes(dg), j, k] -= phi[v, noncons, nnodes(dg), j, k] *
                                                phi_jump[v]
            end
        end
    end

    # Split form volume flux in orientation 2: y direction
    flux_temp .= zero(eltype(flux_temp))
    flux_noncons_temp .= zero(eltype(flux_noncons_temp))

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)

        # pull the contravariant vectors in each coordinate direction
        Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)

        for jj in (j + 1):nnodes(dg)
            u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
            # pull the contravariant vectors and compute the average
            Ja2_node_jj = get_contravariant_vector(2, contravariant_vectors, i, jj, k,
                                                   element)
            Ja2_avg = 0.5f0 * (Ja2_node + Ja2_node_jj)
            # compute the contravariant sharp flux in the direction of the averaged contravariant vector
            fluxtilde2 = volume_flux_cons(u_node, u_node_jj, Ja2_avg, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[j, jj], fluxtilde2,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[jj, j], fluxtilde2,
                                       equations, dg, i, jj, k)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux2_noncons = volume_flux_noncons(u_node, u_node_jj, Ja2_avg,
                                                    equations,
                                                    NonConservativeJump(), noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5f0 * derivative_split[j, jj],
                                           flux2_noncons,
                                           equations, dg, noncons, i, j, k)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           -0.5f0 * derivative_split[jj, j],
                                           flux2_noncons,
                                           equations, dg, noncons, i, jj, k)
            end
        end
    end

    # FV-form flux `fhat` in y direction
    fhat_temp[:, :, 1, :] .= zero(eltype(fhat1_L))
    fhat_noncons_temp[:, :, :, 1, :] .= zero(eltype(fhat1_L))

    # Compute local contribution to non-conservative flux
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_local = get_node_vars(u, equations, dg, i, j, k, element)
        # pull the local contravariant vector
        Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, Ja2_node, equations,
                                               NonConservativeLocal(), noncons),
                           equations, dg, noncons, i, j, k)
        end
    end

    for k in eachnode(dg), j in 1:(nnodes(dg) - 1), i in eachnode(dg)
        # Conservative part
        for v in eachvariable(equations)
            value = fhat_temp[v, i, j, k] + weights[j] * flux_temp[v, i, j, k]
            fhat_temp[v, i, j + 1, k] = value
            fhat2_L[v, i, j + 1, k] = value
            fhat2_R[v, i, j + 1, k] = value
        end
        # Nonconservative part
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons),
            v in eachvariable(equations)

            value = fhat_noncons_temp[v, noncons, i, j, k] +
                    weights[j] * flux_noncons_temp[v, noncons, i, j, k]
            fhat_noncons_temp[v, noncons, i, j + 1, k] = value

            fhat2_L[v, i, j + 1, k] = fhat2_L[v, i, j + 1, k] +
                                      phi[v, noncons, i, j, k] * value
            fhat2_R[v, i, j + 1, k] = fhat2_R[v, i, j + 1, k] +
                                      phi[v, noncons, i, j + 1, k] * value
        end
    end

    # Apply correction term to the flux-differencing formula for nonconservative local * jump fluxes.
    for k in eachnode(dg), i in eachnode(dg)
        u_0 = get_node_vars(u, equations, dg, i, 1, k, element)
        Ja2_node_0 = get_contravariant_vector(2, contravariant_vectors, i, 1, k,
                                              element)

        for j in 2:(nnodes(dg) - 1)
            u_j = get_node_vars(u, equations, dg, i, j, k, element)
            Ja2_node_j = get_contravariant_vector(2, contravariant_vectors, i, j, k,
                                                  element)
            Ja2_avg = 0.5f0 * (Ja2_node_0 + Ja2_node_j)

            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                phi_jump = volume_flux_noncons(u_0, u_j, Ja2_avg, equations,
                                               NonConservativeJump(), noncons)

                for v in eachvariable(equations)
                    # The factor of 2 is missing on each term because Trixi multiplies all the non-cons terms with 0.5
                    fhat2_R[v, i, j, k] -= phi[v, noncons, i, j, k] * phi_jump[v]
                    fhat2_L[v, i, j + 1, k] -= phi[v, noncons, i, j, k] * phi_jump[v]
                end
            end
        end
        u_N = get_node_vars(u, equations, dg, i, nnodes(dg), k, element)
        Ja2_node_N = get_contravariant_vector(2, contravariant_vectors, i, nnodes(dg),
                                              k,
                                              element)
        Ja2_avg = 0.5f0 * (Ja2_node_0 + Ja2_node_N)

        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            phi_jump = volume_flux_noncons(u_0, u_N, Ja2_avg, equations,
                                           NonConservativeJump(), noncons)

            for v in eachvariable(equations)
                # The factor of 2 is missing cause Trixi multiplies all the non-cons terms with 0.5
                fhat2_R[v, i, nnodes(dg), k] -= phi[v, noncons, i, nnodes(dg), k] *
                                                phi_jump[v]
            end
        end
    end

    # Split form volume flux in orientation 3: z direction
    flux_temp .= zero(eltype(flux_temp))
    flux_noncons_temp .= zero(eltype(flux_noncons_temp))

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)

        # pull the contravariant vectors in each coordinate direction
        Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)

        for kk in (k + 1):nnodes(dg)
            u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
            # pull the contravariant vectors and compute the average
            Ja3_node_kk = get_contravariant_vector(3, contravariant_vectors, i, j, kk,
                                                   element)
            Ja3_avg = 0.5f0 * (Ja3_node + Ja3_node_kk)
            # compute the contravariant sharp flux in the direction of the averaged contravariant vector
            fluxtilde3 = volume_flux_cons(u_node, u_node_kk, Ja3_avg, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[k, kk], fluxtilde3,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[kk, k], fluxtilde3,
                                       equations, dg, i, j, kk)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux3_noncons = volume_flux_noncons(u_node, u_node_kk, Ja3_avg,
                                                    equations,
                                                    NonConservativeJump(), noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5f0 * derivative_split[k, kk],
                                           flux3_noncons,
                                           equations, dg, noncons, i, j, k)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           -0.5f0 * derivative_split[kk, k],
                                           flux3_noncons,
                                           equations, dg, noncons, i, j, kk)
            end
        end
    end

    # FV-form flux `fhat` in z direction
    fhat_temp[:, :, :, 1] .= zero(eltype(fhat1_L))
    fhat_noncons_temp[:, :, :, :, 1] .= zero(eltype(fhat1_L))

    # Compute local contribution to non-conservative flux
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_local = get_node_vars(u, equations, dg, i, j, k, element)
        # pull the local contravariant vector
        Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, Ja3_node, equations,
                                               NonConservativeLocal(), noncons),
                           equations, dg, noncons, i, j, k)
        end
    end

    for k in 1:(nnodes(dg) - 1), j in eachnode(dg), i in eachnode(dg)
        # Conservative part
        for v in eachvariable(equations)
            value = fhat_temp[v, i, j, k] + weights[k] * flux_temp[v, i, j, k]
            fhat_temp[v, i, j, k + 1] = value
            fhat3_L[v, i, j, k + 1] = value
            fhat3_R[v, i, j, k + 1] = value
        end
        # Nonconservative part
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons),
            v in eachvariable(equations)

            value = fhat_noncons_temp[v, noncons, i, j, k] +
                    weights[k] * flux_noncons_temp[v, noncons, i, j, k]
            fhat_noncons_temp[v, noncons, i, j, k + 1] = value

            fhat3_L[v, i, j, k + 1] = fhat3_L[v, i, j, k + 1] +
                                      phi[v, noncons, i, j, k] * value
            fhat3_R[v, i, j, k + 1] = fhat3_R[v, i, j, k + 1] +
                                      phi[v, noncons, i, j, k + 1] * value
        end
    end

    # Apply correction term to the flux-differencing formula for nonconservative local * jump fluxes.
    for j in eachnode(dg), i in eachnode(dg)
        u_0 = get_node_vars(u, equations, dg, i, j, 1, element)
        Ja3_node_0 = get_contravariant_vector(3, contravariant_vectors, i, j, 1,
                                              element)

        for k in 2:(nnodes(dg) - 1)
            u_k = get_node_vars(u, equations, dg, i, j, k, element)
            Ja3_node_k = get_contravariant_vector(3, contravariant_vectors, i, j, k,
                                                  element)
            Ja3_avg = 0.5f0 * (Ja3_node_0 + Ja3_node_k)

            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                phi_jump = volume_flux_noncons(u_0, u_k, Ja3_avg, equations,
                                               NonConservativeJump(), noncons)

                for v in eachvariable(equations)
                    # The factor of 2 is missing on each term because Trixi multiplies all the non-cons terms with 0.5
                    fhat3_R[v, i, j, k] -= phi[v, noncons, i, j, k] * phi_jump[v]
                    fhat3_L[v, i, j, k + 1] -= phi[v, noncons, i, j, k] * phi_jump[v]
                end
            end
        end
        u_N = get_node_vars(u, equations, dg, i, j, nnodes(dg), element)
        Ja3_node_N = get_contravariant_vector(3, contravariant_vectors, i, j,
                                              nnodes(dg),
                                              element)
        Ja3_avg = 0.5f0 * (Ja3_node_0 + Ja3_node_N)

        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            phi_jump = volume_flux_noncons(u_0, u_N, Ja3_avg, equations,
                                           NonConservativeJump(), noncons)

            for v in eachvariable(equations)
                # The factor of 2 is missing cause Trixi multiplies all the non-cons terms with 0.5
                fhat3_R[v, i, j, nnodes(dg)] -= phi[v, noncons, i, j, nnodes(dg)] *
                                                phi_jump[v]
            end
        end
    end

    return nothing
end
end # @muladd
