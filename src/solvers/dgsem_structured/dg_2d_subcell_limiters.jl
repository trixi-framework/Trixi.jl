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
@inline function calcflux_fhat!(fhat1_L, fhat1_R, fhat2_L, fhat2_R, u,
                                mesh::Union{StructuredMesh{2}, P4estMesh{2}},
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

    for j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, element)

        # pull the contravariant vectors in each coordinate direction
        Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, element) # x direction

        # All diagonal entries of `derivative_split` are zero. Thus, we can skip
        # the computation of the diagonal terms. In addition, we use the symmetry
        # of the `volume_flux` to save half of the possible two-point flux
        # computations.

        # x direction
        for ii in (i + 1):nnodes(dg)
            u_node_ii = get_node_vars(u, equations, dg, ii, j, element)
            # pull the contravariant vectors and compute the average
            Ja1_node_ii = get_contravariant_vector(1, contravariant_vectors, ii, j,
                                                   element)
            Ja1_avg = 0.5f0 * (Ja1_node + Ja1_node_ii)

            # compute the contravariant sharp flux in the direction of the averaged contravariant vector
            fluxtilde1 = volume_flux(u_node, u_node_ii, Ja1_avg, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[i, ii], fluxtilde1,
                                       equations, dg, i, j)
            multiply_add_to_node_vars!(flux_temp, derivative_split[ii, i], fluxtilde1,
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

        # pull the contravariant vectors in each coordinate direction
        Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, element)

        # y direction
        for jj in (j + 1):nnodes(dg)
            u_node_jj = get_node_vars(u, equations, dg, i, jj, element)
            # pull the contravariant vectors and compute the average
            Ja2_node_jj = get_contravariant_vector(2, contravariant_vectors, i, jj,
                                                   element)
            Ja2_avg = 0.5f0 * (Ja2_node + Ja2_node_jj)
            # compute the contravariant sharp flux in the direction of the averaged contravariant vector
            fluxtilde2 = volume_flux(u_node, u_node_jj, Ja2_avg, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[j, jj], fluxtilde2,
                                       equations, dg, i, j)
            multiply_add_to_node_vars!(flux_temp, derivative_split[jj, j], fluxtilde2,
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

@inline function calc_lambdas_bar_states!(u, t,
                                          mesh::Union{StructuredMesh{2}, P4estMesh{2}},
                                          nonconservative_terms, equations, limiter,
                                          dg, cache, boundary_conditions;
                                          calc_bar_states = true)
    if limiter isa SubcellLimiterIDP && !limiter.bar_states
        return nothing
    end
    (; lambda1, lambda2, bar_states1, bar_states2) = limiter.cache.container_bar_states
    (; normal_direction_xi, normal_direction_eta) = limiter.cache.container_bar_states

    # Calc lambdas and bar states inside elements
    @threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in 2:nnodes(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            u_node_im1 = get_node_vars(u, equations, dg, i - 1, j, element)

            normal_direction = get_node_coords(normal_direction_xi, equations, dg,
                                               i - 1, j, element)

            lambda1[i, j, element] = max_abs_speed_naive(u_node_im1, u_node,
                                                         normal_direction, equations)

            !calc_bar_states && continue

            flux1 = flux(u_node, normal_direction, equations)
            flux1_im1 = flux(u_node_im1, normal_direction, equations)
            for v in eachvariable(equations)
                bar_states1[v, i, j, element] = 0.5 * (u_node[v] + u_node_im1[v]) -
                                                0.5 * (flux1[v] - flux1_im1[v]) /
                                                lambda1[i, j, element]
            end
        end

        for j in 2:nnodes(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            u_node_jm1 = get_node_vars(u, equations, dg, i, j - 1, element)

            normal_direction = get_node_coords(normal_direction_eta, equations, dg, i,
                                               j - 1, element)

            lambda2[i, j, element] = max_abs_speed_naive(u_node_jm1, u_node,
                                                         normal_direction, equations)

            !calc_bar_states && continue

            flux2 = flux(u_node, normal_direction, equations)
            flux2_jm1 = flux(u_node_jm1, normal_direction, equations)
            for v in eachvariable(equations)
                bar_states2[v, i, j, element] = 0.5 * (u_node[v] + u_node_jm1[v]) -
                                                0.5 * (flux2[v] - flux2_jm1[v]) /
                                                lambda2[i, j, element]
            end
        end
    end

    calc_lambdas_bar_states_interface!(u, t, limiter, boundary_conditions, mesh,
                                       equations,
                                       dg, cache; calc_bar_states = calc_bar_states)

    return nothing
end

@inline function calc_lambdas_bar_states_interface!(u, t, limiter, boundary_conditions,
                                                    mesh::StructuredMesh{2}, equations,
                                                    dg, cache; calc_bar_states = true)
    (; contravariant_vectors) = cache.elements
    (; lambda1, lambda2, bar_states1, bar_states2) = limiter.cache.container_bar_states

    # Calc lambdas and bar states at interfaces and periodic boundaries
    @threaded for element in eachelement(dg, cache)
        # Get neighboring element ids
        left = cache.elements.left_neighbors[1, element]
        lower = cache.elements.left_neighbors[2, element]

        if left != 0
            for i in eachnode(dg)
                u_left = get_node_vars(u, equations, dg, nnodes(dg), i, left)
                u_element = get_node_vars(u, equations, dg, 1, i, element)

                Ja1 = get_contravariant_vector(1, contravariant_vectors, 1, i, element)
                lambda = max_abs_speed_naive(u_left, u_element, Ja1, equations)

                lambda1[nnodes(dg) + 1, i, left] = lambda
                lambda1[1, i, element] = lambda

                !calc_bar_states && continue

                flux_left = flux(u_left, Ja1, equations)
                flux_element = flux(u_element, Ja1, equations)
                bar_state = 0.5 * (u_element + u_left) -
                            0.5 * (flux_element - flux_left) / lambda
                for v in eachvariable(equations)
                    bar_states1[v, nnodes(dg) + 1, i, left] = bar_state[v]
                    bar_states1[v, 1, i, element] = bar_state[v]
                end
            end
        end
        if lower != 0
            for i in eachnode(dg)
                u_lower = get_node_vars(u, equations, dg, i, nnodes(dg), lower)
                u_element = get_node_vars(u, equations, dg, i, 1, element)

                Ja2 = get_contravariant_vector(2, contravariant_vectors, i, 1, element)
                lambda = max_abs_speed_naive(u_lower, u_element, Ja2, equations)

                lambda2[i, nnodes(dg) + 1, lower] = lambda
                lambda2[i, 1, element] = lambda

                !calc_bar_states && continue

                flux_lower = flux(u_lower, Ja2, equations)
                flux_element = flux(u_element, Ja2, equations)
                bar_state = 0.5 * (u_element + u_lower) -
                            0.5 * (flux_element - flux_lower) / lambda
                for v in eachvariable(equations)
                    bar_states2[v, i, nnodes(dg) + 1, lower] = bar_state[v]
                    bar_states2[v, i, 1, element] = bar_state[v]
                end
            end
        end
    end

    # Calc lambdas and bar states at physical boundaries
    if isperiodic(mesh)
        return nothing
    end
    linear_indices = LinearIndices(size(mesh))
    if !isperiodic(mesh, 1)
        # - xi direction
        for cell_y in axes(mesh, 2)
            element = linear_indices[begin, cell_y]
            for j in eachnode(dg)
                Ja1 = get_contravariant_vector(1, contravariant_vectors, 1, j, element)
                u_inner = get_node_vars(u, equations, dg, 1, j, element)
                u_outer = get_boundary_outer_state(u_inner, t,
                                                   boundary_conditions[1], Ja1, 1,
                                                   mesh, equations, dg, cache,
                                                   1, j, element)
                lambda1[1, j, element] = max_abs_speed_naive(u_inner, u_outer, Ja1,
                                                             equations)

                !calc_bar_states && continue

                flux_inner = flux(u_inner, Ja1, equations)
                flux_outer = flux(u_outer, Ja1, equations)
                for v in eachvariable(equations)
                    bar_states1[v, 1, j, element] = 0.5 * (u_inner[v] + u_outer[v]) -
                                                    0.5 *
                                                    (flux_inner[v] - flux_outer[v]) /
                                                    lambda1[1, j, element]
                end
            end
        end
        # + xi direction
        for cell_y in axes(mesh, 2)
            element = linear_indices[end, cell_y]
            for j in eachnode(dg)
                Ja1 = get_contravariant_vector(1, contravariant_vectors, nnodes(dg), j,
                                               element)
                u_inner = get_node_vars(u, equations, dg, nnodes(dg), j, element)
                u_outer = get_boundary_outer_state(u_inner, t,
                                                   boundary_conditions[2], Ja1, 2,
                                                   mesh, equations, dg, cache,
                                                   nnodes(dg), j, element)
                lambda1[nnodes(dg) + 1, j, element] = max_abs_speed_naive(u_inner,
                                                                          u_outer, Ja1,
                                                                          equations)

                !calc_bar_states && continue

                flux_inner = flux(u_inner, Ja1, equations)
                flux_outer = flux(u_outer, Ja1, equations)
                for v in eachvariable(equations)
                    bar_states1[v, nnodes(dg) + 1, j, element] = 0.5 * (u_inner[v] +
                                                                  u_outer[v]) -
                                                                 0.5 *
                                                                 (flux_outer[v] -
                                                                  flux_inner[v]) /
                                                                 lambda1[nnodes(dg) + 1,
                                                                         j, element]
                end
            end
        end
    end
    if !isperiodic(mesh, 2)
        # - eta direction
        for cell_x in axes(mesh, 1)
            element = linear_indices[cell_x, begin]
            for i in eachnode(dg)
                Ja2 = get_contravariant_vector(2, contravariant_vectors, i, 1, element)
                u_inner = get_node_vars(u, equations, dg, i, 1, element)
                u_outer = get_boundary_outer_state(u_inner, t,
                                                   boundary_conditions[3], Ja2, 3,
                                                   mesh, equations, dg, cache,
                                                   i, 1, element)
                lambda2[i, 1, element] = max_abs_speed_naive(u_inner, u_outer, Ja2,
                                                             equations)

                !calc_bar_states && continue

                flux_inner = flux(u_inner, Ja2, equations)
                flux_outer = flux(u_outer, Ja2, equations)
                for v in eachvariable(equations)
                    bar_states2[v, i, 1, element] = 0.5 * (u_inner[v] + u_outer[v]) -
                                                    0.5 *
                                                    (flux_inner[v] - flux_outer[v]) /
                                                    lambda2[i, 1, element]
                end
            end
        end
        # + eta direction
        for cell_x in axes(mesh, 1)
            element = linear_indices[cell_x, end]
            for i in eachnode(dg)
                Ja2 = get_contravariant_vector(2, contravariant_vectors, i, nnodes(dg),
                                               element)
                u_inner = get_node_vars(u, equations, dg, i, nnodes(dg), element)
                u_outer = get_boundary_outer_state(u_inner, t,
                                                   boundary_conditions[4], Ja2, 4,
                                                   mesh, equations, dg, cache,
                                                   i, nnodes(dg), element)
                lambda2[i, nnodes(dg) + 1, element] = max_abs_speed_naive(u_inner,
                                                                          u_outer, Ja2,
                                                                          equations)

                !calc_bar_states && continue

                flux_inner = flux(u_inner, Ja2, equations)
                flux_outer = flux(u_outer, Ja2, equations)
                for v in eachvariable(equations)
                    bar_states2[v, i, nnodes(dg) + 1, element] = 0.5 * (u_outer[v] +
                                                                  u_inner[v]) -
                                                                 0.5 *
                                                                 (flux_outer[v] -
                                                                  flux_inner[v]) /
                                                                 lambda2[i,
                                                                         nnodes(dg) + 1,
                                                                         element]
                end
            end
        end
    end

    return nothing
end
end # @muladd
