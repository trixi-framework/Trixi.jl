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
                                u, mesh::TreeMesh{3},
                                have_nonconservative_terms::False, equations,
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

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)

        # All diagonal entries of `derivative_split` are zero. Thus, we can skip
        # the computation of the diagonal terms. In addition, we use the symmetry
        # of the `volume_flux` to save half of the possible two-point flux
        # computations.
        for ii in (i + 1):nnodes(dg)
            u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
            flux1 = volume_flux(u_node, u_node_ii, 1, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[i, ii], flux1,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[ii, i], flux1,
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
        for jj in (j + 1):nnodes(dg)
            u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
            flux2 = volume_flux(u_node, u_node_jj, 2, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[j, jj], flux2,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[jj, j], flux2,
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
        for kk in (k + 1):nnodes(dg)
            u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
            flux3 = volume_flux(u_node, u_node_kk, 3, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[k, kk], flux3,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[kk, k], flux3,
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
end # @muladd
