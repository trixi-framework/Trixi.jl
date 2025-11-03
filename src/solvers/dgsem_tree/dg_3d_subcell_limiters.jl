# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function calc_mortar_weights(equations::AbstractEquations{3},
                             basis::LobattoLegendreBasis, RealT;
                             basis_function = :piecewise_constant)
    n_nodes = nnodes(basis)
    mortar_weights = zeros(RealT, n_nodes, n_nodes, n_nodes, n_nodes, 4) # [node_i (large), node_j (large), node_i (small), node_j (small), small element]
    mortar_weights_sums = zeros(RealT, n_nodes, n_nodes, 2) # [node_i, node_j, small (1) / large (2) element]

    if basis_function == :piecewise_constant
        calc_mortar_weights_piecewise_constant!(equations, mortar_weights, n_nodes,
                                                RealT)
        # elseif basis_function == :piecewise_linear
        #     calc_mortar_weights_piecewise_linear!(equations, mortar_weights, basis)
    else
        error("Unsupported basis function type: $basis_function")
    end

    # Sums of mortar weights for normalization
    for j in eachnode(basis), i in eachnode(basis)
        for l in eachnode(basis), k in eachnode(basis)
            # Add weights from large element to small element
            # Sums for all small elements are equal due to symmetry
            mortar_weights_sums[i, j, 1] += mortar_weights[k, l, i, j, 1]
            # Add weights from small element to large element
            for small_element in 1:4
                mortar_weights_sums[i, j, 2] += mortar_weights[i, j, k, l,
                                                               small_element]
            end
        end
    end

    return mortar_weights, mortar_weights_sums
end

function calc_mortar_weights_piecewise_constant!(equations::AbstractEquations{3},
                                                 mortar_weights, n_nodes, RealT)
    _, weights = gauss_lobatto_nodes_weights(n_nodes, RealT)

    # Local mortar weights are of the form: `w_(ij, kl) = int_S psi_(ij) phi_(kl) ds`,
    # where `psi_(ij)` are the basis functions of the large element and `phi_(kl)` are the basis
    # functions of the small element. `S` is the face connecting both elements.
    # We use piecewise constant basis functions on the LGL subgrid. So, only focus on interval,
    # where both basis functions are non-zero. `interval = [left_bound_x, right_bound_x] x [left_bound_y, right_bound_y]`.
    # `w_(ij, kl) = int_S psi_(ij) phi_(kl) ds = int_(interval) ds = (right_bound_x - left_bound_x) * (right_bound_y - left_bound_y)`.
    # The bounds in each direction are independent and can be computed separately analogously to the 2D case:
    # `right_bound = min(left_bound_large, left_bound_small)`
    # `left_bound = max(right_bound_large, right_bound_small)`
    # If `right_bound <= left_bound`, i.e., both intervals don't overlap, then `w_ij = 0`.

    # Due to the LGL subgrid, the interval bounds are cumulative LGL quadrature weights.
    cum_weights_large = [zero(RealT); cumsum(weights)] .- 1 # on [-1, 1]
    cum_weights_lower = 0.5f0 * cum_weights_large .- 0.5f0  # on [-1, 0]
    cum_weights_upper = cum_weights_lower .+ 1              # on [0, 1]
    # So, for `w_(ij, kl)` we have
    # `right_bound_x = min(cum_weights_large[i], cum_weights_small[k])`
    # `left_bound_x = max(cum_weights_large[i+1], cum_weights_small[k+1])`
    # `right_bound_y = min(cum_weights_large[j], cum_weights_small[l])`
    # `left_bound_y = max(cum_weights_large[j+1], cum_weights_small[l+1])`

    # Illustration of the positions in 3D, where ξ and η are the local coordinates
    # of the mortar element, which are precisely the local coordinates that span
    # the surface of the smaller side.
    # Note that the orientation in the physical space is completely irrelevant here.
    #   ┌─────────────┬─────────────┐  ┌───────────────────────────┐
    #   │             │             │  │                           │
    #   │    small    │    small    │  │                           │
    #   │      3      │      4      │  │                           │
    #   │             │             │  │           large           │
    #   ├─────────────┼─────────────┤  │             5             │
    # η │             │             │  │                           │
    #   │    small    │    small    │  │                           │
    # ↑ │      1      │      2      │  │                           │
    # │ │             │             │  │                           │
    # │ └─────────────┴─────────────┘  └───────────────────────────┘
    # │
    # ⋅────> ξ

    for j in 1:n_nodes, i in 1:n_nodes
        for l in 1:n_nodes, k in 1:n_nodes
            # 1st small and large element element
            left_x = max(cum_weights_large[i], cum_weights_lower[k])
            right_x = min(cum_weights_large[i + 1], cum_weights_lower[k + 1])
            left_y = max(cum_weights_large[j], cum_weights_lower[l])
            right_y = min(cum_weights_large[j + 1], cum_weights_lower[l + 1])

            # Local weight of 0 if intervals do not overlap, i.e., `right <= left`
            if right_x > left_x && right_y > left_y
                mortar_weights[i, j, k, l, 1] = (right_x - left_x) * (right_y - left_y)
            end

            # 2nd small and large element
            left_x = max(cum_weights_large[i], cum_weights_upper[k])
            right_x = min(cum_weights_large[i + 1], cum_weights_upper[k + 1])
            left_y = max(cum_weights_large[j], cum_weights_lower[l])
            right_y = min(cum_weights_large[j + 1], cum_weights_lower[l + 1])
            if right_x > left_x && right_y > left_y
                mortar_weights[i, j, k, l, 2] = (right_x - left_x) * (right_y - left_y)
            end

            # 3rd small and large element
            left_x = max(cum_weights_large[i], cum_weights_lower[k])
            right_x = min(cum_weights_large[i + 1], cum_weights_lower[k + 1])
            left_y = max(cum_weights_large[j], cum_weights_upper[l])
            right_y = min(cum_weights_large[j + 1], cum_weights_upper[l + 1])
            if right_x > left_x && right_y > left_y
                mortar_weights[i, j, k, l, 3] = (right_x - left_x) * (right_y - left_y)
            end

            # 4th small and large element
            left_x = max(cum_weights_large[i], cum_weights_upper[k])
            right_x = min(cum_weights_large[i + 1], cum_weights_upper[k + 1])
            left_y = max(cum_weights_large[j], cum_weights_upper[l])
            right_y = min(cum_weights_large[j + 1], cum_weights_upper[l + 1])
            if right_x > left_x && right_y > left_y
                mortar_weights[i, j, k, l, 4] = (right_x - left_x) * (right_y - left_y)
            end
        end
    end

    return mortar_weights
end

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
    fhat1_L[:, 1, :, :] .= zero(eltype(fhat1_L))
    fhat1_L[:, nnodes(dg) + 1, :, :] .= zero(eltype(fhat1_L))
    fhat1_R[:, 1, :, :] .= zero(eltype(fhat1_R))
    fhat1_R[:, nnodes(dg) + 1, :, :] .= zero(eltype(fhat1_R))

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
    fhat2_L[:, :, 1, :] .= zero(eltype(fhat2_L))
    fhat2_L[:, :, nnodes(dg) + 1, :] .= zero(eltype(fhat2_L))
    fhat2_R[:, :, 1, :] .= zero(eltype(fhat2_R))
    fhat2_R[:, :, nnodes(dg) + 1, :] .= zero(eltype(fhat2_R))

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
    fhat3_L[:, :, :, 1] .= zero(eltype(fhat3_L))
    fhat3_L[:, :, :, nnodes(dg) + 1] .= zero(eltype(fhat3_L))
    fhat3_R[:, :, :, 1] .= zero(eltype(fhat3_R))
    fhat3_R[:, :, :, nnodes(dg) + 1] .= zero(eltype(fhat3_R))

    for k in 1:(nnodes(dg) - 1), j in eachnode(dg), i in eachnode(dg),
        v in eachvariable(equations)

        fhat3_L[v, i, j, k + 1] = fhat3_L[v, i, j, k] +
                                  weights[k] * flux_temp[v, i, j, k]
        fhat3_R[v, i, j, k + 1] = fhat3_L[v, i, j, k + 1]
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
                                u, mesh::TreeMesh{2},
                                have_nonconservative_terms::True, equations,
                                volume_flux::Tuple{F_CONS, F_NONCONS}, dg::DGSEM,
                                element,
                                cache) where {
                                              F_CONS <: Function,
                                              F_NONCONS <:
                                              FluxNonConservative{NonConservativeSymmetric()}
                                              }
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

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)

        # All diagonal entries of `derivative_split` are zero. Thus, we can skip
        # the computation of the diagonal terms. In addition, we use the symmetry
        # of `volume_flux_cons` and `volume_flux_noncons` to save half of the possible two-point flux
        # computations.
        for ii in (i + 1):nnodes(dg)
            u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
            flux1 = volume_flux_cons(u_node, u_node_ii, 1, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[i, ii], flux1,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[ii, i], flux1,
                                       equations, dg, ii, j, k)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux1_noncons = volume_flux_noncons(u_node, u_node_ii, 1, equations,
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
    fhat1_L[:, 1, :, :] .= zero(eltype(fhat1_L))
    fhat1_L[:, nnodes(dg) + 1, :, :] .= zero(eltype(fhat1_L))
    fhat1_R[:, 1, :, :] .= zero(eltype(fhat1_R))
    fhat1_R[:, nnodes(dg) + 1, :, :] .= zero(eltype(fhat1_R))

    fhat_temp[:, 1, :, :] .= zero(eltype(fhat1_L))
    fhat_noncons_temp[:, :, 1, :, :] .= zero(eltype(fhat1_L))

    # Compute local contribution to non-conservative flux
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_local = get_node_vars(u, equations, dg, i, j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, 1, equations,
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
        for jj in (j + 1):nnodes(dg)
            u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
            flux2 = volume_flux_cons(u_node, u_node_jj, 2, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[j, jj], flux2,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[jj, j], flux2,
                                       equations, dg, i, jj, k)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux2_noncons = volume_flux_noncons(u_node, u_node_jj, 2, equations,
                                                    NonConservativeSymmetric(), noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5 * derivative_split[j, jj],
                                           flux2_noncons,
                                           equations, dg, noncons, i, j, k)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5 * derivative_split[jj, j],
                                           flux2_noncons,
                                           equations, dg, noncons, i, jj, k)
            end
        end
    end

    # FV-form flux `fhat` in y direction
    fhat2_L[:, :, 1, :] .= zero(eltype(fhat2_L))
    fhat2_L[:, :, nnodes(dg) + 1, :] .= zero(eltype(fhat2_L))
    fhat2_R[:, :, 1, :] .= zero(eltype(fhat2_R))
    fhat2_R[:, :, nnodes(dg) + 1, :] .= zero(eltype(fhat2_R))

    fhat_temp[:, :, 1, :] .= zero(eltype(fhat1_L))
    fhat_noncons_temp[:, :, :, 1, :] .= zero(eltype(fhat1_L))

    # Compute local contribution to non-conservative flux
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_local = get_node_vars(u, equations, dg, i, j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, 2, equations,
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
        for kk in (k + 1):nnodes(dg)
            u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
            flux3 = volume_flux_cons(u_node, u_node_kk, 3, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[k, kk], flux3,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[kk, k], flux3,
                                       equations, dg, i, j, kk)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux3_noncons = volume_flux_noncons(u_node, u_node_kk, 3, equations,
                                                    NonConservativeSymmetric(), noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5 * derivative_split[k, kk],
                                           flux3_noncons,
                                           equations, dg, noncons, i, j, k)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5 * derivative_split[kk, k],
                                           flux3_noncons,
                                           equations, dg, noncons, i, j, kk)
            end
        end
    end

    # FV-form flux `fhat` in z direction
    fhat3_L[:, :, :, 1] .= zero(eltype(fhat3_L))
    fhat3_L[:, :, :, nnodes(dg) + 1] .= zero(eltype(fhat3_L))
    fhat3_R[:, :, :, 1] .= zero(eltype(fhat3_R))
    fhat3_R[:, :, :, nnodes(dg) + 1] .= zero(eltype(fhat3_R))

    fhat_temp[:, :, :, 1] .= zero(eltype(fhat1_L))
    fhat_noncons_temp[:, :, :, :, 1] .= zero(eltype(fhat1_L))

    # Compute local contribution to non-conservative flux
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_local = get_node_vars(u, equations, dg, i, j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, 3, equations,
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
@inline function calcflux_fhat!(fhat1_L, fhat1_R, fhat2_L, fhat2_R, fhat3_L, fhat3_R,
                                u, mesh::TreeMesh{2},
                                nonconservative_terms::True, equations,
                                volume_flux::Tuple{F_CONS, F_NONCONS}, dg::DGSEM,
                                element,
                                cache) where {
                                              F_CONS <: Function,
                                              F_NONCONS <:
                                              FluxNonConservative{NonConservativeJump()}
                                              }
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

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)

        # All diagonal entries of `derivative_split` are zero. Thus, we can skip
        # the computation of the diagonal terms. In addition, we use the symmetry
        # of `volume_flux_cons` and skew-symmetry of `volume_flux_noncons` to save half of the possible two-point flux
        # computations.
        for ii in (i + 1):nnodes(dg)
            u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
            flux1 = volume_flux_cons(u_node, u_node_ii, 1, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[i, ii], flux1,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[ii, i], flux1,
                                       equations, dg, ii, j, k)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux1_noncons = volume_flux_noncons(u_node, u_node_ii, 1, equations,
                                                    NonConservativeJump(),
                                                    noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5f0 * derivative_split[i, ii],
                                           flux1_noncons,
                                           equations, dg, noncons, i, j, k)
                # Note the sign flip due to skew-symmetry when argument order is swapped
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           -0.5f0 * derivative_split[ii, i],
                                           flux1_noncons,
                                           equations, dg, noncons, ii, j, k)
            end
        end
    end

    # FV-form flux `fhat` in x direction
    fhat1_L[:, 1, :, :] .= zero(eltype(fhat1_L))
    fhat1_L[:, nnodes(dg) + 1, :, :] .= zero(eltype(fhat1_L))
    fhat1_R[:, 1, :, :] .= zero(eltype(fhat1_R))
    fhat1_R[:, nnodes(dg) + 1, :, :] .= zero(eltype(fhat1_R))

    fhat_temp[:, 1, :, :] .= zero(eltype(fhat1_L))
    fhat_noncons_temp[:, :, 1, :, :] .= zero(eltype(fhat1_L))

    # Compute local contribution to non-conservative flux
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_local = get_node_vars(u, equations, dg, i, j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, 1, equations,
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
        for i in 2:(nnodes(dg) - 1)
            u_i = get_node_vars(u, equations, dg, i, j, k, element)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                phi_jump = volume_flux_noncons(u_0, u_i, 1, equations,
                                               NonConservativeJump(), noncons)

                for v in eachvariable(equations)
                    # The factor of 2 is missing on each term because Trixi multiplies all the non-cons terms with 0.5
                    fhat1_R[v, i, j, k] -= phi[v, noncons, i, j, k] * phi_jump[v]
                    fhat1_L[v, i + 1, j, k] -= phi[v, noncons, i, j, k] * phi_jump[v]
                end
            end
        end
        u_N = get_node_vars(u, equations, dg, nnodes(dg), j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            phi_jump = volume_flux_noncons(u_0, u_N, 1, equations,
                                           NonConservativeJump(), noncons)

            for v in eachvariable(equations)
                # The factor of 2 is missing because Trixi multiplies all the non-cons terms with 0.5
                fhat1_R[v, nnodes(dg), j, k] -= phi[v, noncons, nnodes(dg), j, k] *
                                                phi_jump[v]
            end
        end
    end

    ########

    # Split form volume flux in orientation 2: y direction
    flux_temp .= zero(eltype(flux_temp))
    flux_noncons_temp .= zero(eltype(flux_noncons_temp))

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)
        for jj in (j + 1):nnodes(dg)
            u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
            flux2 = volume_flux_cons(u_node, u_node_jj, 2, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[j, jj], flux2,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[jj, j], flux2,
                                       equations, dg, i, jj, k)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux2_noncons = volume_flux_noncons(u_node, u_node_jj, 2, equations,
                                                    NonConservativeJump(),
                                                    noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5 * derivative_split[j, jj],
                                           flux2_noncons,
                                           equations, dg, noncons, i, j, k)
                # Note the sign flip due to skew-symmetry when argument order is swapped
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           -0.5 * derivative_split[jj, j],
                                           flux2_noncons,
                                           equations, dg, noncons, i, jj, k)
            end
        end
    end

    # FV-form flux `fhat` in y direction
    fhat2_L[:, :, 1, :] .= zero(eltype(fhat2_L))
    fhat2_L[:, :, nnodes(dg) + 1, :] .= zero(eltype(fhat2_L))
    fhat2_R[:, :, 1, :] .= zero(eltype(fhat2_R))
    fhat2_R[:, :, nnodes(dg) + 1, :] .= zero(eltype(fhat2_R))

    fhat_temp[:, :, 1, :] .= zero(eltype(fhat1_L))
    fhat_noncons_temp[:, :, :, 1, :] .= zero(eltype(fhat1_L))

    # Compute local contribution to non-conservative flux
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_local = get_node_vars(u, equations, dg, i, j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, 2, equations,
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
        for j in 2:(nnodes(dg) - 1)
            u_j = get_node_vars(u, equations, dg, i, j, k, element)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                phi_jump = volume_flux_noncons(u_0, u_j, 2, equations,
                                               NonConservativeJump(), noncons)

                for v in eachvariable(equations)
                    # The factor of 2 is missing on each term because Trixi multiplies all the non-cons terms with 0.5
                    fhat2_R[v, i, j, k] -= phi[v, noncons, i, j, k] * phi_jump[v]
                    fhat2_L[v, i, j + 1, k] -= phi[v, noncons, i, j, k] * phi_jump[v]
                end
            end
        end
        u_N = get_node_vars(u, equations, dg, i, nnodes(dg), k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            phi_jump = volume_flux_noncons(u_0, u_N, 2, equations,
                                           NonConservativeJump(), noncons)

            for v in eachvariable(equations)
                # The factor of 2 is missing cause Trixi multiplies all the non-cons terms with 0.5
                fhat2_R[v, i, nnodes(dg), k] -= phi[v, noncons, i, nnodes(dg), k] *
                                                phi_jump[v]
            end
        end
    end

    ########

    # Split form volume flux in orientation 3: z direction
    flux_temp .= zero(eltype(flux_temp))
    flux_noncons_temp .= zero(eltype(flux_noncons_temp))

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)
        for kk in (k + 1):nnodes(dg)
            u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
            flux3 = volume_flux_cons(u_node, u_node_kk, 3, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[k, kk], flux3,
                                       equations, dg, i, j, k)
            multiply_add_to_node_vars!(flux_temp, derivative_split[kk, k], flux3,
                                       equations, dg, i, j, kk)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux3_noncons = volume_flux_noncons(u_node, u_node_kk, 3, equations,
                                                    NonConservativeJump(),
                                                    noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5 * derivative_split[k, kk],
                                           flux3_noncons,
                                           equations, dg, noncons, i, j, k)
                # Note the sign flip due to skew-symmetry when argument order is swapped
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           -0.5 * derivative_split[kk, k],
                                           flux3_noncons,
                                           equations, dg, noncons, i, j, kk)
            end
        end
    end

    # FV-form flux `fhat` in y direction
    fhat3_L[:, :, :, 1] .= zero(eltype(fhat3_L))
    fhat3_L[:, :, :, nnodes(dg) + 1] .= zero(eltype(fhat3_L))
    fhat3_R[:, :, :, 1] .= zero(eltype(fhat3_R))
    fhat3_R[:, :, :, nnodes(dg) + 1] .= zero(eltype(fhat3_R))

    fhat_temp[:, :, :, 1] .= zero(eltype(fhat1_L))
    fhat_noncons_temp[:, :, :, :, 1] .= zero(eltype(fhat1_L))

    # Compute local contribution to non-conservative flux
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_local = get_node_vars(u, equations, dg, i, j, k, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            set_node_vars!(phi,
                           volume_flux_noncons(u_local, 3, equations,
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
        for k in 2:(nnodes(dg) - 1)
            u_k = get_node_vars(u, equations, dg, i, j, k, element)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                phi_jump = volume_flux_noncons(u_0, u_k, 2, equations,
                                               NonConservativeJump(), noncons)

                for v in eachvariable(equations)
                    # The factor of 2 is missing on each term because Trixi multiplies all the non-cons terms with 0.5
                    fhat3_R[v, i, j, k] -= phi[v, noncons, i, j, k] * phi_jump[v]
                    fhat3_L[v, i, j, k + 1] -= phi[v, noncons, i, j, k] * phi_jump[v]
                end
            end
        end
        u_N = get_node_vars(u, equations, dg, i, j, nnodes(dg), element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            phi_jump = volume_flux_noncons(u_0, u_N, 3, equations,
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

function prolong2mortars!(cache, u, mesh::TreeMesh{3}, equations,
                          mortar_idp::LobattoLegendreMortarIDP, dg::DGSEM)
    prolong2mortars!(cache, u, mesh, equations, mortar_idp.mortar_l2, dg)

    # The data of both small elements were already copied to the mortar cache
    @threaded for mortar in eachmortar(dg, cache)
        large_element = cache.mortars.neighbor_ids[5, mortar]

        # Copy solutions
        if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
            if cache.mortars.orientations[mortar] == 1
                # IDP mortars in x-direction
                for k in eachnode(dg), j in eachnode(dg)
                    for v in eachvariable(equations)
                        cache.mortars.u_large[v, j, k, mortar] = u[v, nnodes(dg), j, k,
                                                                   large_element]
                    end
                end
            elseif cache.mortars.orientations[mortar] == 2
                # IDP mortars in y-direction
                for k in eachnode(dg), i in eachnode(dg)
                    for v in eachvariable(equations)
                        cache.mortars.u_large[v, i, k, mortar] = u[v, i, nnodes(dg), k,
                                                                   large_element]
                    end
                end
            else
                # IDP mortars in z-direction
                for j in eachnode(dg), i in eachnode(dg)
                    for v in eachvariable(equations)
                        cache.mortars.u_large[v, i, j, mortar] = u[v, i, j, nnodes(dg),
                                                                   large_element]
                    end
                end
            end
        else # large_sides[mortar] == 2 -> small elements on left side
            if cache.mortars.orientations[mortar] == 1
                # IDP mortars in x-direction
                for k in eachnode(dg), j in eachnode(dg)
                    for v in eachvariable(equations)
                        cache.mortars.u_large[v, j, k, mortar] = u[v, 1, j, k,
                                                                   large_element]
                    end
                end
            elseif cache.mortars.orientations[mortar] == 2
                # IDP mortars in y-direction
                for k in eachnode(dg), i in eachnode(dg)
                    for v in eachvariable(equations)
                        cache.mortars.u_large[v, i, k, mortar] = u[v, i, 1, k,
                                                                   large_element]
                    end
                end
            else
                # IDP mortars in z-direction
                for j in eachnode(dg), i in eachnode(dg)
                    for v in eachvariable(equations)
                        cache.mortars.u_large[v, i, j, mortar] = u[v, i, j, 1,
                                                                   large_element]
                    end
                end
            end
        end
    end

    return nothing
end

function calc_mortar_flux_low_order!(surface_flux_values,
                                     mesh::TreeMesh{3},
                                     nonconservative_terms::False, equations,
                                     mortar_idp::LobattoLegendreMortarIDP,
                                     surface_integral, dg::DG, cache)
    @unpack surface_flux = surface_integral
    @unpack u_lower_left, u_lower_right, u_upper_left, u_upper_right, u_large, orientations = cache.mortars
    (; mortar_weights, mortar_weights_sums) = mortar_idp

    @assert mortar_idp.local_factor "Local factor must be true for low-order mortar fluxes."

    @threaded for mortar in eachmortar(dg, cache)
        lower_left_element = cache.mortars.neighbor_ids[1, mortar]
        lower_right_element = cache.mortars.neighbor_ids[2, mortar]
        upper_left_element = cache.mortars.neighbor_ids[3, mortar]
        upper_right_element = cache.mortars.neighbor_ids[4, mortar]
        large_element = cache.mortars.neighbor_ids[5, mortar]

        # Calculate fluxes
        orientation = orientations[mortar]

        if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
            if orientation == 1
                # L2 mortars in x-direction
                direction_small = 1
                direction_large = 2
            elseif orientation == 2
                # L2 mortars in y-direction
                direction_small = 3
                direction_large = 4
            else
                # L2 mortars in z-direction
                direction_small = 5
                direction_large = 6
            end
            small_side = 2
        else # large_sides[mortar] == 2 -> small elements on left side
            if orientation == 1
                # L2 mortars in x-direction
                direction_small = 2
                direction_large = 1
            elseif orientation == 2
                # L2 mortars in y-direction
                direction_small = 4
                direction_large = 3
            else
                # L2 mortars in z-direction
                direction_small = 6
                direction_large = 5
            end
            small_side = 1
        end

        surface_flux_values[:, :, :, direction_small, lower_left_element] .= zero(eltype(surface_flux_values))
        surface_flux_values[:, :, :, direction_small, lower_right_element] .= zero(eltype(surface_flux_values))
        surface_flux_values[:, :, :, direction_small, upper_left_element] .= zero(eltype(surface_flux_values))
        surface_flux_values[:, :, :, direction_small, upper_right_element] .= zero(eltype(surface_flux_values))
        surface_flux_values[:, :, :, direction_large, large_element] .= zero(eltype(surface_flux_values))
        # Lower left element
        for j in eachnode(dg), i in eachnode(dg)
            u_lower_left_local = get_surface_node_vars(u_lower_left, equations, dg,
                                                       i, j, mortar)[small_side]
            for l in eachnode(dg), k in eachnode(dg)
                factor = mortar_weights[k, l, i, j, 1]
                if isapprox(factor, zero(typeof(factor)))
                    continue
                end
                u_large_local = get_node_vars(u_large, equations, dg, k, l, mortar)

                if small_side == 2 # -> small elements on right side
                    flux = surface_flux(u_large_local, u_lower_left_local, orientation,
                                        equations)
                else # small_side == 1 -> small elements on left side
                    flux = surface_flux(u_lower_left_local, u_large_local, orientation,
                                        equations)
                end

                # Lower left element
                multiply_add_to_node_vars!(surface_flux_values,
                                           factor /
                                           mortar_weights_sums[i, j, 1],
                                           flux, equations, dg,
                                           i, j, direction_small, lower_left_element)
                # Large element
                multiply_add_to_node_vars!(surface_flux_values,
                                           factor /
                                           mortar_weights_sums[k, l, 2],
                                           flux, equations, dg,
                                           k, l, direction_large, large_element)
            end
        end
        # Lower right element
        for j in eachnode(dg), i in eachnode(dg)
            u_lower_right_local = get_surface_node_vars(u_lower_right, equations, dg,
                                                        i, j, mortar)[small_side]
            for l in eachnode(dg), k in eachnode(dg)
                factor = mortar_weights[k, l, i, j, 2]
                if isapprox(factor, zero(typeof(factor)))
                    continue
                end
                u_large_local = get_node_vars(u_large, equations, dg, k, l, mortar)

                if small_side == 2 # -> small elements on right side
                    flux = surface_flux(u_large_local, u_lower_right_local, orientation,
                                        equations)
                else # small_side == 1 -> small elements on left side
                    flux = surface_flux(u_lower_right_local, u_large_local, orientation,
                                        equations)
                end

                # Lower right element
                multiply_add_to_node_vars!(surface_flux_values,
                                           factor /
                                           mortar_weights_sums[i, j, 1],
                                           flux, equations, dg,
                                           i, j, direction_small, lower_right_element)
                # Large element
                multiply_add_to_node_vars!(surface_flux_values,
                                           factor /
                                           mortar_weights_sums[k, l, 2],
                                           flux, equations, dg,
                                           k, l, direction_large, large_element)
            end
        end
        # Upper left element
        for j in eachnode(dg), i in eachnode(dg)
            u_upper_left_local = get_surface_node_vars(u_upper_left, equations, dg,
                                                       i, j, mortar)[small_side]
            for l in eachnode(dg), k in eachnode(dg)
                factor = mortar_weights[k, l, i, j, 3]
                if isapprox(factor, zero(typeof(factor)))
                    continue
                end
                u_large_local = get_node_vars(u_large, equations, dg, k, l, mortar)

                if small_side == 2 # -> small elements on right side
                    flux = surface_flux(u_large_local, u_upper_left_local, orientation,
                                        equations)
                else # small_side == 1 -> small elements on left side
                    flux = surface_flux(u_upper_left_local, u_large_local, orientation,
                                        equations)
                end

                # Upper left element
                multiply_add_to_node_vars!(surface_flux_values,
                                           factor /
                                           mortar_weights_sums[i, j, 1],
                                           flux, equations, dg,
                                           i, j, direction_small, upper_left_element)
                # Large element
                multiply_add_to_node_vars!(surface_flux_values,
                                           factor /
                                           mortar_weights_sums[k, l, 2],
                                           flux, equations, dg,
                                           k, l, direction_large, large_element)
            end
        end
        # Upper right element
        for j in eachnode(dg), i in eachnode(dg)
            u_upper_right_local = get_surface_node_vars(u_upper_right, equations, dg,
                                                        i, j, mortar)[small_side]
            for l in eachnode(dg), k in eachnode(dg)
                factor = mortar_weights[k, l, i, j, 4]
                if isapprox(factor, zero(typeof(factor)))
                    continue
                end
                u_large_local = get_node_vars(u_large, equations, dg, k, l, mortar)

                if small_side == 2 # -> small elements on right side
                    flux = surface_flux(u_large_local, u_upper_right_local, orientation,
                                        equations)
                else # small_side == 1 -> small elements on left side
                    flux = surface_flux(u_upper_right_local, u_large_local, orientation,
                                        equations)
                end

                # Upper right element
                multiply_add_to_node_vars!(surface_flux_values,
                                           factor /
                                           mortar_weights_sums[i, j, 1],
                                           flux, equations, dg,
                                           i, j, direction_small, upper_right_element)
                # Large element
                multiply_add_to_node_vars!(surface_flux_values,
                                           factor /
                                           mortar_weights_sums[k, l, 2],
                                           flux, equations, dg,
                                           k, l, direction_large, large_element)
            end
        end
    end

    return nothing
end
end # @muladd
