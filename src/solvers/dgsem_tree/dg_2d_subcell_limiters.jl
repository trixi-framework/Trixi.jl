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

    A3d = Array{uEltype, 3}

    fhat1_L_threaded = A3d[A3d(undef, nvariables(equations),
                               nnodes(dg) + 1, nnodes(dg))
                           for _ in 1:Threads.maxthreadid()]
    fhat2_L_threaded = A3d[A3d(undef, nvariables(equations),
                               nnodes(dg), nnodes(dg) + 1)
                           for _ in 1:Threads.maxthreadid()]
    fhat1_R_threaded = A3d[A3d(undef, nvariables(equations),
                               nnodes(dg) + 1, nnodes(dg))
                           for _ in 1:Threads.maxthreadid()]
    fhat2_R_threaded = A3d[A3d(undef, nvariables(equations),
                               nnodes(dg), nnodes(dg) + 1)
                           for _ in 1:Threads.maxthreadid()]

    flux_temp_threaded = A3d[A3d(undef, nvariables(equations),
                                 nnodes(dg), nnodes(dg))
                             for _ in 1:Threads.maxthreadid()]
    fhat_temp_threaded = A3d[A3d(undef, nvariables(equations),
                                 nnodes(dg), nnodes(dg))
                             for _ in 1:Threads.maxthreadid()]

    antidiffusive_fluxes = ContainerAntidiffusiveFlux2D{uEltype}(0,
                                                                 nvariables(equations),
                                                                 nnodes(dg))

    if have_nonconservative_terms(equations) == true
        A4d = Array{uEltype, 4}

        # Extract the nonconservative flux as a dispatch argument for `n_nonconservative_terms`
        _, volume_flux_noncons = volume_integral.volume_flux_dg

        flux_nonconservative_temp_threaded = A4d[A4d(undef, nvariables(equations),
                                                     n_nonconservative_terms(volume_flux_noncons),
                                                     nnodes(dg), nnodes(dg))
                                                 for _ in 1:Threads.maxthreadid()]
        fhat_nonconservative_temp_threaded = A4d[A4d(undef, nvariables(equations),
                                                     n_nonconservative_terms(volume_flux_noncons),
                                                     nnodes(dg), nnodes(dg))
                                                 for _ in 1:Threads.maxthreadid()]
        phi_threaded = A4d[A4d(undef, nvariables(equations),
                               n_nonconservative_terms(volume_flux_noncons),
                               nnodes(dg), nnodes(dg))
                           for _ in 1:Threads.maxthreadid()]
        cache = (; cache..., flux_nonconservative_temp_threaded,
                 fhat_nonconservative_temp_threaded, phi_threaded)
    end

    return (; cache..., antidiffusive_fluxes,
            fhat1_L_threaded, fhat2_L_threaded, fhat1_R_threaded, fhat2_R_threaded,
            flux_temp_threaded, fhat_temp_threaded)
end

function calc_mortar_weights(equations::AbstractEquations{2},
                             basis::LobattoLegendreBasis, RealT;
                             basis_function = :piecewise_constant)
    n_nodes = nnodes(basis)
    mortar_weights = zeros(RealT, n_nodes, n_nodes, 2) # [node_i (large element), node_i (small element), small element]
    mortar_weights_sums = zeros(RealT, n_nodes, 2)     # [node, left (=1) or large (=2) element]

    if basis_function == :piecewise_constant
        calc_mortar_weights_piecewise_constant!(equations, mortar_weights, n_nodes,
                                                RealT)
    elseif basis_function == :piecewise_linear
        calc_mortar_weights_piecewise_linear!(equations, mortar_weights, basis)
    else
        error("Unsupported basis function type: $basis_function")
    end

    # Sums of mortar weights for normalization
    for i in eachnode(basis)
        for k in eachnode(basis)
            # Add weights from large element to small element
            # Sums for both small elements are equal due to symmetry
            mortar_weights_sums[i, 1] += mortar_weights[k, i, 1]
            # Add weights from small element to large element
            for small_element in 1:2
                mortar_weights_sums[i, 2] += mortar_weights[i, k, small_element]
            end
        end
    end

    return mortar_weights, mortar_weights_sums
end

function calc_mortar_weights_piecewise_constant!(equations::AbstractEquations{2},
                                                 mortar_weights, n_nodes, RealT)
    _, weights = gauss_lobatto_nodes_weights(n_nodes, RealT)

    # Local mortar weights are of the form: `w_ij = int_S psi_i phi_j ds`,
    # where `psi_i` are the basis functions of the large element and `phi_j` are the basis
    # functions of the small element. `S` is the face connecting both elements.
    # We use piecewise constant basis functions on the LGL subgrid. So, only focus on interval,
    # where both basis functions are non-zero. `interval = [left_bound, right_bound]`.
    # `w_ij = int_S psi_i phi_j ds = int_{left_bound}^{right_bound} ds = right_bound - left_bound`.
    # `right_bound = min(left_bound_large, left_bound_small)`
    # `left_bound = max(right_bound_large, right_bound_small)`
    # If `right_bound <= left_bound`, i.e., both intervals don't overlap, then `w_ij = 0`.

    # Due to the LGL subgrid, the interval bounds are cumulative LGL quadrature weights.
    cum_weights_large = [zero(RealT); cumsum(weights)] .- 1 # on [-1, 1]
    cum_weights_lower = 0.5f0 * cum_weights_large .- 0.5f0  # on [-1, 0]
    cum_weights_upper = cum_weights_lower .+ 1              # on [0, 1]
    # So, for `w_ij` we have
    # `right_bound = min(cum_weights_large[i], cum_weights_small[j])`
    # `left_bound = max(cum_weights_large[i+1], cum_weights_small[j+1])`

    for j in 1:n_nodes, i in 1:n_nodes
        # lower and large element element
        left = max(cum_weights_large[i], cum_weights_lower[j])
        right = min(cum_weights_large[i + 1], cum_weights_lower[j + 1])

        # Local weight of 0 if intervals do not overlap, i.e., `right <= left`
        if right > left
            mortar_weights[i, j, 1] = right - left
        end

        # upper and large element
        left = max(cum_weights_large[i], cum_weights_upper[j])
        right = min(cum_weights_large[i + 1], cum_weights_upper[j + 1])
        if right > left
            mortar_weights[i, j, 2] = right - left
        end
    end

    return mortar_weights
end

function calc_mortar_weights_piecewise_linear!(equations::AbstractEquations{2},
                                               mortar_weights, basis)
    (; nodes) = basis
    n_nodes = nnodes(basis)
    nodes_lower = 0.5f0 * nodes .- 0.5f0
    nodes_upper = nodes_lower .+ 1.0f0

    function_product(xi, ii, i, jj, j, nodes_u_l) = (xi^3 / 3 -
                                                     xi^2 / 2 *
                                                     (nodes[ii] + nodes_u_l[jj]) +
                                                     nodes[ii] * nodes_u_l[jj] * xi) /
                                                    ((nodes[ii] - nodes[i]) *
                                                     (nodes_u_l[jj] - nodes_u_l[j]))
    for i in eachnode(basis)
        for j in eachnode(basis)
            # left part of large element function
            if i > 1
                # left part of small element function
                if j > 1
                    # basis function of left element
                    interval_left = max(nodes[i - 1], nodes_lower[j - 1])
                    interval_right = min(nodes[i], nodes_lower[j])
                    if interval_left < interval_right
                        mortar_weights[i, j, 1] += function_product(interval_right,
                                                                    i - 1, i, j - 1, j,
                                                                    nodes_lower) -
                                                   function_product(interval_left,
                                                                    i - 1, i, j - 1, j,
                                                                    nodes_lower)
                    end
                    # basis function of right element
                    interval_left = max(nodes[i - 1], nodes_upper[j - 1])
                    interval_right = min(nodes[i], nodes_upper[j])
                    if interval_left < interval_right
                        mortar_weights[i, j, 2] += function_product(interval_right,
                                                                    i - 1, i, j - 1, j,
                                                                    nodes_upper) -
                                                   function_product(interval_left,
                                                                    i - 1, i, j - 1, j,
                                                                    nodes_upper)
                    end
                end
                # right part of small element function
                if j < n_nodes
                    # basis function of left element
                    interval_left = max(nodes[i - 1], nodes_lower[j])
                    interval_right = min(nodes[i], nodes_lower[j + 1])
                    if interval_left < interval_right
                        mortar_weights[i, j, 1] += function_product(interval_right,
                                                                    i - 1, i, j + 1, j,
                                                                    nodes_lower) -
                                                   function_product(interval_left,
                                                                    i - 1, i, j + 1, j,
                                                                    nodes_lower)
                    end
                    # basis function of right element
                    interval_left = max(nodes[i - 1], nodes_upper[j])
                    interval_right = min(nodes[i], nodes_upper[j + 1])
                    if interval_left < interval_right
                        mortar_weights[i, j, 2] += function_product(interval_right,
                                                                    i - 1, i, j + 1, j,
                                                                    nodes_upper) -
                                                   function_product(interval_left,
                                                                    i - 1, i, j + 1, j,
                                                                    nodes_upper)
                    end
                end
            end
            # right part of large element function
            if i < n_nodes
                # left part of small element function
                if j > 1
                    # basis function of left element
                    interval_left = max(nodes[i], nodes_lower[j - 1])
                    interval_right = min(nodes[i + 1], nodes_lower[j])
                    if interval_left < interval_right
                        mortar_weights[i, j, 1] += function_product(interval_right,
                                                                    i + 1, i, j - 1, j,
                                                                    nodes_lower) -
                                                   function_product(interval_left,
                                                                    i + 1, i, j - 1, j,
                                                                    nodes_lower)
                    end
                    # basis function of right element
                    interval_left = max(nodes[i], nodes_upper[j - 1])
                    interval_right = min(nodes[i + 1], nodes_upper[j])
                    if interval_left < interval_right
                        mortar_weights[i, j, 2] += function_product(interval_right,
                                                                    i + 1, i, j - 1, j,
                                                                    nodes_upper) -
                                                   function_product(interval_left,
                                                                    i + 1, i, j - 1, j,
                                                                    nodes_upper)
                    end
                end
                # right part of small element function
                if j < n_nodes
                    # basis function of left element
                    interval_left = max(nodes[i], nodes_lower[j])
                    interval_right = min(nodes[i + 1], nodes_lower[j + 1])
                    if interval_left < interval_right
                        mortar_weights[i, j, 1] += function_product(interval_right,
                                                                    i + 1, i, j + 1, j,
                                                                    nodes_lower) -
                                                   function_product(interval_left,
                                                                    i + 1, i, j + 1, j,
                                                                    nodes_lower)
                    end
                    # basis function of right element
                    interval_left = max(nodes[i], nodes_upper[j])
                    interval_right = min(nodes[i + 1], nodes_upper[j + 1])
                    if interval_left < interval_right
                        mortar_weights[i, j, 2] += function_product(interval_right,
                                                                    i + 1, i, j + 1, j,
                                                                    nodes_upper) -
                                                   function_product(interval_left,
                                                                    i + 1, i, j + 1, j,
                                                                    nodes_upper)
                    end
                end
            end
        end
    end

    return mortar_weights
end

# Subcell limiting currently only implemented for certain mesh types
function calc_volume_integral!(du, u,
                               mesh::Union{TreeMesh{2}, StructuredMesh{2},
                                           P4estMesh{2}},
                               have_nonconservative_terms, equations,
                               volume_integral::VolumeIntegralSubcellLimiting,
                               dg::DGSEM, cache)
    @unpack limiter = volume_integral

    @threaded for element in eachelement(dg, cache)
        subcell_limiting_kernel!(du, u, element, mesh,
                                 have_nonconservative_terms, equations,
                                 volume_integral, limiter,
                                 dg, cache)
    end

    return nothing
end

@inline function subcell_limiting_kernel!(du, u, element,
                                          mesh::Union{TreeMesh{2}, StructuredMesh{2},
                                                      P4estMesh{2}},
                                          have_nonconservative_terms, equations,
                                          volume_integral, limiter::SubcellLimiterIDP,
                                          dg::DGSEM, cache)
    @unpack inverse_weights = dg.basis # Plays role of inverse DG-subcell sizes
    @unpack volume_flux_dg, volume_flux_fv = volume_integral

    # high-order DG fluxes
    @unpack fhat1_L_threaded, fhat1_R_threaded, fhat2_L_threaded, fhat2_R_threaded = cache

    fhat1_L = fhat1_L_threaded[Threads.threadid()]
    fhat1_R = fhat1_R_threaded[Threads.threadid()]
    fhat2_L = fhat2_L_threaded[Threads.threadid()]
    fhat2_R = fhat2_R_threaded[Threads.threadid()]
    calcflux_fhat!(fhat1_L, fhat1_R, fhat2_L, fhat2_R, u, mesh,
                   have_nonconservative_terms, equations, volume_flux_dg, dg, element,
                   cache)

    # low-order FV fluxes
    @unpack fstar1_L_threaded, fstar1_R_threaded, fstar2_L_threaded, fstar2_R_threaded = cache

    fstar1_L = fstar1_L_threaded[Threads.threadid()]
    fstar2_L = fstar2_L_threaded[Threads.threadid()]
    fstar1_R = fstar1_R_threaded[Threads.threadid()]
    fstar2_R = fstar2_R_threaded[Threads.threadid()]
    calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u, mesh,
                 have_nonconservative_terms, equations, volume_flux_fv, dg, element,
                 cache)

    # antidiffusive flux
    calcflux_antidiffusive!(fhat1_L, fhat1_R, fhat2_L, fhat2_R,
                            fstar1_L, fstar1_R, fstar2_L, fstar2_R,
                            u, mesh, have_nonconservative_terms, equations, limiter, dg,
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
                                mesh::TreeMesh{2}, have_nonconservative_terms::False,
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
@inline function calcflux_fhat!(fhat1_L, fhat1_R, fhat2_L, fhat2_R, u,
                                mesh::TreeMesh{2}, have_nonconservative_terms::True,
                                equations,
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
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
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
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
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
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons),
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
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
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
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
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
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons),
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

# Calculate the DG staggered volume fluxes `fhat` in subcell FV-form inside the element
# (**with non-conservative terms in "local * jump" form**).
#
# See also `flux_differencing_kernel!`.
#
# The calculation of the non-conservative staggered "fluxes" requires non-conservative
# terms that can be written as a product of local and jump contributions.
@inline function calcflux_fhat!(fhat1_L, fhat1_R, fhat2_L, fhat2_R, u,
                                mesh::TreeMesh{2}, nonconservative_terms::True,
                                equations,
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

    for j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, element)

        # All diagonal entries of `derivative_split` are zero. Thus, we can skip
        # the computation of the diagonal terms. In addition, we use the symmetry
        # of `volume_flux_cons` and skew-symmetry of `volume_flux_noncons` to save half of the possible two-point flux
        # computations.
        for ii in (i + 1):nnodes(dg)
            u_node_ii = get_node_vars(u, equations, dg, ii, j, element)
            flux1 = volume_flux_cons(u_node, u_node_ii, 1, equations)
            multiply_add_to_node_vars!(flux_temp, derivative_split[i, ii], flux1,
                                       equations, dg, i, j)
            multiply_add_to_node_vars!(flux_temp, derivative_split[ii, i], flux1,
                                       equations, dg, ii, j)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux1_noncons = volume_flux_noncons(u_node, u_node_ii, 1, equations,
                                                    NonConservativeJump(),
                                                    noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5f0 * derivative_split[i, ii],
                                           flux1_noncons,
                                           equations, dg, noncons, i, j)
                # Note the sign flip due to skew-symmetry when argument order is swapped
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           -0.5f0 * derivative_split[ii, i],
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
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
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
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons),
            v in eachvariable(equations)

            value = fhat_noncons_temp[v, noncons, i, j] +
                    weights[i] * flux_noncons_temp[v, noncons, i, j]
            fhat_noncons_temp[v, noncons, i + 1, j] = value

            fhat1_L[v, i + 1, j] = fhat1_L[v, i + 1, j] + phi[v, noncons, i, j] * value
            fhat1_R[v, i + 1, j] = fhat1_R[v, i + 1, j] +
                                   phi[v, noncons, i + 1, j] * value
        end
    end

    # Apply correction term to the flux-differencing formula for nonconservative local * jump fluxes.
    for j in eachnode(dg)
        u_0 = get_node_vars(u, equations, dg, 1, j, element)
        for i in 2:(nnodes(dg) - 1)
            u_i = get_node_vars(u, equations, dg, i, j, element)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                phi_jump = volume_flux_noncons(u_0, u_i, 1, equations,
                                               NonConservativeJump(), noncons)

                for v in eachvariable(equations)
                    # The factor of 2 is missing on each term because Trixi multiplies all the non-cons terms with 0.5
                    fhat1_R[v, i, j] -= phi[v, noncons, i, j] * phi_jump[v]
                    fhat1_L[v, i + 1, j] -= phi[v, noncons, i, j] * phi_jump[v]
                end
            end
        end
        u_N = get_node_vars(u, equations, dg, nnodes(dg), j, element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            phi_jump = volume_flux_noncons(u_0, u_N, 1, equations,
                                           NonConservativeJump(), noncons)

            for v in eachvariable(equations)
                # The factor of 2 is missing because Trixi multiplies all the non-cons terms with 0.5
                fhat1_R[v, nnodes(dg), j] -= phi[v, noncons, nnodes(dg), j] *
                                             phi_jump[v]
            end
        end
    end

    ########

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
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux2_noncons = volume_flux_noncons(u_node, u_node_jj, 2, equations,
                                                    NonConservativeJump(),
                                                    noncons)
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           0.5 * derivative_split[j, jj],
                                           flux2_noncons,
                                           equations, dg, noncons, i, j)
                # Note the sign flip due to skew-symmetry when argument order is swapped
                multiply_add_to_node_vars!(flux_noncons_temp,
                                           -0.5 * derivative_split[jj, j],
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
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
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
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons),
            v in eachvariable(equations)

            value = fhat_noncons_temp[v, noncons, i, j] +
                    weights[j] * flux_noncons_temp[v, noncons, i, j]
            fhat_noncons_temp[v, noncons, i, j + 1] = value

            fhat2_L[v, i, j + 1] = fhat2_L[v, i, j + 1] + phi[v, noncons, i, j] * value
            fhat2_R[v, i, j + 1] = fhat2_R[v, i, j + 1] +
                                   phi[v, noncons, i, j + 1] * value
        end
    end

    # Apply correction term to the flux-differencing formula for nonconservative local * jump fluxes.
    for i in eachnode(dg)
        u_0 = get_node_vars(u, equations, dg, i, 1, element)
        for j in 2:(nnodes(dg) - 1)
            u_j = get_node_vars(u, equations, dg, i, j, element)
            for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
                phi_jump = volume_flux_noncons(u_0, u_j, 2, equations,
                                               NonConservativeJump(), noncons)

                for v in eachvariable(equations)
                    # The factor of 2 is missing on each term because Trixi multiplies all the non-cons terms with 0.5
                    fhat2_R[v, i, j] -= phi[v, noncons, i, j] * phi_jump[v]
                    fhat2_L[v, i, j + 1] -= phi[v, noncons, i, j] * phi_jump[v]
                end
            end
        end
        u_N = get_node_vars(u, equations, dg, i, nnodes(dg), element)
        for noncons in 1:n_nonconservative_terms(volume_flux_noncons)
            phi_jump = volume_flux_noncons(u_0, u_N, 2, equations,
                                           NonConservativeJump(), noncons)

            for v in eachvariable(equations)
                # The factor of 2 is missing cause Trixi multiplies all the non-cons terms with 0.5
                fhat2_R[v, i, nnodes(dg)] -= phi[v, noncons, i, nnodes(dg)] *
                                             phi_jump[v]
            end
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
                                         have_nonconservative_terms::False, equations,
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

    return nothing
end

# Calculate the antidiffusive flux `antidiffusive_flux` as the subtraction between `fhat` and `fstar` for conservative systems.
@inline function calcflux_antidiffusive!(fhat1_L, fhat1_R, fhat2_L, fhat2_R,
                                         fstar1_L, fstar1_R, fstar2_L, fstar2_R,
                                         u,
                                         mesh::Union{TreeMesh{2}, StructuredMesh{2},
                                                     P4estMesh{2}},
                                         have_nonconservative_terms::True, equations,
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

    return nothing
end

function prolong2mortars!(cache, u, mesh::TreeMesh{2}, equations,
                          mortar_idp::LobattoLegendreMortarIDP, dg::DGSEM)
    prolong2mortars!(cache, u, mesh, equations, mortar_idp.mortar_l2, dg)

    # The data of both small elements were already copied to the mortar cache
    @threaded for mortar in eachmortar(dg, cache)
        large_element = cache.mortars.neighbor_ids[3, mortar]

        # Copy solutions
        if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
            if cache.mortars.orientations[mortar] == 1
                # IDP mortars in x-direction
                for l in eachnode(dg)
                    for v in eachvariable(equations)
                        cache.mortars.u_large[v, l, mortar] = u[v, nnodes(dg), l,
                                                                large_element]
                    end
                end
            else
                # IDP mortars in y-direction
                for l in eachnode(dg)
                    for v in eachvariable(equations)
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
                        cache.mortars.u_large[v, l, mortar] = u[v, 1, l, large_element]
                    end
                end
            else
                # IDP mortars in y-direction
                for l in eachnode(dg)
                    for v in eachvariable(equations)
                        cache.mortars.u_large[v, l, mortar] = u[v, l, 1, large_element]
                    end
                end
            end
        end
    end

    return nothing
end

function calc_mortar_flux!(surface_flux_values, mesh,
                           nonconservative_terms, equations,
                           mortar_idp::LobattoLegendreMortarIDP, surface_integral,
                           dg::DG, cache)
    # low order fluxes
    @trixi_timeit timer() "calc_mortar_flux_low_order!" calc_mortar_flux_low_order!(surface_flux_values,
                                                                                    mesh,
                                                                                    nonconservative_terms,
                                                                                    equations,
                                                                                    mortar_idp,
                                                                                    surface_integral,
                                                                                    dg,
                                                                                    cache)

    # high order fluxes
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes
    @trixi_timeit timer() "calc_mortar_flux!" calc_mortar_flux!(surface_flux_values_high_order,
                                                                mesh,
                                                                nonconservative_terms,
                                                                equations,
                                                                mortar_idp.mortar_l2,
                                                                dg.surface_integral, dg,
                                                                cache)

    return nothing
end

function calc_mortar_flux_low_order!(surface_flux_values,
                                     mesh::TreeMesh{2},
                                     nonconservative_terms::False, equations,
                                     mortar_idp::LobattoLegendreMortarIDP,
                                     surface_integral, dg::DG, cache)
    @unpack surface_flux = surface_integral
    @unpack u_lower, u_upper, u_large, orientations = cache.mortars
    (; weights) = dg.basis
    (; mortar_weights, mortar_weights_sums, local_factor) = mortar_idp

    @threaded for mortar in eachmortar(dg, cache)
        large_element = cache.mortars.neighbor_ids[3, mortar]
        upper_element = cache.mortars.neighbor_ids[2, mortar]
        lower_element = cache.mortars.neighbor_ids[1, mortar]

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
            small_side = 2
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
            small_side = 1
        end

        surface_flux_values[:, :, direction_small, lower_element] .= zero(eltype(surface_flux_values))
        surface_flux_values[:, :, direction_small, upper_element] .= zero(eltype(surface_flux_values))
        surface_flux_values[:, :, direction_large, large_element] .= zero(eltype(surface_flux_values))
        # Lower element
        for i in eachnode(dg)
            u_lower_local = get_surface_node_vars(u_lower, equations, dg,
                                                  i, mortar)[small_side]
            for k in eachnode(dg)
                factor = mortar_weights[k, i, 1]
                if local_factor && isapprox(factor, zero(typeof(factor)))
                    continue
                end
                u_large_local = get_node_vars(u_large, equations, dg, k, mortar)

                if small_side == 2 # -> small elements on right side
                    flux = surface_flux(u_large_local, u_lower_local, orientation,
                                        equations)
                else # small_side == 1 -> small elements on left side
                    flux = surface_flux(u_lower_local, u_large_local, orientation,
                                        equations)
                end

                if local_factor
                    # Lower element
                    multiply_add_to_node_vars!(surface_flux_values,
                                               factor /
                                               mortar_weights_sums[i, 1],
                                               flux, equations, dg,
                                               i, direction_small, lower_element)
                    # Large element
                    multiply_add_to_node_vars!(surface_flux_values,
                                               factor /
                                               mortar_weights_sums[k, 2],
                                               flux, equations, dg,
                                               k, direction_large, large_element)
                else
                    # Lower element
                    multiply_add_to_node_vars!(surface_flux_values,
                                               0.5f0 * weights[k], flux,
                                               equations, dg, i, direction_small,
                                               lower_element)
                    # Large element
                    multiply_add_to_node_vars!(surface_flux_values,
                                               0.25f0 * weights[i], flux,
                                               equations, dg, k, direction_large,
                                               large_element)
                end
            end
        end
        # Upper element
        for i in eachnode(dg)
            u_upper_local = get_surface_node_vars(u_upper, equations, dg,
                                                  i, mortar)[small_side]
            for k in eachnode(dg)
                factor = mortar_weights[k, i, 2]
                if local_factor && isapprox(factor, zero(typeof(factor)))
                    continue
                end
                u_large_local = get_node_vars(u_large, equations, dg, k, mortar)

                if small_side == 2 # -> small elements on right side
                    flux = surface_flux(u_large_local, u_upper_local, orientation,
                                        equations)
                else # small_side == 1 -> small elements on left side
                    flux = surface_flux(u_upper_local, u_large_local, orientation,
                                        equations)
                end

                if local_factor
                    # Upper element
                    multiply_add_to_node_vars!(surface_flux_values,
                                               factor /
                                               mortar_weights_sums[i, 2],
                                               flux, equations, dg,
                                               i, direction_small, upper_element)
                    # Large element
                    multiply_add_to_node_vars!(surface_flux_values,
                                               factor /
                                               mortar_weights_sums[k, 2],
                                               flux, equations, dg,
                                               k, direction_large, large_element)
                else
                    # Upper element
                    multiply_add_to_node_vars!(surface_flux_values,
                                               0.5f0 * weights[k], flux,
                                               equations, dg, i, direction_small,
                                               upper_element)
                    # Large element
                    multiply_add_to_node_vars!(surface_flux_values,
                                               0.25f0 * weights[i], flux,
                                               equations, dg, k, direction_large,
                                               large_element)
                end
            end
        end
    end

    return nothing
end

@inline function element_solutions_to_mortars!(mortars,
                                               mortar_idp::LobattoLegendreMortarIDPAlternative,
                                               leftright, mortar,
                                               u_large::AbstractArray{<:Any, 2})
    multiply_dimensionwise!(view(mortars.u_upper, leftright, :, :, mortar),
                            mortar_idp.forward_upper_low_order, u_large)
    multiply_dimensionwise!(view(mortars.u_lower, leftright, :, :, mortar),
                            mortar_idp.forward_lower_low_order, u_large)
    return nothing
end

"""
    get_boundary_outer_state(u_inner, t,
                             boundary_condition::BoundaryConditionDirichlet,
                             orientation_or_normal, direction,
                             mesh, equations, dg, cache, indices...)
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
                                          mesh::Union{TreeMesh, StructuredMesh},
                                          equations, dg, cache, indices...)
    (; node_coordinates) = cache.elements

    x = get_node_coords(node_coordinates, equations, dg, indices...)
    u_outer = boundary_condition.boundary_value_function(x, t, equations)

    return u_outer
end
end # @muladd
