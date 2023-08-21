# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function create_cache(mesh::TreeMesh{2}, equations,
                      volume_integral::VolumeIntegralSubcellLimiting, dg::DG, uEltype)
    cache = create_cache(mesh, equations,
                         VolumeIntegralPureLGLFiniteVolume(volume_integral.volume_flux_fv),
                         dg, uEltype)

    A3dp1_x = Array{uEltype, 3}
    A3dp1_y = Array{uEltype, 3}
    A3d = Array{uEltype, 3}

    fhat1_threaded = A3dp1_x[A3dp1_x(undef, nvariables(equations), nnodes(dg) + 1,
                                     nnodes(dg)) for _ in 1:Threads.nthreads()]
    fhat2_threaded = A3dp1_y[A3dp1_y(undef, nvariables(equations), nnodes(dg),
                                     nnodes(dg) + 1) for _ in 1:Threads.nthreads()]
    flux_temp_threaded = A3d[A3d(undef, nvariables(equations), nnodes(dg), nnodes(dg))
                             for _ in 1:Threads.nthreads()]

    antidiffusive_fluxes = Trixi.ContainerAntidiffusiveFlux2D{uEltype}(0,
                                                                       nvariables(equations),
                                                                       nnodes(dg))

    return (; cache..., antidiffusive_fluxes, fhat1_threaded, fhat2_threaded,
            flux_temp_threaded)
end

function calc_volume_integral!(du, u,
                               mesh::TreeMesh{2},
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

@inline function subcell_limiting_kernel!(du, u,
                                          element, mesh::TreeMesh{2},
                                          nonconservative_terms::False, equations,
                                          volume_integral, limiter::SubcellLimiterIDP,
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
    calcflux_antidiffusive!(fhat1, fhat2, fstar1_L, fstar2_L, u, mesh,
                            nonconservative_terms, equations, limiter, dg, element,
                            cache)

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
@inline function calcflux_fhat!(fhat1, fhat2, u,
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
    fhat1[:, 1, :] .= zero(eltype(fhat1))
    fhat1[:, nnodes(dg) + 1, :] .= zero(eltype(fhat1))

    for j in eachnode(dg), i in 1:(nnodes(dg) - 1), v in eachvariable(equations)
        fhat1[v, i + 1, j] = fhat1[v, i, j] + weights[i] * flux_temp[v, i, j]
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
    fhat2[:, :, 1] .= zero(eltype(fhat2))
    fhat2[:, :, nnodes(dg) + 1] .= zero(eltype(fhat2))

    for j in 1:(nnodes(dg) - 1), i in eachnode(dg), v in eachvariable(equations)
        fhat2[v, i, j + 1] = fhat2[v, i, j] + weights[j] * flux_temp[v, i, j]
    end

    return nothing
end

# Calculate the antidiffusive flux `antidiffusive_flux` as the subtraction between `fhat` and `fstar`.
@inline function calcflux_antidiffusive!(fhat1, fhat2, fstar1, fstar2, u, mesh,
                                         nonconservative_terms, equations,
                                         limiter::SubcellLimiterIDP, dg, element, cache)
    @unpack antidiffusive_flux1, antidiffusive_flux2 = cache.antidiffusive_fluxes

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

    antidiffusive_flux1[:, 1, :, element] .= zero(eltype(antidiffusive_flux1))
    antidiffusive_flux1[:, nnodes(dg) + 1, :, element] .= zero(eltype(antidiffusive_flux1))

    antidiffusive_flux2[:, :, 1, element] .= zero(eltype(antidiffusive_flux2))
    antidiffusive_flux2[:, :, nnodes(dg) + 1, element] .= zero(eltype(antidiffusive_flux2))

    return nothing
end
end # @muladd
