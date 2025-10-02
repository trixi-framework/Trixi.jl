# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function create_cache(mesh::P4estMesh{3},
                      equations, volume_integral::VolumeIntegralSubcellLimiting,
                      dg::DG, uEltype)
    cache = create_cache(mesh, equations,
                         VolumeIntegralPureLGLFiniteVolume(volume_integral.volume_flux_fv),
                         dg, uEltype)

    A4dp1_x = Array{uEltype, 4}
    A4dp1_y = Array{uEltype, 4}
    A4dp1_z = Array{uEltype, 4}
    A4d = Array{uEltype, 4}
    A5d = Array{uEltype, 5}

    fhat1_L_threaded = A4dp1_x[A4dp1_x(undef, nvariables(equations), nnodes(dg) + 1,
                                       nnodes(dg), nnodes(dg))
                               for _ in 1:Threads.nthreads()]
    fhat1_R_threaded = A4dp1_x[A4dp1_x(undef, nvariables(equations), nnodes(dg) + 1,
                                       nnodes(dg), nnodes(dg))
                               for _ in 1:Threads.nthreads()]
    fhat2_L_threaded = A4dp1_y[A4dp1_y(undef, nvariables(equations), nnodes(dg),
                                       nnodes(dg) + 1, nnodes(dg))
                               for _ in 1:Threads.nthreads()]
    fhat2_R_threaded = A4dp1_y[A4dp1_y(undef, nvariables(equations), nnodes(dg),
                                       nnodes(dg) + 1, nnodes(dg))
                               for _ in 1:Threads.nthreads()]
    fhat3_L_threaded = A4dp1_z[A4dp1_z(undef, nvariables(equations), nnodes(dg),
                                       nnodes(dg), nnodes(dg) + 1)
                               for _ in 1:Threads.nthreads()]
    fhat3_R_threaded = A4dp1_z[A4dp1_z(undef, nvariables(equations), nnodes(dg),
                                       nnodes(dg), nnodes(dg) + 1)
                               for _ in 1:Threads.nthreads()]
    flux_temp_threaded = A4d[A4d(undef, nvariables(equations), nnodes(dg), nnodes(dg),
                                 nnodes(dg))
                             for _ in 1:Threads.nthreads()]
    fhat_temp_threaded = A4d[A4d(undef, nvariables(equations), nnodes(dg),
                                 nnodes(dg), nnodes(dg)) for _ in 1:Threads.nthreads()]
    antidiffusive_fluxes = ContainerAntidiffusiveFlux3D{uEltype}(0,
                                                                 nvariables(equations),
                                                                 nnodes(dg))

    if have_nonconservative_terms(equations) == true
        error("Unsupported system of equations with nonconservative terms")
    end

    return (; cache..., antidiffusive_fluxes,
            fhat1_L_threaded, fhat1_R_threaded, fhat2_L_threaded, fhat2_R_threaded,
            fhat3_L_threaded, fhat3_R_threaded, flux_temp_threaded, fhat_temp_threaded)
end

@inline function subcell_limiting_kernel!(du, u, element,
                                          mesh::P4estMesh{3},
                                          nonconservative_terms, equations,
                                          volume_integral, limiter::SubcellLimiterIDP,
                                          dg::DGSEM, cache)
    @unpack inverse_weights = dg.basis
    @unpack volume_flux_dg, volume_flux_fv = volume_integral

    # high-order DG fluxes
    @unpack fhat1_L_threaded, fhat1_R_threaded, fhat2_L_threaded, fhat2_R_threaded, fhat3_L_threaded, fhat3_R_threaded = cache

    fhat1_L = fhat1_L_threaded[Threads.threadid()]
    fhat1_R = fhat1_R_threaded[Threads.threadid()]
    fhat2_L = fhat2_L_threaded[Threads.threadid()]
    fhat2_R = fhat2_R_threaded[Threads.threadid()]
    fhat3_L = fhat3_L_threaded[Threads.threadid()]
    fhat3_R = fhat3_R_threaded[Threads.threadid()]
    calcflux_fhat!(fhat1_L, fhat1_R, fhat2_L, fhat2_R, fhat3_L, fhat3_R, u, mesh,
                   nonconservative_terms, equations, volume_flux_dg, dg, element,
                   cache)

    # low-order FV fluxes
    @unpack fstar1_L_threaded, fstar1_R_threaded, fstar2_L_threaded, fstar2_R_threaded, fstar3_L_threaded, fstar3_R_threaded = cache

    fstar1_L = fstar1_L_threaded[Threads.threadid()]
    fstar1_R = fstar1_R_threaded[Threads.threadid()]
    fstar2_L = fstar2_L_threaded[Threads.threadid()]
    fstar2_R = fstar2_R_threaded[Threads.threadid()]
    fstar3_L = fstar3_L_threaded[Threads.threadid()]
    fstar3_R = fstar3_R_threaded[Threads.threadid()]
    calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, fstar3_L, fstar3_R, u, mesh,
                 nonconservative_terms, equations, volume_flux_fv, dg, element,
                 cache)

    # antidiffusive flux
    calcflux_antidiffusive!(fhat1_L, fhat1_R, fhat2_L, fhat2_R, fhat3_L, fhat3_R,
                            fstar1_L, fstar1_R, fstar2_L, fstar2_R, fstar3_L, fstar3_R,
                            u, mesh, nonconservative_terms, equations, limiter, dg,
                            element, cache)

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

# Calculate the antidiffusive flux `antidiffusive_flux` as the subtraction between `fhat` and `fstar` for conservative systems.
@inline function calcflux_antidiffusive!(fhat1_L, fhat1_R, fhat2_L, fhat2_R,
                                         fhat3_L, fhat3_R,
                                         fstar1_L, fstar1_R, fstar2_L, fstar2_R,
                                         fstar3_L, fstar3_R,
                                         u, mesh::P4estMesh{3},
                                         nonconservative_terms::False, equations,
                                         limiter::SubcellLimiterIDP, dg, element, cache)
    @unpack antidiffusive_flux1_L, antidiffusive_flux1_R, antidiffusive_flux2_L, antidiffusive_flux2_R, antidiffusive_flux3_L, antidiffusive_flux3_R = cache.antidiffusive_fluxes

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
