# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Cache creation for VolumeIntegralFiniteVolume
# Thread-local storage for the n+1 interface fluxes within each block.
# Slots 1 and n+1 (the element-boundary interfaces) stay zero; they are
# handled by the surface integral.
function create_cache(mesh::TreeMesh{1}, equations,
                      volume_integral::VolumeIntegralFiniteVolume,
                      dg::BlockFV, cache_containers, uEltype)
    n = nnodes(dg)
    MA = MArray{Tuple{nvariables(equations), n + 1}, uEltype, 2,
                nvariables(equations) * (n + 1)}
    fstar_threaded = [MA(undef) for _ in 1:Threads.maxthreadid()]
    for fstar in fstar_threaded
        fstar[:, 1] .= zero(uEltype)
        fstar[:, n + 1] .= zero(uEltype)
    end
    return (; fstar_threaded)
end

#####################################################################
# Volume integral: FV flux differences at internal faces
# The update for reference-element cell i is:
# du[i] += inv_h * (fstar[i+1] - fstar[i])
# where inv_h = n/2 = 1/h_ref (uniform cell size h_ref = 2/n).
# Boundary slots fstar[1] and fstar[n+1] are kept zero so that the
# surface integral can add the element-boundary fluxes separately.
function calc_volume_integral!(backend::Nothing, du, u,
                               mesh::TreeMesh{1},
                               have_nonconservative_terms::False, equations,
                               volume_integral::VolumeIntegralFiniteVolume,
                               dg::BlockFV, cache)
    @unpack surface_flux = volume_integral
    @unpack fstar_threaded = cache
    inv_h = nnodes(dg) * one(eltype(u)) / 2  # = 1 / h_ref

    @threaded for element in eachelement(dg, cache)
        fstar = fstar_threaded[Threads.threadid()]

        # Fluxes at internal interfaces i + 1/2 for i = 1, ..., n-1
        for i in 2:nnodes(dg)
            u_ll = get_node_vars(u, equations, dg, i - 1, element)
            u_rr = get_node_vars(u, equations, dg, i, element)
            f = surface_flux(u_ll, u_rr, 1, equations)
            set_node_vars!(fstar, f, equations, dg, i)
        end

        # Apply flux differences to du (boundary slots are zero)
        for i in eachnode(dg)
            for v in eachvariable(equations)
                du[v, i, element] = du[v, i, element] +
                                    inv_h * (fstar[v, i + 1] - fstar[v, i])
            end
        end
    end

    return nothing
end

#####################################################################
# Surface integral: element-boundary fluxes added to the boundary cells
# After apply_jacobian! multiplies by -inverse_jacobian, the combined
# volume + surface contribution gives the correct FV flux-difference update
# for every cell, including the outermost ones.
function calc_surface_integral!(backend::Nothing, du, u,
                                mesh::TreeMesh{1},
                                equations, surface_integral::SurfaceIntegralWeakForm,
                                dg::BlockFV, cache)
    @unpack surface_flux_values = cache.elements
    inv_h = nnodes(dg) * one(eltype(du)) / 2  # = n/2 = 1/h_ref

    @threaded for element in eachelement(dg, cache)
        for v in eachvariable(equations)
            # Left element boundary (direction 1 = -x)
            du[v, 1, element] = du[v, 1, element] -
                                inv_h * surface_flux_values[v, 1, element]
            # Right element boundary (direction 2 = +x)
            du[v, nnodes(dg), element] = du[v, nnodes(dg), element] +
                                         inv_h * surface_flux_values[v, 2, element]
        end
    end

    return nothing
end

#####################################################################
# Integrate a function over the domain using FV quadrature
function integrate_via_indices(func::Func, u,
                               mesh::TreeMesh{1}, equations,
                               dg::BlockFV, cache, args...;
                               normalize = true) where {Func}
    @unpack weights = dg.basis

    integral = zero(func(u, 1, 1, equations, dg, args...))

    @batch reduction=(+, integral) for element in eachelement(dg, cache)
        volume_jacobian_ = volume_jacobian(element, mesh, cache)
        for i in eachnode(dg)
            integral += volume_jacobian_ * weights[i] *
                        func(u, i, element, equations, dg, args...)
        end
    end

    if normalize
        integral = integral / total_volume(mesh)
    end

    return integral
end

#####################################################################
# Compute discrete L2 and L∞ error norms
# No polynomial interpolation is needed; the solution is a cell average at
# each FV cell center, so we evaluate the exact solution there directly.
function calc_error_norms(func, u, t, analyzer,
                          mesh::TreeMesh{1}, equations, initial_condition,
                          dg::BlockFV, cache, cache_analysis)
    @unpack weights = dg.basis
    @unpack node_coordinates = cache.elements

    l2_error = zero(func(get_node_vars(u, equations, dg, 1, 1), equations))
    linf_error = copy(l2_error)

    for element in eachelement(dg, cache)
        volume_jacobian_ = volume_jacobian(element, mesh, cache)

        for i in eachnode(dg)
            x = get_node_coords(node_coordinates, equations, dg, i, element)
            u_exact = initial_condition(x, t, equations)
            diff = func(u_exact, equations) -
                   func(get_node_vars(u, equations, dg, i, element), equations)
            l2_error += diff .^ 2 * (weights[i] * volume_jacobian_)
            linf_error = @. max(linf_error, abs(diff))
        end
    end

    total_volume_ = total_volume(mesh)
    l2_error = @. sqrt(l2_error / total_volume_)

    return l2_error, linf_error
end
end # @muladd
