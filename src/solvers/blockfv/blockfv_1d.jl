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
                      volume_integral::Union{VolumeIntegralFiniteVolume,
                                             VolumeIntegralFiniteVolumeO2},
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
# Second-order volume integral with reconstructed states at internal faces
function calc_volume_integral!(backend::Nothing, du, u,
                               mesh::TreeMesh{1},
                               have_nonconservative_terms::False, equations,
                               volume_integral::VolumeIntegralFiniteVolumeO2,
                               dg::BlockFVO2, cache)
    @unpack (sc_interface_coords, surface_flux, slope_limiter,
    cons2recon, recon2cons) = volume_integral
    @unpack fstar_threaded = cache
    inv_h = nnodes(dg) * one(eltype(u)) / 2

    @threaded for element in eachelement(dg, cache)
        fstar = fstar_threaded[Threads.threadid()]

        # Each BlockFVO2 element is split into n equal FV cells on [-1, 1].
        # Cell averages live at the cell centers; numerical fluxes are stored
        # in fstar at the n+1 faces (element boundaries + internal faces).
        # Schematic for n_nodes = 4:
        #
        #   ξ = -1                                           ξ = +1
        #        |--- u₁ ---|--- u₂ ---|--- u₃ ---|--- u₄ ---|
        #       f₁         f₂         f₃         f₄         f₅
        #    (bound) (internal faces: sc_interface_coords) (bound)
        #
        # Volume loop only fills internal faces i = 2, ..., n (f₂..f₄ above).
        # Boundary faces f₁ and f₅ are zero until the surface integral adds them.
        #
        # At internal face i (between cells i-1 and i), high-order reconstruction
        # needs up to four neighboring cell averages:
        #
        #            u_ll        u_lr   |   u_rl        u_rr
        #              ·          ·     |     ·          ·
        #           cell i-2   cell i-1 |  cell i    cell i+1
        #                               ^
        #                          face i (flux)
        #
        # Near the element ends the missing neighbor is clamped to the
        # outermost cell (volume-local stencil; no values from other elements).

        for i in 2:nnodes(dg)
            # Four-point stencil, clamped to the element (volume-local)
            u_ll = cons2recon(get_node_vars(u, equations, dg, max(1, i - 2), element),
                              equations)
            u_lr = cons2recon(get_node_vars(u, equations, dg, i - 1, element),
                              equations)
            u_rl = cons2recon(get_node_vars(u, equations, dg, i, element),
                              equations)
            u_rr = cons2recon(get_node_vars(u, equations, dg,
                                            min(nnodes(dg), i + 1), element),
                              equations)

            u_l, u_r = reconstruction_O2(u_ll, u_lr, u_rl, u_rr,
                                         sc_interface_coords, i,
                                         slope_limiter, dg)

            f = surface_flux(recon2cons(u_l, equations),
                             recon2cons(u_r, equations), 1, equations)
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
# Surface reconstruction for BlockFVO2 interfaces or boundaries.
# Reconstruct to element face ξ = +/- 1 using reconstruction_O2,
# then extrapolate from the near-boundary internal face
@inline function reconstruct_element_face(u, equations, dg::BlockFVO2, element, face,
                                          volume_integral)
    @unpack sc_interface_coords, slope_limiter,
    cons2recon, recon2cons = volume_integral
    nodes = dg.basis.nodes
    n = nnodes(dg)

    #the one node case, just return the node value
    if n == 1
        return get_node_vars(u, equations, dg, 1, element)
    end

    # Reconstruct at ξ = ±1 by extrapolating from the nearest internal face.
    #
    #   ξ = -1                                              ξ = +1
    #        |--- u₁ ---|--- u₂ --- ... ---|--- u_n ---|
    #                   f₂                            f_n
    #        ^                                         ^
    #     left face                               right face
    #
    # Right (face = +1): reconstruct at f_n, then extrapolate from u_n to ξ = +1.
    # Left  (face = -1): reconstruct at f₂, then extrapolate from u₁ to ξ = -1.
    if face > 0
        # Right face on ξ = +1
        i = n
        u_ll = cons2recon(get_node_vars(u, equations, dg, max(1, i - 2), element),
                          equations)
        u_lr = cons2recon(get_node_vars(u, equations, dg, i - 1, element), equations)
        u_rl = cons2recon(get_node_vars(u, equations, dg, i, element), equations)
        u_rr = u_rl
        _, u_face = reconstruction_O2(u_ll, u_lr, u_rl, u_rr,
                                      sc_interface_coords, i, slope_limiter, dg)
        x_c = nodes[i]
        return recon2cons(u_rl +
                          (u_face - u_rl) / (sc_interface_coords[i - 1] - x_c) *
                          (face - x_c), equations)
    else
        # Left face ξ = -1
        i = 2
        u_ll = cons2recon(get_node_vars(u, equations, dg, 1, element), equations)
        u_lr = u_ll
        u_rl = cons2recon(get_node_vars(u, equations, dg, 2, element), equations)
        u_rr = cons2recon(get_node_vars(u, equations, dg, min(n, 3), element),
                          equations)
        u_face, _ = reconstruction_O2(u_ll, u_lr, u_rl, u_rr,
                                      sc_interface_coords, i, slope_limiter, dg)
        x_c = nodes[i - 1]
        return recon2cons(u_lr +
                          (u_face - u_lr) / (sc_interface_coords[i - 1] - x_c) *
                          (face - x_c), equations)
    end
end

function prolong2interfaces!(cache, u,
                             mesh::TreeMesh{1}, equations, dg::BlockFVO2)
    @unpack interfaces = cache
    @unpack neighbor_ids = interfaces
    interfaces_u = interfaces.u
    volume_integral = dg.volume_integral

    @threaded for interface in eachinterface(dg, cache)
        left_element = neighbor_ids[1, interface]
        right_element = neighbor_ids[2, interface]

        u_left = reconstruct_element_face(u, equations, dg, left_element, 1,
                                          volume_integral)
        u_right = reconstruct_element_face(u, equations, dg, right_element, -1,
                                           volume_integral)
        for v in eachvariable(equations)
            interfaces_u[1, v, interface] = u_left[v]
            interfaces_u[2, v, interface] = u_right[v]
        end
    end

    return nothing
end

function prolong2boundaries!(backend::Nothing, cache, u,
                             mesh::TreeMesh{1}, equations, dg::BlockFVO2)
    @unpack boundaries = cache
    @unpack neighbor_sides = boundaries
    volume_integral = dg.volume_integral

    @threaded for boundary in eachboundary(dg, cache)
        element = boundaries.neighbor_ids[boundary]

        if neighbor_sides[boundary] == 1
            u_b = reconstruct_element_face(u, equations, dg, element, 1,
                                           volume_integral)
            for v in eachvariable(equations)
                boundaries.u[1, v, boundary] = u_b[v]
            end
        else
            u_b = reconstruct_element_face(u, equations, dg, element, -1,
                                           volume_integral)
            for v in eachvariable(equations)
                boundaries.u[2, v, boundary] = u_b[v]
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
