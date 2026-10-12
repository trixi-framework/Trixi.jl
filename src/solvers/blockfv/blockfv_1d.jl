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
    # Left/right neighbor element ids for each element, 0 means a domain boundary.
    # Only O2 reconstruction reads this.
    element_neighbors = zeros(Int, 2, nelements(dg, cache_containers))
    return (; fstar_threaded, element_neighbors)
end

# Fill neighbors from the mesh interfaces to enable element-spanning reconstruction.
@inline function update_element_neighbors!(element_neighbors, dg, cache)
    fill!(element_neighbors, 0)
    @unpack neighbor_ids = cache.interfaces
    for interface in eachinterface(dg, cache)
        left = neighbor_ids[1, interface]
        right = neighbor_ids[2, interface]
        element_neighbors[2, left] = right
        element_neighbors[1, right] = left
    end
    return nothing
end

# Look up a cell average and its reference coordinate, allowing indices that
# go into a neighboring element. If a neighbor does not exist (domain boundary) or the
# index would leave the neighbor too, the outermost available cell is returned
# and marked invalid.
@inline function extended_sample(u, equations, dg, idx, element, left, right)
    n = nnodes(dg)
    nodes = dg.basis.nodes
    dx = 2 * one(eltype(nodes)) / n

    if 1 <= idx <= n
        return get_node_vars(u, equations, dg, idx, element), nodes[idx], true
    elseif idx < 1
        j = n + idx
        x = nodes[1] + (idx - 1) * dx
        if left != 0 && j >= 1
            return get_node_vars(u, equations, dg, j, left), x, true
        else
            el = left == 0 ? element : left
            return get_node_vars(u, equations, dg, 1, el), x, false
        end
    else # idx > n
        j = idx - n
        x = nodes[n] + j * dx
        if right != 0 && j <= n
            return get_node_vars(u, equations, dg, j, right), x, true
        else
            el = right == 0 ? element : right
            return get_node_vars(u, equations, dg, n, el), x, false
        end
    end
end

# Reconstruct left and right states at a face from the four neighboring cell
# averages. Indices may reach into adjacent elements so
# that two small blocks match 1 larger block at the same combined size.
#
#            u_ll        u_lr   |   u_rl        u_rr
#              ·          ·     |     ·          ·
#           cell i-2   cell i-1 |  cell i    cell i+1
#                               ^
#                          face i (flux)
#
# Interior sides use the same limited slopes as reconstruction_O2_full /
# reconstruction_O2_inner. If an outer sample is missing at a domain boundary,
# the full reconstruction keeps the unlimited slope across the face, and the inner uses a zero slope.
@inline function reconstruct_face(u, equations, dg, i, x_face, element,
                                  left, right, volume_integral)
    @unpack reconstruction_mode, slope_limiter, cons2recon = volume_integral

    u_ll, x_ll, valid_ll = extended_sample(u, equations, dg, i - 2, element, left,
                                           right)
    u_lr, x_lr, _ = extended_sample(u, equations, dg, i - 1, element, left, right)
    u_rl, x_rl, _ = extended_sample(u, equations, dg, i, element, left, right)
    u_rr, x_rr, valid_rr = extended_sample(u, equations, dg, i + 1, element, left,
                                           right)

    u_ll, u_lr = cons2recon(u_ll, equations), cons2recon(u_lr, equations)
    u_rl, u_rr = cons2recon(u_rl, equations), cons2recon(u_rr, equations)

    reconstruction_mode === reconstruction_constant && return u_lr, u_rl

    s_m = (u_rl - u_lr) / (x_rl - x_lr)
    # For a missing outer neighbor, the full reconstruction keeps the cross-face slope and the inner uses a zero slope.
    s_boundary = reconstruction_mode === reconstruction_O2_inner ? zero(s_m) : s_m
    s_l = valid_ll ? slope_limiter.((u_lr - u_ll) / (x_lr - x_ll), s_m) : s_boundary
    s_r = valid_rr ? slope_limiter.(s_m, (u_rr - u_rl) / (x_rr - x_rl)) : s_boundary

    return u_lr + s_l * (x_face - x_lr), u_rl + s_r * (x_face - x_rl)
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
# Second-order volume integral with reconstructed states at internal faces.
# Extended indices reach into neighboring elements when present.
function calc_volume_integral!(backend::Nothing, du, u,
                               mesh::TreeMesh{1},
                               have_nonconservative_terms::False, equations,
                               volume_integral::VolumeIntegralFiniteVolumeO2,
                               dg::BlockFV, cache)
    @unpack sc_interface_coords, surface_flux, recon2cons = volume_integral
    @unpack fstar_threaded, element_neighbors = cache
    update_element_neighbors!(element_neighbors, dg, cache)

    n = nnodes(dg)
    inv_h = n * one(eltype(u)) / 2

    @threaded for element in eachelement(dg, cache)
        fstar = fstar_threaded[Threads.threadid()]
        left = element_neighbors[1, element]
        right = element_neighbors[2, element]

        # Each BlockFV element is split into n equal FV cells on [-1, 1].
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

        for i in 2:n
            u_l, u_r = reconstruct_face(u, equations, dg, i,
                                        sc_interface_coords[i - 1],
                                        element, left, right, volume_integral)
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
# Surface reconstruction for BlockFV with VolumeIntegralFiniteVolumeO2.
# Reconstruct to element face ξ = +/- 1 using `reconstruction_mode`,
# then extrapolate from the near-boundary internal face
# Passing an adjacent element lets the stencil cross the interface
#
#   ξ = -1                                              ξ = +1
#        |--- u₁ ---|--- u₂ --- ... ---|--- u_n ---|
#                   f₂                            f_n
#        ^                                         ^
#     left face                               right face
#
@inline function reconstruct_element_face(u, equations, dg::BlockFV, element, face,
                                          volume_integral::VolumeIntegralFiniteVolumeO2,
                                          neighbor_element::Integer = 0)
    @unpack recon2cons = volume_integral
    n = nnodes(dg)
    neighbor = Int(neighbor_element)

    # A single-cell element has no neighbors, so there is no reconstruction to do.
    if n == 1 && neighbor == 0
        return get_node_vars(u, equations, dg, 1, element)
    end

    if neighbor != 0
        # An interface between two elements is an ordinary internal interface of a larger block.
        if face > 0
            u_l, _ = reconstruct_face(u, equations, dg, n + 1, face, element,
                                      0, neighbor, volume_integral)
            return recon2cons(u_l, equations)
        else
            _, u_r = reconstruct_face(u, equations, dg, 1, face, element,
                                      neighbor, 0, volume_integral)
            return recon2cons(u_r, equations)
        end
    else
        # On the element boundary, there is no outside cell.
        if face > 0
            _, u_b = reconstruct_face(u, equations, dg, n, face, element,
                                      0, 0, volume_integral)
            return recon2cons(u_b, equations)
        else
            u_b, _ = reconstruct_face(u, equations, dg, 2, face, element,
                                      0, 0, volume_integral)
            return recon2cons(u_b, equations)
        end
    end
end

function prolong2interfaces!(cache, u,
                             mesh::TreeMesh{1}, equations,
                             dg::DG{<:UniformFiniteVolumeBasis, Mortar, SurfaceIntegral,
                                    <:VolumeIntegralFiniteVolumeO2}) where {Mortar,
                                                                            SurfaceIntegral
                                                                            }
    @unpack interfaces = cache
    @unpack neighbor_ids = interfaces
    interfaces_u = interfaces.u
    volume_integral = dg.volume_integral

    @threaded for interface in eachinterface(dg, cache)
        left_element = neighbor_ids[1, interface]
        right_element = neighbor_ids[2, interface]

        u_left = reconstruct_element_face(u, equations, dg, left_element, 1,
                                          volume_integral, right_element)
        u_right = reconstruct_element_face(u, equations, dg, right_element, -1,
                                           volume_integral, left_element)
        for v in eachvariable(equations)
            interfaces_u[1, v, interface] = u_left[v]
            interfaces_u[2, v, interface] = u_right[v]
        end
    end

    return nothing
end

function prolong2boundaries!(backend::Nothing, cache, u,
                             mesh::TreeMesh{1}, equations,
                             dg::DG{<:UniformFiniteVolumeBasis, Mortar, SurfaceIntegral,
                                    <:VolumeIntegralFiniteVolumeO2}) where {Mortar,
                                                                            SurfaceIntegral
                                                                            }
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
