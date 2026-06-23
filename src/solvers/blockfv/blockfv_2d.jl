# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Cache creation for VolumeIntegralFiniteVolume in 2D
# fstar_x[v, i, j]: flux at interface between cells (i-1,j) and (i,j) in x-direction
#                   shape (nvars, n+1, n); slots i=1 and i=n+1 are zero (boundary)
# fstar_y[v, i, j]: flux at interface between cells (i,j-1) and (i,j) in y-direction
#                   shape (nvars, n, n+1); slots j=1 and j=n+1 are zero (boundary)
function create_cache(mesh::TreeMesh{2}, equations,
                      volume_integral::VolumeIntegralFiniteVolume,
                      dg::BlockFV, cache_containers, uEltype)
    n = nnodes(dg)
    nv = nvariables(equations)

    MA_x = MArray{Tuple{nv, n + 1, n}, uEltype, 3, nv * (n + 1) * n}
    fstar_x_threaded = [MA_x(undef) for _ in 1:Threads.maxthreadid()]

    MA_y = MArray{Tuple{nv, n, n + 1}, uEltype, 3, nv * n * (n + 1)}
    fstar_y_threaded = [MA_y(undef) for _ in 1:Threads.maxthreadid()]

    for (fx, fy) in zip(fstar_x_threaded, fstar_y_threaded)
        fx[:, 1, :] .= zero(uEltype)
        fx[:, n + 1, :] .= zero(uEltype)
        fy[:, :, 1] .= zero(uEltype)
        fy[:, :, n + 1] .= zero(uEltype)
    end

    return (; fstar_x_threaded, fstar_y_threaded)
end

#####################################################################
# Volume integral: FV flux differences at internal faces in x and y
# For cell (i,j):
#   du[v,i,j] += inv_h * (fstar_x[v,i+1,j] - fstar_x[v,i,j])
#              + inv_h * (fstar_y[v,i,j+1] - fstar_y[v,i,j])
# Boundary slots in fstar are kept zero; the surface integral adds the
# actual element-boundary fluxes.
function calc_volume_integral!(backend::Nothing, du, u,
                               mesh::TreeMesh{2},
                               have_nonconservative_terms::False, equations,
                               volume_integral::VolumeIntegralFiniteVolume,
                               dg::BlockFV, cache)
    @unpack surface_flux = volume_integral
    @unpack fstar_x_threaded, fstar_y_threaded = cache
    inv_h = nnodes(dg) * one(eltype(u)) / 2

    @threaded for element in eachelement(dg, cache)
        fstar_x = fstar_x_threaded[Threads.threadid()]
        fstar_y = fstar_y_threaded[Threads.threadid()]

        # x-direction: internal interfaces at i + 1/2 for i = 1, ..., n-1
        for j in eachnode(dg)
            for i in 2:nnodes(dg)
                u_ll = get_node_vars(u, equations, dg, i - 1, j, element)
                u_rr = get_node_vars(u, equations, dg, i, j, element)
                f = surface_flux(u_ll, u_rr, 1, equations)
                set_node_vars!(fstar_x, f, equations, dg, i, j)
            end
        end

        # y-direction: internal interfaces at j + 1/2 for j = 1, ..., n-1
        for j in 2:nnodes(dg)
            for i in eachnode(dg)
                u_ll = get_node_vars(u, equations, dg, i, j - 1, element)
                u_rr = get_node_vars(u, equations, dg, i, j, element)
                f = surface_flux(u_ll, u_rr, 2, equations)
                set_node_vars!(fstar_y, f, equations, dg, i, j)
            end
        end

        # Apply flux differences
        for j in eachnode(dg)
            for i in eachnode(dg)
                for v in eachvariable(equations)
                    du[v, i, j, element] = (du[v, i, j, element] +
                                            inv_h *
                                            (fstar_x[v, i + 1, j] - fstar_x[v, i, j]) +
                                            inv_h *
                                            (fstar_y[v, i, j + 1] - fstar_y[v, i, j]))
                end
            end
        end
    end

    return nothing
end

#####################################################################
# Surface integral: element-boundary fluxes added to the boundary cells
# Directions: 1 = -x (left), 2 = +x (right), 3 = -y (bottom), 4 = +y (top)
# surface_flux_values has shape (nvars, nnodes, 4, nelements)
function calc_surface_integral!(backend::Nothing, du, u,
                                mesh::TreeMesh{2},
                                equations, surface_integral::SurfaceIntegralWeakForm,
                                dg::BlockFV, cache)
    @unpack surface_flux_values = cache.elements
    inv_h = nnodes(dg) * one(eltype(du)) / 2

    @threaded for element in eachelement(dg, cache)
        for j in eachnode(dg)
            for v in eachvariable(equations)
                # left boundary (-x): update leftmost column i=1
                du[v, 1, j, element] = du[v, 1, j, element] -
                                       inv_h * surface_flux_values[v, j, 1, element]
                # right boundary (+x): update rightmost column i=n
                du[v, nnodes(dg), j, element] = du[v, nnodes(dg), j, element] +
                                                inv_h *
                                                surface_flux_values[v, j, 2, element]
            end
        end
        for i in eachnode(dg)
            for v in eachvariable(equations)
                # bottom boundary (-y): update bottom row j=1
                du[v, i, 1, element] = du[v, i, 1, element] -
                                       inv_h * surface_flux_values[v, i, 3, element]
                # top boundary (+y): update top row j=n
                du[v, i, nnodes(dg), element] = du[v, i, nnodes(dg), element] +
                                                inv_h *
                                                surface_flux_values[v, i, 4, element]
            end
        end
    end

    return nothing
end

#####################################################################
# Integrate a function over the 2D domain using FV quadrature
function integrate_via_indices(func::Func, u,
                               mesh::TreeMesh{2}, equations,
                               dg::BlockFV, cache, args...;
                               normalize = true) where {Func}
    @unpack weights = dg.basis

    integral = zero(func(u, 1, 1, 1, equations, dg, args...))

    @batch reduction=(+, integral) for element in eachelement(dg, cache)
        volume_jacobian_ = volume_jacobian(element, mesh, cache)
        for j in eachnode(dg)
            for i in eachnode(dg)
                integral += volume_jacobian_ * weights[i] * weights[j] *
                            func(u, i, j, element, equations, dg, args...)
            end
        end
    end

    if normalize
        integral = integral / total_volume(mesh)
    end

    return integral
end

#####################################################################
# Discrete L2 and L∞ error norms in 2D
# Evaluates the exact solution at each FV cell center (no interpolation needed).
function calc_error_norms(func, u, t, analyzer,
                          mesh::TreeMesh{2}, equations, initial_condition,
                          dg::BlockFV, cache, cache_analysis)
    @unpack weights = dg.basis
    @unpack node_coordinates = cache.elements

    l2_error = zero(func(get_node_vars(u, equations, dg, 1, 1, 1), equations))
    linf_error = copy(l2_error)

    for element in eachelement(dg, cache)
        volume_jacobian_ = volume_jacobian(element, mesh, cache)
        for j in eachnode(dg)
            for i in eachnode(dg)
                x = get_node_coords(node_coordinates, equations, dg, i, j, element)
                u_exact = initial_condition(x, t, equations)
                diff = func(u_exact, equations) -
                       func(get_node_vars(u, equations, dg, i, j, element), equations)
                l2_error += diff .^ 2 * (weights[i] * weights[j] * volume_jacobian_)
                linf_error = @. max(linf_error, abs(diff))
            end
        end
    end

    total_volume_ = total_volume(mesh)
    l2_error = @. sqrt(l2_error / total_volume_)
    return l2_error, linf_error
end

@inline function element_solutions_to_mortars!(mortars,
                                                mortar_l2::UniformFiniteVolumeBasis,
                                                leftright,
                                                mortar,
                                                u_large::AbstractArray{<:Any, 2})

    # Project the solution from the large element to the two small mortar sides
    # by duplicating each large-element node
    if size(u_large, 2) % 2 == 1
        for i in 1:size(u_large, 2)
            # Copy values to the upper small element
            mortars.u_upper[leftright, :, i, mortar] = u_large[:, div(i + 1, 2)]

            # Copy values to the lower small element
            # (middle node is shared for odd numbers of nodes)
            mortars.u_lower[leftright, :, i, mortar] =
                u_large[:, div(i, 2) + 1 + div(size(u_large,2), 2)]
        end
    else
        for i in 1:size(u_large, 2)
            # Copy values to the upper small element
            mortars.u_upper[leftright, :, i, mortar] = u_large[:, div(i + 1, 2)]

            # Copy values to the lower small element
            mortars.u_lower[leftright, :, i, mortar] =
                u_large[:, div(i + 1, 2) + div(size(u_large,2), 2)]
        end
    end

    return nothing
end

@inline function mortar_fluxes_to_elements!(surface_flux_values,
                                            mesh::TreeMesh{2}, equations,
                                            mortar_l2:: UniformFiniteVolumeBasis,
                                            dg::BlockFV, cache,
                                            mortar, fstar_primary_upper,
                                            fstar_primary_lower,
                                            fstar_secondary_upper,
                                            fstar_secondary_lower)
    large_element = cache.mortars.neighbor_ids[3, mortar]
    upper_element = cache.mortars.neighbor_ids[2, mortar]
    lower_element = cache.mortars.neighbor_ids[1, mortar]

    # Copy flux small to small
    if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
        if cache.mortars.orientations[mortar] == 1
            # L2 mortars in x-direction
            direction = 1
        else
            # L2 mortars in y-direction
            direction = 3
        end
    else # large_sides[mortar] == 2 -> small elements on left side
        if cache.mortars.orientations[mortar] == 1
            # L2 mortars in x-direction
            direction = 2
        else
            # L2 mortars in y-direction
            direction = 4
        end
    end
    surface_flux_values[:, :, direction, upper_element] .= fstar_primary_upper
    surface_flux_values[:, :, direction, lower_element] .= fstar_primary_lower

    # Determine on which face of the large element the mortar is located and which direction it corresponds to:
    if cache.mortars.large_sides[mortar] == 1 # -> large element on left side
        if cache.mortars.orientations[mortar] == 1
            # L2 mortars in x-direction
            direction = 2
        else
            # L2 mortars in y-direction
            direction = 4
        end
    else # large_sides[mortar] == 2 -> large element on right side
        if cache.mortars.orientations[mortar] == 1
            # L2 mortars in x-direction
            direction = 1
        else
            # L2 mortars in y-direction
            direction = 3
        end
    end

    #Project fluxes from the two small elements to the large element. 
    #The fluxes on the small elements are already computed and stored in fstar_primary_upper and fstar_primary_lower. 
    #The fluxes on the large element are computed by averaging the fluxes from the two small elements.
    for v in eachvariable(equations)  
        if nnodes(dg) % 2 == 1
            #for an odd number of nodes, average the interface values at the center node
            surface_flux_values[v, (nnodes(dg) + 1 ) ÷ 2, direction, large_element] =
                0.5 * (fstar_primary_upper[v, end] + fstar_primary_lower[v, 1])
            for i in eachnode(mortar_l2)
                if i <= nnodes(dg) ÷ 2
                    #average neighboring fluxes from the upper small element
                    surface_flux_values[v, i, direction, large_element] =
                    0.5 * (fstar_primary_upper[v, 2*i-1] + fstar_primary_upper[v, 2*i])
                elseif i==(nnodes(dg) + 1 ) ÷ 2
                    continue
                    #center node already set above
                else
                    #average neighboring fluxes from the lower small element
                    surface_flux_values[v, i, direction, large_element] =
                    0.5 * (fstar_primary_lower[v, 2*i-1-nnodes(dg)] + fstar_primary_lower[v, 2*i- nnodes(dg)])
                end
            end
        else
            for i in eachnode(mortar_l2)
                if i <= nnodes(dg) ÷ 2
                    #average neighboring fluxes from the upper small element
                    surface_flux_values[v, i, direction, large_element] =
                    0.5 * (fstar_primary_upper[v, 2*i-1] + fstar_primary_upper[v, 2*i])
                else
                    #average neighboring fluxes from the lower small element
                    surface_flux_values[v, i, direction, large_element] =
                    0.5 * (fstar_primary_lower[v, 2*i-1-nnodes(dg)] + fstar_primary_lower[v, 2*i- nnodes(dg)])
                end  
            end
        end
    end
    return nothing
end
end # @muladd