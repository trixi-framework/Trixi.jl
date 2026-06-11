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
end # @muladd



function prolong2mortars!(cache, u,
                          mesh::TreeMesh{2}, equations,
                          mortar_l2::UniformFiniteVolumeBasis,
                          dg::BlockFV)
    @threaded for mortar in eachmortar(dg, cache)
        large_element = cache.mortars.neighbor_ids[3, mortar]
        upper_element = cache.mortars.neighbor_ids[2, mortar]
        lower_element = cache.mortars.neighbor_ids[1, mortar]

        # Copy solution small to small
        if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
            if cache.mortars.orientations[mortar] == 1
                # L2 mortars in x-direction
                for l in eachnode(dg)
                    for v in eachvariable(equations)
                        cache.mortars.u_upper[2, v, l, mortar] = u[v, 1, l,
                                                                   upper_element]
                        cache.mortars.u_lower[2, v, l, mortar] = u[v, 1, l,
                                                                   lower_element]
                    end
                end
            else
                # L2 mortars in y-direction
                for l in eachnode(dg)
                    for v in eachvariable(equations)
                        cache.mortars.u_upper[2, v, l, mortar] = u[v, l, 1,
                                                                   upper_element]
                        cache.mortars.u_lower[2, v, l, mortar] = u[v, l, 1,
                                                                   lower_element]
                    end
                end
            end
        else # large_sides[mortar] == 2 -> small elements on left side
            if cache.mortars.orientations[mortar] == 1
                # L2 mortars in x-direction
                for l in eachnode(dg)
                    for v in eachvariable(equations)
                        cache.mortars.u_upper[1, v, l, mortar] = u[v, nnodes(dg), l,
                                                                   upper_element]
                        cache.mortars.u_lower[1, v, l, mortar] = u[v, nnodes(dg), l,
                                                                   lower_element]
                    end
                end
            else
                # L2 mortars in y-direction
                for l in eachnode(dg)
                    for v in eachvariable(equations)
                        cache.mortars.u_upper[1, v, l, mortar] = u[v, l, nnodes(dg),
                                                                   upper_element]
                        cache.mortars.u_lower[1, v, l, mortar] = u[v, l, nnodes(dg),
                                                                   lower_element]
                    end
                end
            end
        end

        # Interpolate large element face data to small interface locations
        if cache.mortars.large_sides[mortar] == 1 # -> large element on left side
            leftright = 1
            if cache.mortars.orientations[mortar] == 1
                # L2 mortars in x-direction
                u_large = view(u, :, nnodes(dg), :, large_element)
                element_solutions_to_mortars!(cache.mortars, mortar_l2, leftright,
                                              mortar, u_large)
            else
                # L2 mortars in y-direction
                u_large = view(u, :, :, nnodes(dg), large_element)
                element_solutions_to_mortars!(cache.mortars, mortar_l2, leftright,
                                              mortar, u_large)
            end
        else # large_sides[mortar] == 2 -> large element on right side
            leftright = 2
            if cache.mortars.orientations[mortar] == 1
                # L2 mortars in x-direction
                u_large = view(u, :, 1, :, large_element)
                element_solutions_to_mortars!(cache.mortars, mortar_l2, leftright,
                                              mortar, u_large)
            else
                # L2 mortars in y-direction
                u_large = view(u, :, :, 1, large_element)
                element_solutions_to_mortars!(cache.mortars, mortar_l2, leftright,
                                              mortar, u_large)
            end
        end
    end

    return nothing
end
#element_solution_to_mortars! brauch man eigentlich nicht, da die Lösung auf den Mortars (wird nur kopiert) 
#direkt in `prolong2mortars!` geschrieben werden kann, aber so ist es vielleicht 
#etwas übersichtlicher, muss auch noch angepasst werden

@inline function element_solutions_to_mortars!(mortars,
                                               mortar_l2::UniformFiniteVolumeBasis,
                                               leftright, mortar,
                                               u_large::AbstractArray{<:Any, 2})
    multiply_dimensionwise!(view(mortars.u_upper, leftright, :, :, mortar),
                            mortar_l2.forward_upper, u_large)
    multiply_dimensionwise!(view(mortars.u_lower, leftright, :, :, mortar),
                            mortar_l2.forward_lower, u_large)
    return nothing
end

function calc_mortar_flux!(surface_flux_values,
                           mesh::TreeMesh{2},
                           have_nonconservative_terms::False, equations,
                           mortar_l2::BlockFVMortar,
                           surface_integral, dg::BlockFV, cache)
    @unpack surface_flux = surface_integral
    @unpack u_lower, u_upper, orientations = cache.mortars
    @unpack fstar_primary_upper_threaded, fstar_primary_lower_threaded = cache

    @threaded for mortar in eachmortar(dg, cache)
        # Choose thread-specific pre-allocated container
        fstar_primary_upper = fstar_primary_upper_threaded[Threads.threadid()]
        fstar_primary_lower = fstar_primary_lower_threaded[Threads.threadid()]

        # Calculate fluxes
        orientation = orientations[mortar]
        calc_fstar!(fstar_primary_upper, equations, surface_flux, dg, u_upper, mortar,
                    orientation)
        calc_fstar!(fstar_primary_lower, equations, surface_flux, dg, u_lower, mortar,
                    orientation)

        # For non-conservative equations, we need two numerical fluxes
        # (primary and secondary). To use the same implementation of
        # `mortar_fluxes_to_elements!`, we pass the primary fluxes as
        # secondary fluxes as well in the conservative case. This is
        # possible since for conservative equations, numerical fluxes
        # are unique at interfaces (instead of having two different
        # fluxes/fluctuations for non-conservative equations).
        mortar_fluxes_to_elements!(surface_flux_values,
                                   mesh, equations, mortar_l2, dg, cache, mortar,
                                   fstar_primary_upper, fstar_primary_lower,
                                   fstar_primary_upper, fstar_primary_lower)
    end

    return nothing
end

function calc_mortar_flux!(surface_flux_values,
                           mesh::TreeMesh{2},
                           have_nonconservative_terms::True, equations,
                           mortar_l2::BlockFVMortar,
                           surface_integral, dg::BlockFV, cache)
    surface_flux, nonconservative_flux = surface_integral.surface_flux
    @unpack u_lower, u_upper, orientations, large_sides = cache.mortars
    @unpack (fstar_primary_upper_threaded, fstar_primary_lower_threaded,
    fstar_secondary_upper_threaded, fstar_secondary_lower_threaded) = cache

    @threaded for mortar in eachmortar(dg, cache)
        # Choose thread-specific pre-allocated container
        fstar_primary_upper = fstar_primary_upper_threaded[Threads.threadid()]
        fstar_primary_lower = fstar_primary_lower_threaded[Threads.threadid()]
        fstar_secondary_upper = fstar_secondary_upper_threaded[Threads.threadid()]
        fstar_secondary_lower = fstar_secondary_lower_threaded[Threads.threadid()]

        # Calculate fluxes
        orientation = orientations[mortar]
        calc_fstar!(fstar_primary_upper, equations, surface_flux, dg, u_upper, mortar,
                    orientation)
        calc_fstar!(fstar_primary_lower, equations, surface_flux, dg, u_lower, mortar,
                    orientation)
        calc_fstar!(fstar_secondary_upper, equations, surface_flux, dg, u_upper, mortar,
                    orientation)
        calc_fstar!(fstar_secondary_lower, equations, surface_flux, dg, u_lower, mortar,
                    orientation)

        # Add nonconservative fluxes.
        # These need to be adapted on the geometry (left/right) since the order of
        # the arguments matters, based on the global SBP operator interpretation.
        # The same interpretation (global SBP operators coupled discontinuously via
        # central fluxes/SATs) explains why we need the factor 0.5.
        # Alternatively, you can also follow the argumentation of Bohm et al. 2018
        # ("nonconservative diamond flux")
        if large_sides[mortar] == 1 # -> small elements on right side
            for i in eachnode(dg)
                # Pull the left and right solutions
                u_upper_ll, u_upper_rr = get_surface_node_vars(u_upper, equations, dg,
                                                               i, mortar)
                u_lower_ll, u_lower_rr = get_surface_node_vars(u_lower, equations, dg,
                                                               i, mortar)
                # Call pointwise nonconservative term
                noncons_primary_upper = nonconservative_flux(u_upper_ll, u_upper_rr,
                                                             orientation, equations)
                noncons_primary_lower = nonconservative_flux(u_lower_ll, u_lower_rr,
                                                             orientation, equations)
                noncons_secondary_upper = nonconservative_flux(u_upper_rr, u_upper_ll,
                                                               orientation, equations)
                noncons_secondary_lower = nonconservative_flux(u_lower_rr, u_lower_ll,
                                                               orientation, equations)
                # Add to primary and secondary temporary storage
                multiply_add_to_node_vars!(fstar_primary_upper, 0.5f0,
                                           noncons_primary_upper, equations,
                                           dg, i)
                multiply_add_to_node_vars!(fstar_primary_lower, 0.5f0,
                                           noncons_primary_lower, equations,
                                           dg, i)
                multiply_add_to_node_vars!(fstar_secondary_upper, 0.5f0,
                                           noncons_secondary_upper, equations,
                                           dg, i)
                multiply_add_to_node_vars!(fstar_secondary_lower, 0.5f0,
                                           noncons_secondary_lower, equations,
                                           dg, i)
            end
        else # large_sides[mortar] == 2 -> small elements on the left
            for i in eachnode(dg)
                # Pull the left and right solutions
                u_upper_ll, u_upper_rr = get_surface_node_vars(u_upper, equations, dg,
                                                               i, mortar)
                u_lower_ll, u_lower_rr = get_surface_node_vars(u_lower, equations, dg,
                                                               i, mortar)
                # Call pointwise nonconservative term
                noncons_primary_upper = nonconservative_flux(u_upper_rr, u_upper_ll,
                                                             orientation, equations)
                noncons_primary_lower = nonconservative_flux(u_lower_rr, u_lower_ll,
                                                             orientation, equations)
                noncons_secondary_upper = nonconservative_flux(u_upper_ll, u_upper_rr,
                                                               orientation, equations)
                noncons_secondary_lower = nonconservative_flux(u_lower_ll, u_lower_rr,
                                                               orientation, equations)
                # Add to primary and secondary temporary storage
                multiply_add_to_node_vars!(fstar_primary_upper, 0.5f0,
                                           noncons_primary_upper, equations,
                                           dg, i)
                multiply_add_to_node_vars!(fstar_primary_lower, 0.5f0,
                                           noncons_primary_lower, equations,
                                           dg, i)
                multiply_add_to_node_vars!(fstar_secondary_upper, 0.5f0,
                                           noncons_secondary_upper, equations,
                                           dg, i)
                multiply_add_to_node_vars!(fstar_secondary_lower, 0.5f0,
                                           noncons_secondary_lower, equations,
                                           dg, i)
            end
        end

        mortar_fluxes_to_elements!(surface_flux_values,
                                   mesh, equations, mortar_l2, dg, cache, mortar,
                                   fstar_primary_upper, fstar_primary_lower,
                                   fstar_secondary_upper, fstar_secondary_lower)
    end

    return nothing
end



@inline function calc_fstar!(destination::AbstractArray{<:Any, 2}, equations,
                             surface_flux, dg::BlockFV,
                             u_interfaces, interface, orientation)
    for i in eachnode(dg)
        # Call pointwise two-point numerical flux function
        u_ll, u_rr = get_surface_node_vars(u_interfaces, equations, dg, i, interface)
        flux = surface_flux(u_ll, u_rr, orientation, equations)

        # Copy flux to left and right element storage
        set_node_vars!(destination, flux, equations, dg, i)
    end

    return nothing
end

@inline function mortar_fluxes_to_elements!(surface_flux_values,
                                            mesh::TreeMesh{2}, equations,
                                            mortar:: BlockFVMortar,
                                            dg::BlockFV, cache,
                                            mortar_l2, fstar_primary_upper,
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

    # Project small fluxes to large element: hier nur mittelwert der flüsse wegen FV
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

    # TODO: Taal performance
    # for v in eachvariable(equations)
    #   # The code below is semantically equivalent to
    #   # surface_flux_values[v, :, direction, large_element] .=
    #   #   (mortar_l2.reverse_upper * fstar_upper[v, :] + mortar_l2.reverse_lower * fstar_lower[v, :])
    #   # but faster and does not allocate.
    #   # Note that `true * some_float == some_float` in Julia, i.e. `true` acts as
    #   # a universal `one`. Hence, the second `mul!` means "add the matrix-vector
    #   # product to the current value of the destination".
    #   @views mul!(surface_flux_values[v, :, direction, large_element],
    #               mortar_l2.reverse_upper, fstar_upper[v, :])
    #   @views mul!(surface_flux_values[v, :, direction, large_element],
    #               mortar_l2.reverse_lower, fstar_lower[v, :], true, true)
    # end
    # The code above could be replaced by the following code. However, the relative efficiency
    # depends on the types of fstar_upper/fstar_lower and dg.l2mortar_reverse_upper.
    # Using StaticArrays for both makes the code above faster for common test cases.
    multiply_dimensionwise!(view(surface_flux_values, :, :, direction, large_element),
                            mortar_l2.reverse_upper, fstar_secondary_upper,
                            mortar_l2.reverse_lower, fstar_secondary_lower)

    return nothing
end
