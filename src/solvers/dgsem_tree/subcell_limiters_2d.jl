# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

###############################################################################
# IDP Limiting
###############################################################################

# this method is used when the limiter is constructed as for shock-capturing volume integrals
function create_cache(limiter::Type{SubcellLimiterIDP}, equations::AbstractEquations{2},
                      basis::LobattoLegendreBasis, bound_keys, bar_states)
    subcell_limiter_coefficients = Trixi.ContainerSubcellLimiterIDP2D{real(basis)}(0,
                                                                                   nnodes(basis),
                                                                                   bound_keys)

    cache = (;)
    if bar_states
        container_bar_states = Trixi.ContainerBarStates{real(basis)}(0,
                                                                     nvariables(equations),
                                                                     nnodes(basis))
        cache = (; cache..., container_bar_states)
    end

    # Memory for bounds checking routine with `BoundsCheckCallback`.
    # Local variable contains the maximum deviation since the last export.
    # Using a threaded vector to parallelize bounds check.
    idp_bounds_delta_local = Dict{Symbol, Vector{real(basis)}}()
    # Global variable contains the total maximum deviation.
    idp_bounds_delta_global = Dict{Symbol, real(basis)}()
    # Note: False sharing causes critical performance issues on multiple threads when using a vector
    # of length `Threads.nthreads()`. Initializing a vector of length `n * Threads.nthreads()`
    # and then only using every n-th entry, fixes the problem and allows proper scaling.
    # Since there are no processors with caches over 128B, we use `n = 128B / size(uEltype)`
    stride_size = div(128, sizeof(eltype(basis.nodes))) # = n
    for key in bound_keys
        idp_bounds_delta_local[key] = [zero(real(basis))
                                       for _ in 1:(stride_size * Threads.nthreads())]
        idp_bounds_delta_global[key] = zero(real(basis))
    end

    return (; cache..., subcell_limiter_coefficients, idp_bounds_delta_local,
            idp_bounds_delta_global)
end

function (limiter::SubcellLimiterIDP)(u::AbstractArray{<:Any, 4}, semi, dg::DGSEM, t,
                                      dt;
                                      kwargs...)
    mesh, _, _, _ = mesh_equations_solver_cache(semi)

    @unpack alpha = limiter.cache.subcell_limiter_coefficients
    # TODO: Do not abuse `reset_du!` but maybe implement a generic `set_zero!`
    @trixi_timeit timer() "reset alpha" reset_du!(alpha, dg, semi.cache)

    if limiter.smoothness_indicator
        elements = semi.cache.element_ids_dgfv
    else
        elements = eachelement(dg, semi.cache)
    end

    if limiter.local_minmax
        @trixi_timeit timer() "local min/max limiting" idp_local_minmax!(alpha, limiter,
                                                                         u, t, dt, semi,
                                                                         elements)
    end
    if limiter.positivity
        @trixi_timeit timer() "positivity" idp_positivity!(alpha, limiter, u, dt,
                                                           semi, elements)
    end
    if limiter.spec_entropy
        @trixi_timeit timer() "spec_entropy" idp_spec_entropy!(alpha, limiter, u, t,
                                                               dt, semi, mesh, elements)
    end
    if limiter.math_entropy
        @trixi_timeit timer() "math_entropy" idp_math_entropy!(alpha, limiter, u, t,
                                                               dt, semi, mesh, elements)
    end

    # Calculate alpha1 and alpha2
    @unpack alpha1, alpha2 = limiter.cache.subcell_limiter_coefficients
    @threaded for element in elements
        for j in eachnode(dg), i in 2:nnodes(dg)
            alpha1[i, j, element] = max(alpha[i - 1, j, element], alpha[i, j, element])
        end
        for j in 2:nnodes(dg), i in eachnode(dg)
            alpha2[i, j, element] = max(alpha[i, j - 1, element], alpha[i, j, element])
        end
        for i in eachnode(dg)
            alpha1[1, i, element] = zero(eltype(alpha1))
            alpha1[nnodes(dg) + 1, i, element] = zero(eltype(alpha1))
            alpha2[i, 1, element] = zero(eltype(alpha2))
            alpha2[i, nnodes(dg) + 1, element] = zero(eltype(alpha2))
        end
    end

    return nothing
end

###############################################################################
# Calculation of local bounds using low-order FV solution

@inline function calc_bounds_twosided!(var_min, var_max, variable, u, t, semi)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    # Calc bounds inside elements
    @threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            var_min[i, j, element] = typemax(eltype(var_min))
            var_max[i, j, element] = typemin(eltype(var_max))
        end
        # Calculate bounds at Gauss-Lobatto nodes using u
        for j in eachnode(dg), i in eachnode(dg)
            var = u[variable, i, j, element]
            var_min[i, j, element] = min(var_min[i, j, element], var)
            var_max[i, j, element] = max(var_max[i, j, element], var)

            if i > 1
                var_min[i - 1, j, element] = min(var_min[i - 1, j, element], var)
                var_max[i - 1, j, element] = max(var_max[i - 1, j, element], var)
            end
            if i < nnodes(dg)
                var_min[i + 1, j, element] = min(var_min[i + 1, j, element], var)
                var_max[i + 1, j, element] = max(var_max[i + 1, j, element], var)
            end
            if j > 1
                var_min[i, j - 1, element] = min(var_min[i, j - 1, element], var)
                var_max[i, j - 1, element] = max(var_max[i, j - 1, element], var)
            end
            if j < nnodes(dg)
                var_min[i, j + 1, element] = min(var_min[i, j + 1, element], var)
                var_max[i, j + 1, element] = max(var_max[i, j + 1, element], var)
            end
        end
    end

    # Values at element boundary
    calc_bounds_twosided_interface!(var_min, var_max, variable, u, t, semi, mesh)
end

@inline function calc_bounds_twosided_interface!(var_min, var_max, variable, u, t, semi,
                                                 mesh::TreeMesh2D)
    _, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; boundary_conditions) = semi
    # Calc bounds at interfaces and periodic boundaries
    for interface in eachinterface(dg, cache)
        # Get neighboring element ids
        left = cache.interfaces.neighbor_ids[1, interface]
        right = cache.interfaces.neighbor_ids[2, interface]

        orientation = cache.interfaces.orientations[interface]

        for i in eachnode(dg)
            index_left = (nnodes(dg), i)
            index_right = (1, i)
            if orientation == 2
                index_left = reverse(index_left)
                index_right = reverse(index_right)
            end
            var_left = u[variable, index_left..., left]
            var_right = u[variable, index_right..., right]

            var_min[index_right..., right] = min(var_min[index_right..., right],
                                                 var_left)
            var_max[index_right..., right] = max(var_max[index_right..., right],
                                                 var_left)

            var_min[index_left..., left] = min(var_min[index_left..., left], var_right)
            var_max[index_left..., left] = max(var_max[index_left..., left], var_right)
        end
    end

    # Calc bounds at physical boundaries
    for boundary in eachboundary(dg, cache)
        element = cache.boundaries.neighbor_ids[boundary]
        orientation = cache.boundaries.orientations[boundary]
        neighbor_side = cache.boundaries.neighbor_sides[boundary]

        for i in eachnode(dg)
            if neighbor_side == 2 # Element is on the right, boundary on the left
                index = (1, i)
                boundary_index = 1
            else # Element is on the left, boundary on the right
                index = (nnodes(dg), i)
                boundary_index = 2
            end
            if orientation == 2
                index = reverse(index)
                boundary_index += 2
            end
            u_inner = get_node_vars(u, equations, dg, index..., element)
            u_outer = get_boundary_outer_state(u_inner, cache, t,
                                               boundary_conditions[boundary_index],
                                               orientation, boundary_index,
                                               mesh, equations, dg,
                                               index..., element)
            var_outer = u_outer[variable]

            var_min[index..., element] = min(var_min[index..., element], var_outer)
            var_max[index..., element] = max(var_max[index..., element], var_outer)
        end
    end

    return nothing
end

@inline function calc_bounds_onesided!(var_minmax, minmax, typeminmax, variable, u, t,
                                       semi)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    # Calc bounds inside elements
    @threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            var_minmax[i, j, element] = typeminmax(eltype(var_minmax))
        end

        # Calculate bounds at Gauss-Lobatto nodes using u
        for j in eachnode(dg), i in eachnode(dg)
            var = variable(get_node_vars(u, equations, dg, i, j, element), equations)
            var_minmax[i, j, element] = minmax(var_minmax[i, j, element], var)

            if i > 1
                var_minmax[i - 1, j, element] = minmax(var_minmax[i - 1, j, element],
                                                       var)
            end
            if i < nnodes(dg)
                var_minmax[i + 1, j, element] = minmax(var_minmax[i + 1, j, element],
                                                       var)
            end
            if j > 1
                var_minmax[i, j - 1, element] = minmax(var_minmax[i, j - 1, element],
                                                       var)
            end
            if j < nnodes(dg)
                var_minmax[i, j + 1, element] = minmax(var_minmax[i, j + 1, element],
                                                       var)
            end
        end
    end

    # Values at element boundary
    calc_bounds_onesided_interface!(var_minmax, minmax, variable, u, t, semi, mesh)
end

@inline function calc_bounds_onesided_interface!(var_minmax, minmax, variable, u, t,
                                                 semi, mesh::TreeMesh2D)
    _, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; boundary_conditions) = semi
    # Calc bounds at interfaces and periodic boundaries
    for interface in eachinterface(dg, cache)
        # Get neighboring element ids
        left = cache.interfaces.neighbor_ids[1, interface]
        right = cache.interfaces.neighbor_ids[2, interface]

        orientation = cache.interfaces.orientations[interface]
        for i in eachnode(dg)
            index_left = (nnodes(dg), i)
            index_right = (1, i)
            if orientation == 2
                index_left = reverse(index_left)
                index_right = reverse(index_right)
            end
            var_left = variable(get_node_vars(u, equations, dg, index_left..., left),
                                equations)
            var_right = variable(get_node_vars(u, equations, dg, index_right..., right),
                                 equations)

            var_minmax[index_right..., right] = minmax(var_minmax[index_right...,
                                                                  right], var_left)
            var_minmax[index_left..., left] = minmax(var_minmax[index_left..., left],
                                                     var_right)
        end
    end

    # Calc bounds at physical boundaries
    for boundary in eachboundary(dg, cache)
        element = cache.boundaries.neighbor_ids[boundary]
        orientation = cache.boundaries.orientations[boundary]
        neighbor_side = cache.boundaries.neighbor_sides[boundary]

        for i in eachnode(dg)
            if neighbor_side == 2 # Element is on the right, boundary on the left
                index = (1, i)
                boundary_index = 1
            else # Element is on the left, boundary on the right
                index = (nnodes(dg), i)
                boundary_index = 2
            end
            if orientation == 2
                index = reverse(index)
                boundary_index += 2
            end
            u_inner = get_node_vars(u, equations, dg, index..., element)
            u_outer = get_boundary_outer_state(u_inner, cache, t,
                                               boundary_conditions[boundary_index],
                                               orientation, boundary_index,
                                               mesh, equations, dg,
                                               index..., element)
            var_outer = variable(u_outer, equations)

            var_minmax[index..., element] = minmax(var_minmax[index..., element],
                                                   var_outer)
        end
    end

    return nothing
end

###############################################################################
# Local minimum/maximum limiting

@inline function idp_local_minmax!(alpha, limiter, u, t, dt, semi, elements)
    mesh, _, _, _ = mesh_equations_solver_cache(semi)

    for variable in limiter.local_minmax_variables_cons
        idp_local_minmax!(alpha, limiter, u, t, dt, semi, mesh, elements, variable)
    end

    return nothing
end

@inline function idp_local_minmax!(alpha, limiter, u, t, dt, semi, mesh::TreeMesh{2},
                                   elements, variable)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    variable_string = string(variable)
    var_min = variable_bounds[Symbol(variable_string, "_min")]
    var_max = variable_bounds[Symbol(variable_string, "_max")]
    if !limiter.bar_states
        calc_bounds_twosided!(var_min, var_max, variable, u, t, semi)
    end

    @threaded for element in elements
        inverse_jacobian = cache.elements.inverse_jacobian[element]
        for j in eachnode(dg), i in eachnode(dg)
            idp_local_minmax_inner!(alpha, inverse_jacobian, u, dt, dg, cache, variable,
                                    var_min, var_max, i, j, element)
        end
    end

    return nothing
end

@inline function idp_local_minmax!(alpha, limiter, u, t, dt, semi,
                                   mesh::Union{StructuredMesh{2}, P4estMesh{2}},
                                   elements, variable)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    variable_string = string(variable)
    var_min = variable_bounds[Symbol(variable_string, "_min")]
    var_max = variable_bounds[Symbol(variable_string, "_max")]
    if !limiter.bar_states
        calc_bounds_twosided!(var_min, var_max, variable, u, t, semi)
    end

    @threaded for element in elements
        for j in eachnode(dg), i in eachnode(dg)
            inverse_jacobian = cache.elements.inverse_jacobian[i, j, element]
            idp_local_minmax_inner!(alpha, inverse_jacobian, u, dt, dg, cache, variable,
                                    var_min, var_max, i, j, element)
        end
    end

    return nothing
end

# Function barrier to dispatch outer function by mesh type
@inline function idp_local_minmax_inner!(alpha, inverse_jacobian, u, dt, dg, cache,
                                         variable, var_min, var_max, i, j, element)
    (; antidiffusive_flux1_L, antidiffusive_flux2_L, antidiffusive_flux1_R, antidiffusive_flux2_R) = cache.antidiffusive_fluxes
    (; inverse_weights) = dg.basis

    var = u[variable, i, j, element]
    # Real Zalesak type limiter
    #   * Zalesak (1979). "Fully multidimensional flux-corrected transport algorithms for fluids"
    #   * Kuzmin et al. (2010). "Failsafe flux limiting and constrained data projections for equations of gas dynamics"
    #   Note: The Zalesak limiter has to be computed, even if the state is valid, because the correction is
    #         for each interface, not each node

    Qp = max(0, (var_max[i, j, element] - var) / dt)
    Qm = min(0, (var_min[i, j, element] - var) / dt)

    # Calculate Pp and Pm
    # Note: Boundaries of antidiffusive_flux1/2 are constant 0, so they make no difference here.
    val_flux1_local = inverse_weights[i] *
                      antidiffusive_flux1_R[variable, i, j, element]
    val_flux1_local_ip1 = -inverse_weights[i] *
                          antidiffusive_flux1_L[variable, i + 1, j, element]
    val_flux2_local = inverse_weights[j] *
                      antidiffusive_flux2_R[variable, i, j, element]
    val_flux2_local_jp1 = -inverse_weights[j] *
                          antidiffusive_flux2_L[variable, i, j + 1, element]

    Pp = max(0, val_flux1_local) + max(0, val_flux1_local_ip1) +
         max(0, val_flux2_local) + max(0, val_flux2_local_jp1)
    Pm = min(0, val_flux1_local) + min(0, val_flux1_local_ip1) +
         min(0, val_flux2_local) + min(0, val_flux2_local_jp1)

    Qp = max(0, (var_max[i, j, element] - var) / dt)
    Qm = min(0, (var_min[i, j, element] - var) / dt)

    Pp = inverse_jacobian * Pp
    Pm = inverse_jacobian * Pm

    # Compute blending coefficient avoiding division by zero
    # (as in paper of [Guermond, Nazarov, Popov, Thomas] (4.8))
    Qp = abs(Qp) /
         (abs(Pp) + eps(typeof(Qp)) * 100 * abs(var_max[i, j, element]))
    Qm = abs(Qm) /
         (abs(Pm) + eps(typeof(Qm)) * 100 * abs(var_max[i, j, element]))

    # Calculate alpha at nodes
    alpha[i, j, element] = max(alpha[i, j, element], 1 - min(1, Qp, Qm))

    return nothing
end

###############################################################################
# Local minimum limiting of specific entropy

@inline function idp_spec_entropy!(alpha, limiter, u, t, dt, semi, mesh::TreeMesh{2},
                                   elements)
    _, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    s_min = variable_bounds[:spec_entropy_min]
    if !limiter.bar_states
        calc_bounds_onesided!(s_min, min, typemax, entropy_spec, u, t, semi)
    end

    # Perform Newton's bisection method to find new alpha
    @threaded for element in elements
        inverse_jacobian = cache.elements.inverse_jacobian[element]
        for j in eachnode(dg), i in eachnode(dg)
            u_local = get_node_vars(u, equations, dg, i, j, element)
            newton_loops_alpha!(alpha, s_min[i, j, element], u_local, inverse_jacobian,
                                i, j, element, dt, equations, dg, cache, limiter,
                                entropy_spec, initial_check_entropy_spec_newton_idp,
                                final_check_standard_newton_idp)
        end
    end

    return nothing
end

@inline function idp_spec_entropy!(alpha, limiter, u, t, dt, semi,
                                   mesh::Union{StructuredMesh{2}, P4estMesh{2}},
                                   elements)
    _, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    s_min = variable_bounds[:spec_entropy_min]
    if !limiter.bar_states
        calc_bounds_onesided!(s_min, min, typemax, entropy_spec, u, t, semi)
    end

    # Perform Newton's bisection method to find new alpha
    @threaded for element in elements
        for j in eachnode(dg), i in eachnode(dg)
            inverse_jacobian = cache.elements.inverse_jacobian[i, j, element]
            u_local = get_node_vars(u, equations, dg, i, j, element)
            newton_loops_alpha!(alpha, s_min[i, j, element], u_local, inverse_jacobian,
                                i, j, element, dt, equations, dg, cache, limiter,
                                entropy_spec, initial_check_entropy_spec_newton_idp,
                                final_check_standard_newton_idp)
        end
    end

    return nothing
end

###############################################################################
# Local maximum limiting of mathematical entropy

@inline function idp_math_entropy!(alpha, limiter, u, t, dt, semi, mesh::TreeMesh{2},
                                   elements)
    _, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    s_max = variable_bounds[:math_entropy_max]
    if !limiter.bar_states
        calc_bounds_onesided!(s_max, max, typemin, entropy_math, u, t, semi)
    end

    # Perform Newton's bisection method to find new alpha
    @threaded for element in elements
        inverse_jacobian = cache.elements.inverse_jacobian[element]
        for j in eachnode(dg), i in eachnode(dg)
            u_local = get_node_vars(u, equations, dg, i, j, element)
            newton_loops_alpha!(alpha, s_max[i, j, element], u_local, inverse_jacobian,
                                i, j, element, dt, equations, dg, cache, limiter,
                                entropy_math, initial_check_entropy_math_newton_idp,
                                final_check_standard_newton_idp)
        end
    end

    return nothing
end

@inline function idp_math_entropy!(alpha, limiter, u, t, dt, semi,
                                   mesh::Union{StructuredMesh{2}, P4estMesh{2}},
                                   elements)
    _, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    s_max = variable_bounds[:math_entropy_max]
    if !limiter.bar_states
        calc_bounds_onesided!(s_max, max, typemin, entropy_math, u, t, semi)
    end

    # Perform Newton's bisection method to find new alpha
    @threaded for element in elements
        for j in eachnode(dg), i in eachnode(dg)
            inverse_jacobian = cache.elements.inverse_jacobian[i, j, element]
            u_local = get_node_vars(u, equations, dg, i, j, element)
            newton_loops_alpha!(alpha, s_max[i, j, element], u_local, inverse_jacobian,
                                i, j, element, dt, equations, dg, cache, limiter,
                                entropy_math, initial_check_entropy_math_newton_idp,
                                final_check_standard_newton_idp)
        end
    end

    return nothing
end

###############################################################################
# Global positivity limiting

@inline function idp_positivity!(alpha, limiter, u, dt, semi, elements)
    mesh, _, _, _ = mesh_equations_solver_cache(semi)

    # Conservative variables
    for variable in limiter.positivity_variables_cons
        @trixi_timeit timer() "conservative variables" idp_positivity_conservative!(alpha,
                                                                                    limiter,
                                                                                    u,
                                                                                    dt,
                                                                                    semi,
                                                                                    mesh,
                                                                                    elements,
                                                                                    variable)
    end

    # Nonlinear variables
    for variable in limiter.positivity_variables_nonlinear
        @trixi_timeit timer() "nonlinear variables" idp_positivity_nonlinear!(alpha,
                                                                              limiter,
                                                                              u, dt,
                                                                              semi,
                                                                              mesh,
                                                                              elements,
                                                                              variable)
    end

    return nothing
end

###############################################################################
# Global positivity limiting of conservative variables

@inline function idp_positivity_conservative!(alpha, limiter, u, dt, semi,
                                              mesh::TreeMesh{2}, elements, variable)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_min = variable_bounds[Symbol(string(variable), "_min")]

    @threaded for element in elements
        inverse_jacobian = cache.elements.inverse_jacobian[element]
        for j in eachnode(dg), i in eachnode(dg)
            idp_positivity_conservative_inner!(alpha, inverse_jacobian, limiter, u, dt,
                                               dg, cache, variable, var_min,
                                               i, j, element)
        end
    end

    return nothing
end

@inline function idp_positivity_conservative!(alpha, limiter, u, dt, semi,
                                              mesh::Union{StructuredMesh{2},
                                                          P4estMesh{2}},
                                              elements, variable)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_min = variable_bounds[Symbol(string(variable), "_min")]

    @threaded for element in elements
        for j in eachnode(dg), i in eachnode(dg)
            inverse_jacobian = cache.elements.inverse_jacobian[i, j, element]
            idp_positivity_conservative_inner!(alpha, inverse_jacobian, limiter, u, dt,
                                               dg, cache, variable, var_min,
                                               i, j, element)
        end
    end

    return nothing
end

# Function barrier to dispatch outer function by mesh type
@inline function idp_positivity_conservative_inner!(alpha, inverse_jacobian, limiter, u,
                                                    dt, dg, cache, variable, var_min,
                                                    i, j, element)
    (; antidiffusive_flux1_L, antidiffusive_flux2_L, antidiffusive_flux1_R, antidiffusive_flux2_R) = cache.antidiffusive_fluxes
    (; inverse_weights) = dg.basis
    (; positivity_correction_factor) = limiter

    var = u[variable, i, j, element]
    if var < 0
        error("Safe low-order method produces negative value for conservative variable $variable. Try a smaller time step.")
    end

    # Compute bound
    if limiter.local_minmax &&
       variable in limiter.local_minmax_variables_cons &&
       var_min[i, j, element] >= positivity_correction_factor * var
        # Local limiting is more restrictive that positivity limiting
        # => Skip positivity limiting for this node
        return nothing
    end
    var_min[i, j, element] = positivity_correction_factor * var

    # Real one-sided Zalesak-type limiter
    # * Zalesak (1979). "Fully multidimensional flux-corrected transport algorithms for fluids"
    # * Kuzmin et al. (2010). "Failsafe flux limiting and constrained data projections for equations of gas dynamics"
    # Note: The Zalesak limiter has to be computed, even if the state is valid, because the correction is
    #       for each interface, not each node
    Qm = min(0, (var_min[i, j, element] - var) / dt)

    # Calculate Pm
    # Note: Boundaries of antidiffusive_flux1/2 are constant 0, so they make no difference here.
    val_flux1_local = inverse_weights[i] *
                      antidiffusive_flux1_R[variable, i, j, element]
    val_flux1_local_ip1 = -inverse_weights[i] *
                          antidiffusive_flux1_L[variable, i + 1, j, element]
    val_flux2_local = inverse_weights[j] *
                      antidiffusive_flux2_R[variable, i, j, element]
    val_flux2_local_jp1 = -inverse_weights[j] *
                          antidiffusive_flux2_L[variable, i, j + 1, element]

    Pm = min(0, val_flux1_local) + min(0, val_flux1_local_ip1) +
         min(0, val_flux2_local) + min(0, val_flux2_local_jp1)
    Pm = inverse_jacobian * Pm

    # Compute blending coefficient avoiding division by zero
    # (as in paper of [Guermond, Nazarov, Popov, Thomas] (4.8))
    Qm = abs(Qm) / (abs(Pm) + eps(typeof(Qm)) * 100)

    # Calculate alpha
    alpha[i, j, element] = max(alpha[i, j, element], 1 - Qm)

    return nothing
end

###############################################################################
# Global positivity limiting of nonlinear variables

@inline function idp_positivity_nonlinear!(alpha, limiter, u, dt, semi,
                                           mesh::TreeMesh{2}, elements, variable)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_min = variable_bounds[Symbol(string(variable), "_min")]

    @threaded for element in elements
        inverse_jacobian = cache.elements.inverse_jacobian[element]
        for j in eachnode(dg), i in eachnode(dg)
            idp_positivity_nonlinear_inner!(alpha, inverse_jacobian, limiter, u, dt,
                                            semi, dg, cache, variable, var_min,
                                            i, j, element)
        end
    end

    return nothing
end

@inline function idp_positivity_nonlinear!(alpha, limiter, u, dt, semi,
                                           mesh::Union{StructuredMesh{2}, P4estMesh{2}},
                                           elements, variable)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_min = variable_bounds[Symbol(string(variable), "_min")]

    @threaded for element in elements
        for j in eachnode(dg), i in eachnode(dg)
            inverse_jacobian = cache.elements.inverse_jacobian[i, j, element]
            idp_positivity_nonlinear_inner!(alpha, inverse_jacobian, limiter, u, dt,
                                            semi, dg, cache, variable, var_min,
                                            i, j, element)
        end
    end

    return nothing
end

# Function barrier to dispatch outer function by mesh type
@inline function idp_positivity_nonlinear_inner!(alpha, inverse_jacobian, limiter, u,
                                                 dt, semi, dg, cache, variable, var_min,
                                                 i, j, element)
    _, equations, _, _ = mesh_equations_solver_cache(semi)

    u_local = get_node_vars(u, equations, dg, i, j, element)
    var = variable(u_local, equations)
    if var < 0
        error("Safe low-order method produces negative value for conservative variable $variable. Try a smaller time step.")
    end
    var_min[i, j, element] = limiter.positivity_correction_factor * var

    # Perform Newton's bisection method to find new alpha
    newton_loops_alpha!(alpha, var_min[i, j, element], u_local, inverse_jacobian, i, j,
                        element, dt, equations, dg, cache, limiter, variable,
                        initial_check_nonnegative_newton_idp,
                        final_check_nonnegative_newton_idp)

    return nothing
end

###############################################################################
# Newton-bisection method

@inline function newton_loops_alpha!(alpha, bound, u, inverse_jacobian, i, j, element,
                                     dt, equations, dg, cache, limiter,
                                     variable, initial_check, final_check)
    (; inverse_weights) = dg.basis
    (; antidiffusive_flux1_L, antidiffusive_flux2_L, antidiffusive_flux1_R, antidiffusive_flux2_R) = cache.antidiffusive_fluxes

    (; gamma_constant_newton) = limiter

    # negative xi direction
    antidiffusive_flux = gamma_constant_newton * inverse_jacobian * inverse_weights[i] *
                         get_node_vars(antidiffusive_flux1_R, equations, dg, i, j,
                                       element)
    newton_loop!(alpha, bound, u, i, j, element, variable, initial_check, final_check,
                 equations, dt, limiter, antidiffusive_flux)

    # positive xi direction
    antidiffusive_flux = -gamma_constant_newton * inverse_jacobian *
                         inverse_weights[i] *
                         get_node_vars(antidiffusive_flux1_L, equations, dg, i + 1, j,
                                       element)
    newton_loop!(alpha, bound, u, i, j, element, variable, initial_check, final_check,
                 equations, dt, limiter, antidiffusive_flux)

    # negative eta direction
    antidiffusive_flux = gamma_constant_newton * inverse_jacobian * inverse_weights[j] *
                         get_node_vars(antidiffusive_flux2_R, equations, dg, i, j,
                                       element)
    newton_loop!(alpha, bound, u, i, j, element, variable, initial_check, final_check,
                 equations, dt, limiter, antidiffusive_flux)

    # positive eta direction
    antidiffusive_flux = -gamma_constant_newton * inverse_jacobian *
                         inverse_weights[j] *
                         get_node_vars(antidiffusive_flux2_L, equations, dg, i, j + 1,
                                       element)
    newton_loop!(alpha, bound, u, i, j, element, variable, initial_check, final_check,
                 equations, dt, limiter, antidiffusive_flux)

    return nothing
end

@inline function newton_loop!(alpha, bound, u, i, j, element, variable, initial_check,
                              final_check, equations, dt, limiter, antidiffusive_flux)
    newton_reltol, newton_abstol = limiter.newton_tolerances

    beta = 1 - alpha[i, j, element]

    beta_L = 0 # alpha = 1
    beta_R = beta # No higher beta (lower alpha) than the current one

    u_curr = u + beta * dt * antidiffusive_flux

    # If state is valid, perform initial check and return if correction is not needed
    if isvalid(u_curr, equations)
        goal = goal_function_newton_idp(variable, bound, u_curr, equations)

        initial_check(bound, goal, newton_abstol) && return nothing
    end

    # Newton iterations
    for iter in 1:(limiter.max_iterations_newton)
        beta_old = beta

        # If the state is valid, evaluate d(goal)/d(beta)
        if isvalid(u_curr, equations)
            dgoal_dbeta = dgoal_function_newton_idp(variable, u_curr, dt,
                                                    antidiffusive_flux, equations)
        else # Otherwise, perform a bisection step
            dgoal_dbeta = 0
        end

        if dgoal_dbeta != 0
            # Update beta with Newton's method
            beta = beta - goal / dgoal_dbeta
        end

        # Check bounds
        if (beta < beta_L) || (beta > beta_R) || (dgoal_dbeta == 0) || isnan(beta)
            # Out of bounds, do a bisection step
            beta = 0.5 * (beta_L + beta_R)
            # Get new u
            u_curr = u + beta * dt * antidiffusive_flux

            # If the state is invalid, finish bisection step without checking tolerance and iterate further
            if !isvalid(u_curr, equations)
                beta_R = beta
                continue
            end

            # Check new beta for condition and update bounds
            goal = goal_function_newton_idp(variable, bound, u_curr, equations)
            if initial_check(bound, goal, newton_abstol)
                # New beta fulfills condition
                beta_L = beta
            else
                # New beta does not fulfill condition
                beta_R = beta
            end
        else
            # Get new u
            u_curr = u + beta * dt * antidiffusive_flux

            # If the state is invalid, redefine right bound without checking tolerance and iterate further
            if !isvalid(u_curr, equations)
                beta_R = beta
                continue
            end

            # Evaluate goal function
            goal = goal_function_newton_idp(variable, bound, u_curr, equations)
        end

        # Check relative tolerance
        if abs(beta_old - beta) <= newton_reltol
            break
        end

        # Check absolute tolerance
        if final_check(bound, goal, newton_abstol)
            break
        end
    end

    new_alpha = 1 - beta
    if alpha[i, j, element] > new_alpha + newton_abstol
        error("Alpha is getting smaller. old: $(alpha[i, j, element]), new: $new_alpha")
    else
        alpha[i, j, element] = new_alpha
    end

    return nothing
end

### Auxiliary routines for Newton's bisection method ###
# Initial checks
@inline function initial_check_entropy_spec_newton_idp(bound, goal, newton_abstol)
    goal <= max(newton_abstol, abs(bound) * newton_abstol)
end

@inline function initial_check_entropy_math_newton_idp(bound, goal, newton_abstol)
    goal >= -max(newton_abstol, abs(bound) * newton_abstol)
end

@inline initial_check_nonnegative_newton_idp(bound, goal, newton_abstol) = goal <= 0

# Goal and d(Goal)d(u) function
@inline goal_function_newton_idp(variable, bound, u, equations) = bound -
                                                                  variable(u, equations)
@inline function dgoal_function_newton_idp(variable, u, dt, antidiffusive_flux,
                                           equations)
    -dot(gradient_conservative(variable, u, equations), dt * antidiffusive_flux)
end

# Final checks
@inline function final_check_standard_newton_idp(bound, goal, newton_abstol)
    abs(goal) < max(newton_abstol, abs(bound) * newton_abstol)
end

@inline function final_check_nonnegative_newton_idp(bound, goal, newton_abstol)
    (goal <= eps()) && (goal > -max(newton_abstol, abs(bound) * newton_abstol))
end

###############################################################################
# Monolithic Convex Limiting
###############################################################################

# this method is used when the limiter is constructed as for shock-capturing volume integrals
function create_cache(limiter::Type{SubcellLimiterMCL}, equations::AbstractEquations{2},
                      basis::LobattoLegendreBasis, positivity_limiter_pressure)
    subcell_limiter_coefficients = Trixi.ContainerSubcellLimiterMCL2D{real(basis)}(0,
                                                                                   nvariables(equations),
                                                                                   nnodes(basis))
    container_bar_states = Trixi.ContainerBarStates{real(basis)}(0,
                                                                 nvariables(equations),
                                                                 nnodes(basis))

    # Memory for bounds checking routine with `BoundsCheckCallback`.
    # Local variable contains the maximum deviation since the last export.
    # [min / max, variable]
    mcl_bounds_delta_local = zeros(real(basis), 2,
                                   nvariables(equations) + positivity_limiter_pressure)
    # Global variable contains the total maximum deviation.
    # [min / max, variable]
    mcl_bounds_delta_global = zeros(real(basis), 2,
                                    nvariables(equations) + positivity_limiter_pressure)

    return (; subcell_limiter_coefficients, container_bar_states,
            mcl_bounds_delta_local, mcl_bounds_delta_global)
end
end # @muladd
