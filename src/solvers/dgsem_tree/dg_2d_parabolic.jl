# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# This file collects all methods that have been updated to work with parabolic systems of equations
#
# assumptions: parabolic terms are of the form div(f(u, grad(u))) and
# will be discretized first order form as follows:
#               1. compute grad(u)
#               2. compute f(u, grad(u))
#               3. compute div(f(u, grad(u))) (i.e., the "regular" rhs! call)
# boundary conditions will be applied to both grad(u) and div(f(u, grad(u))).
function rhs_parabolic!(du, u, t, mesh::Union{TreeMesh{2}, TreeMesh{3}},
                        equations_parabolic::AbstractEquationsParabolic,
                        boundary_conditions_parabolic, source_terms,
                        dg::DG, parabolic_scheme, cache, cache_parabolic)
    @unpack viscous_container = cache_parabolic
    @unpack u_transformed, gradients, flux_viscous = viscous_container

    # Convert conservative variables to a form more suitable for viscous flux calculations
    @trixi_timeit timer() "transform variables" begin
        transform_variables!(u_transformed, u, mesh, equations_parabolic,
                             dg, parabolic_scheme, cache)
    end

    # Compute the gradients of the transformed variables
    @trixi_timeit timer() "calculate gradient" begin
        calc_gradient!(gradients, u_transformed, t, mesh,
                       equations_parabolic, boundary_conditions_parabolic,
                       dg, parabolic_scheme, cache)
    end

    # Compute and store the viscous fluxes
    @trixi_timeit timer() "calculate viscous fluxes" begin
        calc_viscous_fluxes!(flux_viscous, gradients, u_transformed, mesh,
                             equations_parabolic, dg, cache)
    end

    # The remainder of this function is essentially a regular rhs! for parabolic
    # equations (i.e., it computes the divergence of the viscous fluxes)
    #
    # OBS! In `calc_viscous_fluxes!`, the viscous flux values at the volume nodes of each element have
    # been computed and stored in `fluxes_viscous`. In the following, we *reuse* (abuse) the
    # `interfaces` and `boundaries` containers in `cache` to interpolate and store the
    # *fluxes* at the element surfaces, as opposed to interpolating and storing the *solution* (as it
    # is done in the hyperbolic operator). That is, `interfaces.u`/`boundaries.u` store *viscous flux values*
    # and *not the solution*.  The advantage is that a) we do not need to allocate more storage, b) we
    # do not need to recreate the existing data structure only with a different name, and c) we do not
    # need to interpolate solutions *and* gradients to the surfaces.

    # Reset du
    @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

    # Calculate volume integral.
    # This calls the specialized version for the viscous fluxes from
    # `dg_2d_parabolic.jl` or `dg_3d_parabolic.jl`.
    @trixi_timeit timer() "volume integral" begin
        calc_volume_integral!(du, flux_viscous, mesh, equations_parabolic, dg, cache)
    end

    # Prolong solution to interfaces.
    # This calls the specialized version for the viscous fluxes from
    # `dg_2d_parabolic.jl` or `dg_3d_parabolic.jl`.
    @trixi_timeit timer() "prolong2interfaces" begin
        prolong2interfaces!(cache, flux_viscous, mesh, equations_parabolic, dg)
    end

    # Calculate interface fluxes
    # This calls the specialized version for the viscous fluxes from
    # `dg_2d_parabolic.jl` or `dg_3d_parabolic.jl`.
    @trixi_timeit timer() "interface flux" begin
        calc_interface_flux!(cache.elements.surface_flux_values,
                             mesh, equations_parabolic, dg,
                             parabolic_scheme, cache)
    end

    # Prolong viscous fluxes to boundaries.
    # This calls the specialized version for the viscous fluxes from
    # `dg_2d_parabolic.jl` or `dg_3d_parabolic.jl`.
    @trixi_timeit timer() "prolong2boundaries" begin
        prolong2boundaries!(cache, flux_viscous, mesh, equations_parabolic,
                            dg.surface_integral, dg)
    end

    # Calculate boundary fluxes.
    # This calls the specialized version for parabolic equations.
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux_divergence!(cache, t,
                                       boundary_conditions_parabolic, mesh,
                                       equations_parabolic,
                                       dg.surface_integral, dg)
    end

    # Prolong viscous fluxes to mortars.
    # This calls the specialized version for the viscous fluxes from
    # `dg_2d_parabolic.jl` or `dg_3d_parabolic.jl`.
    @trixi_timeit timer() "prolong2mortars" begin
        prolong2mortars!(cache, flux_viscous, mesh, equations_parabolic,
                         dg.mortar, dg)
    end

    # Calculate mortar fluxes.
    # This calls the specialized version from 
    # `dg_2d_parabolic.jl` or `dg_3d_parabolic.jl`.
    @trixi_timeit timer() "mortar flux" begin
        calc_mortar_flux!(cache.elements.surface_flux_values,
                          mesh, equations_parabolic, dg.mortar, dg.surface_integral,
                          dg, parabolic_scheme, Divergence(), cache)
    end

    # Calculate surface integrals.
    # This reuses `calc_surface_integral!` for the purely hyperbolic case.
    @trixi_timeit timer() "surface integral" begin
        calc_surface_integral!(du, u, mesh, equations_parabolic,
                               dg.surface_integral, dg, cache)
    end

    # Apply Jacobian from mapping to reference element
    @trixi_timeit timer() "Jacobian" begin
        apply_jacobian_parabolic!(du, mesh, equations_parabolic, dg, cache)
    end

    return nothing
end

# Transform solution variables prior to taking the gradient
# (e.g., conservative to primitive variables). Defaults to doing nothing.
# TODO: can we avoid copying data?
function transform_variables!(u_transformed, u, mesh::Union{TreeMesh{2}, P4estMesh{2}},
                              equations_parabolic::AbstractEquationsParabolic,
                              dg::DG, parabolic_scheme, cache)
    transformation = gradient_variable_transformation(equations_parabolic)

    @threaded for element in eachelement(dg, cache)
        # Calculate volume terms in one element
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations_parabolic, dg, i, j, element)
            u_transformed_node = transformation(u_node, equations_parabolic)
            set_node_vars!(u_transformed, u_transformed_node, equations_parabolic, dg,
                           i, j, element)
        end
    end

    return nothing
end

# This is the version used when calculating the divergence of the viscous fluxes
function calc_volume_integral!(du, flux_viscous,
                               mesh::TreeMesh{2},
                               equations_parabolic::AbstractEquationsParabolic,
                               dg::DGSEM, cache)
    @unpack derivative_dhat = dg.basis
    flux_viscous_x, flux_viscous_y = flux_viscous

    @threaded for element in eachelement(dg, cache)
        # Calculate volume terms in one element
        for j in eachnode(dg), i in eachnode(dg)
            flux_1_node = get_node_vars(flux_viscous_x, equations_parabolic, dg,
                                        i, j, element)
            flux_2_node = get_node_vars(flux_viscous_y, equations_parabolic, dg,
                                        i, j, element)

            for ii in eachnode(dg)
                multiply_add_to_node_vars!(du, derivative_dhat[ii, i], flux_1_node,
                                           equations_parabolic, dg, ii, j, element)
            end

            for jj in eachnode(dg)
                multiply_add_to_node_vars!(du, derivative_dhat[jj, j], flux_2_node,
                                           equations_parabolic, dg, i, jj, element)
            end
        end
    end

    return nothing
end

# This is the version used when calculating the divergence of the viscous fluxes.
# Specialization `flux_viscous::Tuple` needed to
# avoid amibiguity with the hyperbolic version of `prolong2interfaces!` in dg_2d.jl
# which is for the variables itself, i.e., `u::Array{uEltype, 4}`.
function prolong2interfaces!(cache, flux_viscous::Tuple,
                             mesh::TreeMesh{2},
                             equations_parabolic::AbstractEquationsParabolic,
                             dg::DG)
    @unpack interfaces = cache
    @unpack orientations, neighbor_ids = interfaces
    interfaces_u = interfaces.u

    flux_viscous_x, flux_viscous_y = flux_viscous

    @threaded for interface in eachinterface(dg, cache)
        left_element = neighbor_ids[1, interface]
        right_element = neighbor_ids[2, interface]

        if orientations[interface] == 1
            # interface in x-direction
            for j in eachnode(dg), v in eachvariable(equations_parabolic)
                # OBS! `interfaces_u` stores the interpolated *fluxes* and *not the solution*!
                interfaces_u[1, v, j, interface] = flux_viscous_x[v, nnodes(dg), j,
                                                                  left_element]
                interfaces_u[2, v, j, interface] = flux_viscous_x[v, 1, j,
                                                                  right_element]
            end
        else # if orientations[interface] == 2
            # interface in y-direction
            for i in eachnode(dg), v in eachvariable(equations_parabolic)
                # OBS! `interfaces_u` stores the interpolated *fluxes* and *not the solution*!
                interfaces_u[1, v, i, interface] = flux_viscous_y[v, i, nnodes(dg),
                                                                  left_element]
                interfaces_u[2, v, i, interface] = flux_viscous_y[v, i, 1,
                                                                  right_element]
            end
        end
    end

    return nothing
end

# This is the version used when calculating the divergence of the viscous fluxes
function calc_interface_flux!(surface_flux_values, mesh::TreeMesh{2},
                              equations_parabolic,
                              dg::DG, parabolic_scheme, cache)
    @unpack neighbor_ids, orientations = cache.interfaces

    @threaded for interface in eachinterface(dg, cache)
        # Get neighboring elements
        left_id = neighbor_ids[1, interface]
        right_id = neighbor_ids[2, interface]

        # Determine interface direction with respect to elements:
        # orientation = 1: left -> 2, right -> 1
        # orientation = 2: left -> 4, right -> 3
        left_direction = 2 * orientations[interface]
        right_direction = 2 * orientations[interface] - 1

        for i in eachnode(dg)
            # Get precomputed fluxes at interfaces
            flux_ll, flux_rr = get_surface_node_vars(cache.interfaces.u,
                                                     equations_parabolic,
                                                     dg, i, interface)

            # Compute interface flux as mean of left and right viscous fluxes
            flux = flux_parabolic(flux_ll, flux_rr, Divergence(),
                                  mesh, equations_parabolic, parabolic_scheme)

            # Copy flux to left and right element storage
            for v in eachvariable(equations_parabolic)
                surface_flux_values[v, i, left_direction, left_id] = flux[v]
                surface_flux_values[v, i, right_direction, right_id] = flux[v]
            end
        end
    end

    return nothing
end

# This is the version used when calculating the divergence of the viscous fluxes.
# Specialization `flux_viscous::Tuple` needed to
# avoid amibiguity with the hyperbolic version of `prolong2boundaries!` in dg_2d.jl
# which is for the variables itself, i.e., `u::Array{uEltype, 4}`.
function prolong2boundaries!(cache, flux_viscous::Tuple,
                             mesh::TreeMesh{2},
                             equations_parabolic::AbstractEquationsParabolic,
                             surface_integral, dg::DG)
    @unpack boundaries = cache
    @unpack orientations, neighbor_sides, neighbor_ids = boundaries
    boundaries_u = boundaries.u
    flux_viscous_x, flux_viscous_y = flux_viscous

    @threaded for boundary in eachboundary(dg, cache)
        element = neighbor_ids[boundary]

        if orientations[boundary] == 1
            # boundary in x-direction
            if neighbor_sides[boundary] == 1
                # element in -x direction of boundary
                for l in eachnode(dg), v in eachvariable(equations_parabolic)
                    # OBS! `boundaries_u` stores the interpolated *fluxes* and *not the solution*!
                    boundaries_u[1, v, l, boundary] = flux_viscous_x[v, nnodes(dg), l,
                                                                     element]
                end
            else # Element in +x direction of boundary
                for l in eachnode(dg), v in eachvariable(equations_parabolic)
                    # OBS! `boundaries_u` stores the interpolated *fluxes* and *not the solution*!
                    boundaries_u[2, v, l, boundary] = flux_viscous_x[v, 1, l,
                                                                     element]
                end
            end
        else # if orientations[boundary] == 2
            # boundary in y-direction
            if neighbor_sides[boundary] == 1
                # element in -y direction of boundary
                for l in eachnode(dg), v in eachvariable(equations_parabolic)
                    # OBS! `boundaries_u` stores the interpolated *fluxes* and *not the solution*!
                    boundaries_u[1, v, l, boundary] = flux_viscous_y[v, l, nnodes(dg),
                                                                     element]
                end
            else
                # element in +y direction of boundary
                for l in eachnode(dg), v in eachvariable(equations_parabolic)
                    # OBS! `boundaries_u` stores the interpolated *fluxes* and *not the solution*!
                    boundaries_u[2, v, l, boundary] = flux_viscous_y[v, l, 1,
                                                                     element]
                end
            end
        end
    end

    return nothing
end

function calc_viscous_fluxes!(flux_viscous,
                              gradients, u_transformed,
                              mesh::Union{TreeMesh{2}, P4estMesh{2}},
                              equations_parabolic::AbstractEquationsParabolic,
                              dg::DG, cache)
    gradients_x, gradients_y = gradients
    flux_viscous_x, flux_viscous_y = flux_viscous # output arrays

    @threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            # Get solution and gradients
            u_node = get_node_vars(u_transformed, equations_parabolic, dg,
                                   i, j, element)
            gradients_1_node = get_node_vars(gradients_x, equations_parabolic, dg,
                                             i, j, element)
            gradients_2_node = get_node_vars(gradients_y, equations_parabolic, dg,
                                             i, j, element)

            # Calculate viscous flux and store each component for later use
            flux_viscous_node_x = flux(u_node, (gradients_1_node, gradients_2_node), 1,
                                       equations_parabolic)
            flux_viscous_node_y = flux(u_node, (gradients_1_node, gradients_2_node), 2,
                                       equations_parabolic)
            set_node_vars!(flux_viscous_x, flux_viscous_node_x, equations_parabolic, dg,
                           i, j, element)
            set_node_vars!(flux_viscous_y, flux_viscous_node_y, equations_parabolic, dg,
                           i, j, element)
        end
    end

    return nothing
end

# TODO: parabolic; decide if we should keep this, and if so, extend to 3D.
function get_unsigned_normal_vector_2d(direction)
    if direction > 4 || direction < 1
        error("Direction = $direction; in 2D, direction should be 1, 2, 3, or 4.")
    end
    if direction == 1 || direction == 2
        return SVector(1.0, 0.0)
    else
        return SVector(0.0, 1.0)
    end
end

function calc_gradient_boundary_flux!(cache, t,
                                      boundary_conditions_parabolic::BoundaryConditionPeriodic,
                                      mesh::Union{TreeMesh{2}, P4estMesh{2}},
                                      equations_parabolic::AbstractEquationsParabolic,
                                      surface_integral, dg::DG)
    return nothing
end

function calc_boundary_flux_divergence!(cache, t,
                                        boundary_conditions_parabolic::BoundaryConditionPeriodic,
                                        mesh::Union{TreeMesh{2}, P4estMesh{2}},
                                        equations_parabolic::AbstractEquationsParabolic,
                                        surface_integral, dg::DG)
    return nothing
end

function calc_gradient_boundary_flux!(cache, t,
                                      boundary_conditions_parabolic::NamedTuple,
                                      mesh::TreeMesh{2}, # for dispatch only
                                      equations_parabolic::AbstractEquationsParabolic,
                                      surface_integral, dg::DG)
    @unpack surface_flux_values = cache.elements
    @unpack n_boundaries_per_direction = cache.boundaries

    # Calculate indices
    lasts = accumulate(+, n_boundaries_per_direction)
    firsts = lasts - n_boundaries_per_direction .+ 1

    # Calc boundary fluxes in each direction
    calc_gradient_boundary_flux_by_direction!(surface_flux_values, t,
                                              boundary_conditions_parabolic[1],
                                              equations_parabolic, surface_integral, dg,
                                              cache,
                                              1, firsts[1], lasts[1])
    calc_gradient_boundary_flux_by_direction!(surface_flux_values, t,
                                              boundary_conditions_parabolic[2],
                                              equations_parabolic, surface_integral, dg,
                                              cache,
                                              2, firsts[2], lasts[2])
    calc_gradient_boundary_flux_by_direction!(surface_flux_values, t,
                                              boundary_conditions_parabolic[3],
                                              equations_parabolic, surface_integral, dg,
                                              cache,
                                              3, firsts[3], lasts[3])
    calc_gradient_boundary_flux_by_direction!(surface_flux_values, t,
                                              boundary_conditions_parabolic[4],
                                              equations_parabolic, surface_integral, dg,
                                              cache,
                                              4, firsts[4], lasts[4])

    return nothing
end

function calc_gradient_boundary_flux_by_direction!(surface_flux_values::AbstractArray{<:Any,
                                                                                      4},
                                                   t,
                                                   boundary_condition,
                                                   equations_parabolic::AbstractEquationsParabolic,
                                                   surface_integral, dg::DG, cache,
                                                   direction, first_boundary,
                                                   last_boundary)
    @unpack surface_flux = surface_integral
    @unpack u, neighbor_ids, neighbor_sides, node_coordinates, orientations = cache.boundaries

    @threaded for boundary in first_boundary:last_boundary
        # Get neighboring element
        neighbor = neighbor_ids[boundary]

        for i in eachnode(dg)
            # Get boundary flux
            u_ll, u_rr = get_surface_node_vars(u, equations_parabolic, dg, i, boundary)
            if neighbor_sides[boundary] == 1 # Element is on the left, boundary on the right
                u_inner = u_ll
            else # Element is on the right, boundary on the left
                u_inner = u_rr
            end

            # TODO: revisit if we want more general boundary treatments.
            # This assumes the gradient numerical flux at the boundary is the gradient variable,
            # which is consistent with BR1, LDG.
            flux_inner = u_inner

            x = get_node_coords(node_coordinates, equations_parabolic, dg, i, boundary)
            flux = boundary_condition(flux_inner, u_inner,
                                      get_unsigned_normal_vector_2d(direction),
                                      x, t, Gradient(), equations_parabolic)

            # Copy flux to left and right element storage
            for v in eachvariable(equations_parabolic)
                surface_flux_values[v, i, direction, neighbor] = flux[v]
            end
        end
    end

    return nothing
end

function calc_boundary_flux_divergence!(cache, t,
                                        boundary_conditions_parabolic::NamedTuple,
                                        mesh::TreeMesh{2},
                                        equations_parabolic::AbstractEquationsParabolic,
                                        surface_integral, dg::DG)
    @unpack surface_flux_values = cache.elements
    @unpack n_boundaries_per_direction = cache.boundaries

    # Calculate indices
    lasts = accumulate(+, n_boundaries_per_direction)
    firsts = lasts - n_boundaries_per_direction .+ 1

    # Calc boundary fluxes in each direction
    calc_boundary_flux_by_direction_divergence!(surface_flux_values, t,
                                                boundary_conditions_parabolic[1],
                                                equations_parabolic, surface_integral,
                                                dg, cache,
                                                1, firsts[1], lasts[1])
    calc_boundary_flux_by_direction_divergence!(surface_flux_values, t,
                                                boundary_conditions_parabolic[2],
                                                equations_parabolic, surface_integral,
                                                dg, cache,
                                                2, firsts[2], lasts[2])
    calc_boundary_flux_by_direction_divergence!(surface_flux_values, t,
                                                boundary_conditions_parabolic[3],
                                                equations_parabolic, surface_integral,
                                                dg, cache,
                                                3, firsts[3], lasts[3])
    calc_boundary_flux_by_direction_divergence!(surface_flux_values, t,
                                                boundary_conditions_parabolic[4],
                                                equations_parabolic, surface_integral,
                                                dg, cache,
                                                4, firsts[4], lasts[4])

    return nothing
end
function calc_boundary_flux_by_direction_divergence!(surface_flux_values::AbstractArray{<:Any,
                                                                                        4},
                                                     t,
                                                     boundary_condition,
                                                     equations_parabolic::AbstractEquationsParabolic,
                                                     surface_integral, dg::DG, cache,
                                                     direction, first_boundary,
                                                     last_boundary)
    @unpack surface_flux = surface_integral

    # Note: cache.boundaries.u contains the unsigned normal component (using "orientation", not "direction")
    # of the viscous flux, as computed in `prolong2boundaries!`
    @unpack u, neighbor_ids, neighbor_sides, node_coordinates, orientations = cache.boundaries

    @threaded for boundary in first_boundary:last_boundary
        # Get neighboring element
        neighbor = neighbor_ids[boundary]

        for i in eachnode(dg)
            # Get viscous boundary fluxes
            flux_ll, flux_rr = get_surface_node_vars(u, equations_parabolic, dg, i,
                                                     boundary)
            if neighbor_sides[boundary] == 1 # Element is on the left, boundary on the right
                flux_inner = flux_ll
            else # Element is on the right, boundary on the left
                flux_inner = flux_rr
            end

            x = get_node_coords(node_coordinates, equations_parabolic, dg, i, boundary)

            # TODO: add a field in `cache.boundaries` for gradient information.
            # Here, we pass in `u_inner = nothing` since we overwrite cache.boundaries.u with gradient information.
            # This currently works with Dirichlet/Neuman boundary conditions for LaplaceDiffusion2D and
            # NoSlipWall/Adiabatic boundary conditions for CompressibleNavierStokesDiffusion2D as of 2022-6-27.
            # It will not work with implementations which utilize `u_inner` to impose boundary conditions, such as
            # the `Slip` boundary condition, which can be imposed to realize reflective or symmetric boundaries.
            flux = boundary_condition(flux_inner, nothing,
                                      get_unsigned_normal_vector_2d(direction),
                                      x, t, Divergence(), equations_parabolic)

            # Copy flux to left and right element storage
            for v in eachvariable(equations_parabolic)
                surface_flux_values[v, i, direction, neighbor] = flux[v]
            end
        end
    end

    return nothing
end

# Specialization `flux_viscous::Tuple` needed to
# avoid amibiguity with the hyperbolic version of `prolong2mortars!` in dg_2d.jl
# which is for the variables itself, i.e., `u::Array{uEltype, 4}`.
function prolong2mortars!(cache, flux_viscous::Tuple,
                          mesh::TreeMesh{2},
                          equations_parabolic::AbstractEquationsParabolic,
                          mortar_l2::LobattoLegendreMortarL2,
                          dg::DGSEM)
    flux_viscous_x, flux_viscous_y = flux_viscous
    @threaded for mortar in eachmortar(dg, cache)
        large_element = cache.mortars.neighbor_ids[3, mortar]
        upper_element = cache.mortars.neighbor_ids[2, mortar]
        lower_element = cache.mortars.neighbor_ids[1, mortar]

        # Copy solution small to small
        if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
            if cache.mortars.orientations[mortar] == 1
                # L2 mortars in x-direction
                for l in eachnode(dg)
                    for v in eachvariable(equations_parabolic)
                        cache.mortars.u_upper[2, v, l, mortar] = flux_viscous_x[v,
                                                                                1,
                                                                                l,
                                                                                upper_element]
                        cache.mortars.u_lower[2, v, l, mortar] = flux_viscous_x[v,
                                                                                1,
                                                                                l,
                                                                                lower_element]
                    end
                end
            else
                # L2 mortars in y-direction
                for l in eachnode(dg)
                    for v in eachvariable(equations_parabolic)
                        cache.mortars.u_upper[2, v, l, mortar] = flux_viscous_y[v,
                                                                                l,
                                                                                1,
                                                                                upper_element]
                        cache.mortars.u_lower[2, v, l, mortar] = flux_viscous_y[v,
                                                                                l,
                                                                                1,
                                                                                lower_element]
                    end
                end
            end
        else # large_sides[mortar] == 2 -> small elements on left side
            if cache.mortars.orientations[mortar] == 1
                # L2 mortars in x-direction
                for l in eachnode(dg)
                    for v in eachvariable(equations_parabolic)
                        cache.mortars.u_upper[1, v, l, mortar] = flux_viscous_x[v,
                                                                                nnodes(dg),
                                                                                l,
                                                                                upper_element]
                        cache.mortars.u_lower[1, v, l, mortar] = flux_viscous_x[v,
                                                                                nnodes(dg),
                                                                                l,
                                                                                lower_element]
                    end
                end
            else
                # L2 mortars in y-direction
                for l in eachnode(dg)
                    for v in eachvariable(equations_parabolic)
                        cache.mortars.u_upper[1, v, l, mortar] = flux_viscous_y[v,
                                                                                l,
                                                                                nnodes(dg),
                                                                                upper_element]
                        cache.mortars.u_lower[1, v, l, mortar] = flux_viscous_y[v,
                                                                                l,
                                                                                nnodes(dg),
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
                u_large = view(flux_viscous_x, :, nnodes(dg), :, large_element)
                element_solutions_to_mortars!(cache.mortars, mortar_l2, leftright,
                                              mortar, u_large)
            else
                # L2 mortars in y-direction
                u_large = view(flux_viscous_y, :, :, nnodes(dg), large_element)
                element_solutions_to_mortars!(cache.mortars, mortar_l2, leftright,
                                              mortar, u_large)
            end
        else # large_sides[mortar] == 2 -> large element on right side
            leftright = 2
            if cache.mortars.orientations[mortar] == 1
                # L2 mortars in x-direction
                u_large = view(flux_viscous_x, :, 1, :, large_element)
                element_solutions_to_mortars!(cache.mortars, mortar_l2, leftright,
                                              mortar, u_large)
            else
                # L2 mortars in y-direction
                u_large = view(flux_viscous_y, :, :, 1, large_element)
                element_solutions_to_mortars!(cache.mortars, mortar_l2, leftright,
                                              mortar, u_large)
            end
        end
    end

    return nothing
end

# NOTE: Use analogy to "calc_mortar_flux!" for hyperbolic eqs with no nonconservative terms.
# Reasoning: "calc_interface_flux!" for parabolic part is implemented as the version for
# hyperbolic terms with conserved terms only, i.e., no nonconservative terms.
function calc_mortar_flux!(surface_flux_values, mesh::TreeMesh{2},
                           equations_parabolic::AbstractEquationsParabolic,
                           mortar_l2::LobattoLegendreMortarL2,
                           surface_integral, dg::DG,
                           parabolic_scheme, gradient_or_divergence,
                           cache)
    @unpack surface_flux = surface_integral
    @unpack u_lower, u_upper, orientations = cache.mortars
    @unpack fstar_primary_upper_threaded, fstar_primary_lower_threaded = cache

    @threaded for mortar in eachmortar(dg, cache)
        # Choose thread-specific pre-allocated container
        fstar_upper = fstar_primary_upper_threaded[Threads.threadid()]
        fstar_lower = fstar_primary_lower_threaded[Threads.threadid()]

        # Calculate fluxes
        orientation = orientations[mortar]
        calc_fstar!(fstar_upper, mesh, equations_parabolic, surface_flux,
                    dg, parabolic_scheme, gradient_or_divergence,
                    u_upper, mortar, orientation)
        calc_fstar!(fstar_lower, mesh, equations_parabolic, surface_flux,
                    dg, parabolic_scheme, gradient_or_divergence,
                    u_lower, mortar, orientation)

        mortar_fluxes_to_elements!(surface_flux_values,
                                   mesh, equations_parabolic, mortar_l2, dg, cache,
                                   mortar, fstar_upper, fstar_lower)
    end

    return nothing
end

@inline function calc_fstar!(destination::AbstractArray{<:Any, 2},
                             mesh, equations_parabolic::AbstractEquationsParabolic,
                             surface_flux, dg::DGSEM,
                             parabolic_scheme, gradient_or_divergence,
                             u_interfaces, interface, orientation)
    for i in eachnode(dg)
        # Call pointwise two-point numerical flux function
        u_ll, u_rr = get_surface_node_vars(u_interfaces, equations_parabolic, dg, i,
                                           interface)

        flux = flux_parabolic(u_ll, u_rr, gradient_or_divergence,
                              mesh, equations_parabolic, parabolic_scheme)

        # Copy flux to left and right element storage
        set_node_vars!(destination, flux, equations_parabolic, dg, i)
    end

    return nothing
end

@inline function mortar_fluxes_to_elements!(surface_flux_values,
                                            mesh::TreeMesh{2},
                                            equations_parabolic::AbstractEquationsParabolic,
                                            mortar_l2::LobattoLegendreMortarL2,
                                            dg::DGSEM, cache,
                                            mortar, fstar_upper, fstar_lower)
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
    surface_flux_values[:, :, direction, upper_element] .= fstar_upper
    surface_flux_values[:, :, direction, lower_element] .= fstar_lower

    # Project small fluxes to large element
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
                            mortar_l2.reverse_upper, fstar_upper,
                            mortar_l2.reverse_lower, fstar_lower)

    return nothing
end

function calc_gradient_volume_integral!(gradients, u_transformed,
                                        mesh::TreeMesh{2}, # for dispatch only
                                        equations_parabolic::AbstractEquationsParabolic,
                                        dg::DGSEM, cache)
    @unpack derivative_dhat = dg.basis
    gradients_x, gradients_y = gradients

    @threaded for element in eachelement(dg, cache)
        # Calculate volume terms in one element,
        # corresponds to `kernel` functions for the hyperbolic part of the flux
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u_transformed, equations_parabolic, dg,
                                   i, j, element)

            for ii in eachnode(dg)
                multiply_add_to_node_vars!(gradients_x, derivative_dhat[ii, i],
                                           u_node, equations_parabolic, dg,
                                           ii, j, element)
            end

            for jj in eachnode(dg)
                multiply_add_to_node_vars!(gradients_y, derivative_dhat[jj, j],
                                           u_node, equations_parabolic, dg,
                                           i, jj, element)
            end
        end
    end

    return nothing
end

function calc_gradient_interface_flux!(surface_flux_values,
                                       mesh::TreeMesh{2}, equations, dg::DG,
                                       parabolic_scheme, cache)
    @unpack neighbor_ids, orientations = cache.interfaces

    @threaded for interface in eachinterface(dg, cache)
        # Get neighboring elements
        left_id = neighbor_ids[1, interface]
        right_id = neighbor_ids[2, interface]

        # Determine interface direction with respect to elements:
        # orientation = 1: left -> 2, right -> 1
        # orientation = 2: left -> 4, right -> 3
        left_direction = 2 * orientations[interface]
        right_direction = 2 * orientations[interface] - 1

        for i in eachnode(dg)
            # Call pointwise Riemann solver
            u_ll, u_rr = get_surface_node_vars(cache.interfaces.u,
                                               equations, dg, i,
                                               interface)

            flux = flux_parabolic(u_ll, u_rr, Gradient(),
                                  mesh, equations, parabolic_scheme)

            # Copy flux to left and right element storage
            for v in eachvariable(equations)
                surface_flux_values[v, i, left_direction, left_id] = flux[v]
                surface_flux_values[v, i, right_direction, right_id] = flux[v]
            end
        end
    end

    return nothing
end

function calc_gradient_surface_integral!(gradients,
                                         mesh::TreeMesh{2}, # for dispatch only
                                         equations_parabolic::AbstractEquationsParabolic,
                                         dg::DGSEM, cache)
    @unpack boundary_interpolation = dg.basis
    @unpack surface_flux_values = cache.elements

    gradients_x, gradients_y = gradients

    # Note that all fluxes have been computed with outward-pointing normal vectors.
    # Access the factors only once before beginning the loop to increase performance.
    # We also use explicit assignments instead of `+=` to let `@muladd` turn these
    # into FMAs (see comment at the top of the file).
    factor_1 = boundary_interpolation[1, 1]
    factor_2 = boundary_interpolation[nnodes(dg), 2]

    @threaded for element in eachelement(dg, cache)
        for l in eachnode(dg)
            for v in eachvariable(equations_parabolic)
                # surface at -x
                gradients_x[v, 1, l, element] = (gradients_x[v,
                                                             1, l,
                                                             element] -
                                                 surface_flux_values[v,
                                                                     l, 1,
                                                                     element] *
                                                 factor_1)

                # surface at +x
                gradients_x[v, nnodes(dg), l, element] = (gradients_x[v,
                                                                      nnodes(dg), l,
                                                                      element] +
                                                          surface_flux_values[v,
                                                                              l, 2,
                                                                              element] *
                                                          factor_2)

                # surface at -y
                gradients_y[v, l, 1, element] = (gradients_y[v,
                                                             l, 1,
                                                             element] -
                                                 surface_flux_values[v,
                                                                     l, 3,
                                                                     element] *
                                                 factor_1)

                # surface at +y
                gradients_y[v, l, nnodes(dg), element] = (gradients_y[v,
                                                                      l, nnodes(dg),
                                                                      element] +
                                                          surface_flux_values[v,
                                                                              l, 4,
                                                                              element] *
                                                          factor_2)
            end
        end
    end

    return nothing
end

# Calculate the gradient of the transformed variables
function calc_gradient!(gradients, u_transformed, t,
                        mesh::TreeMesh{2}, equations_parabolic,
                        boundary_conditions_parabolic, dg::DG, parabolic_scheme,
                        cache)
    gradients_x, gradients_y = gradients

    # Reset gradients
    @trixi_timeit timer() "reset gradients" begin
        reset_du!(gradients_x, dg, cache)
        reset_du!(gradients_y, dg, cache)
    end

    # Calculate volume integral
    @trixi_timeit timer() "volume integral" begin
        calc_gradient_volume_integral!(gradients, u_transformed,
                                       mesh, equations_parabolic, dg, cache)
    end

    # Prolong solution to interfaces
    # This reuses `prolong2interfaces!` for the purely hyperbolic case.
    @trixi_timeit timer() "prolong2interfaces" begin
        prolong2interfaces!(cache, u_transformed, mesh,
                            equations_parabolic, dg)
    end

    # Calculate interface fluxes
    @trixi_timeit timer() "interface flux" begin
        @unpack surface_flux_values = cache.elements
        calc_gradient_interface_flux!(surface_flux_values, mesh, equations_parabolic,
                                      dg, parabolic_scheme, cache)
    end

    # Prolong solution to boundaries
    # This reuses `prolong2boundaries!` for the purely hyperbolic case.
    @trixi_timeit timer() "prolong2boundaries" begin
        prolong2boundaries!(cache, u_transformed, mesh, equations_parabolic,
                            dg.surface_integral, dg)
    end

    # Calculate boundary fluxes
    @trixi_timeit timer() "boundary flux" begin
        calc_gradient_boundary_flux!(cache, t,
                                     boundary_conditions_parabolic, mesh,
                                     equations_parabolic,
                                     dg.surface_integral, dg)
    end

    # Prolong solution to mortars.
    # This reuses `prolong2mortars` for the purely hyperbolic case.
    @trixi_timeit timer() "prolong2mortars" begin
        prolong2mortars!(cache, u_transformed, mesh, equations_parabolic,
                         dg.mortar, dg)
    end

    # Calculate mortar fluxes
    @trixi_timeit timer() "mortar flux" begin
        calc_mortar_flux!(surface_flux_values, mesh, equations_parabolic,
                          dg.mortar, dg.surface_integral, dg,
                          parabolic_scheme, Gradient(), cache)
    end

    # Calculate surface integrals
    @trixi_timeit timer() "surface integral" begin
        calc_gradient_surface_integral!(gradients, mesh, equations_parabolic,
                                        dg, cache)
    end

    # Apply Jacobian from mapping to reference element
    @trixi_timeit timer() "Jacobian" begin
        apply_jacobian_parabolic!(gradients_x, mesh, equations_parabolic, dg,
                                  cache)
        apply_jacobian_parabolic!(gradients_y, mesh, equations_parabolic, dg,
                                  cache)
    end

    return nothing
end

# This method is called when a `SemidiscretizationHyperbolicParabolic` is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache_parabolic(mesh::TreeMesh{2},
                                equations_hyperbolic::AbstractEquations,
                                dg::DG, n_elements, uEltype)
    viscous_container = init_viscous_container_2d(nvariables(equations_hyperbolic),
                                                  nnodes(dg), n_elements,
                                                  uEltype)

    cache_parabolic = (; viscous_container)

    return cache_parabolic
end

# Needed to *not* flip the sign of the inverse Jacobian.
# This is because the parabolic fluxes are assumed to be of the form
#   `du/dt + df/dx = dg/dx + source(x,t)`,
# where f(u) is the inviscid flux and g(u) is the viscous flux.
function apply_jacobian_parabolic!(du, mesh::TreeMesh{2},
                                   equations::AbstractEquationsParabolic, dg::DG, cache)
    @unpack inverse_jacobian = cache.elements

    @threaded for element in eachelement(dg, cache)
        factor = inverse_jacobian[element]

        for j in eachnode(dg), i in eachnode(dg)
            for v in eachvariable(equations)
                du[v, i, j, element] *= factor
            end
        end
    end

    return nothing
end
end # @muladd
