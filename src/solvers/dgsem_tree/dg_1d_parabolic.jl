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
function rhs_parabolic!(du, u, t, mesh::TreeMesh{1},
                        equations_parabolic::AbstractEquationsParabolic,
                        boundary_conditions_parabolic, source_terms,
                        dg::DG, parabolic_scheme, cache, cache_parabolic)
    @unpack viscous_container = cache_parabolic
    @unpack u_transformed, gradients, flux_viscous = viscous_container

    # Convert conservative variables to a form more suitable for viscous flux calculations
    @trixi_timeit timer() "transform variables" begin
        transform_variables!(u_transformed, u, mesh, equations_parabolic,
                             dg, parabolic_scheme, cache, cache_parabolic)
    end

    # Compute the gradients of the transformed variables
    @trixi_timeit timer() "calculate gradient" begin
        calc_gradient!(gradients, u_transformed, t, mesh, equations_parabolic,
                       boundary_conditions_parabolic, dg, cache, cache_parabolic)
    end

    # Compute and store the viscous fluxes
    @trixi_timeit timer() "calculate viscous fluxes" begin
        calc_viscous_fluxes!(flux_viscous, gradients, u_transformed, mesh,
                             equations_parabolic, dg, cache, cache_parabolic)
    end

    # The remainder of this function is essentially a regular rhs! for
    # parabolic equations (i.e., it computes the divergence of the viscous fluxes)
    #
    # OBS! In `calc_viscous_fluxes!`, the viscous flux values at the volume nodes of each element have
    # been computed and stored in `fluxes_viscous`. In the following, we *reuse* (abuse) the
    # `interfaces` and `boundaries` containers in `cache_parabolic` to interpolate and store the
    # *fluxes* at the element surfaces, as opposed to interpolating and storing the *solution* (as it
    # is done in the hyperbolic operator). That is, `interfaces.u`/`boundaries.u` store *viscous flux values*
    # and *not the solution*.  The advantage is that a) we do not need to allocate more storage, b) we
    # do not need to recreate the existing data structure only with a different name, and c) we do not
    # need to interpolate solutions *and* gradients to the surfaces.

    # TODO: parabolic; reconsider current data structure reuse strategy

    # Reset du
    @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

    # Calculate volume integral
    @trixi_timeit timer() "volume integral" begin
        calc_volume_integral!(du, flux_viscous, mesh, equations_parabolic, dg, cache)
    end

    # Prolong solution to interfaces
    @trixi_timeit timer() "prolong2interfaces" begin
        prolong2interfaces!(cache_parabolic, flux_viscous, mesh, equations_parabolic,
                            dg.surface_integral, dg, cache)
    end

    # Calculate interface fluxes
    @trixi_timeit timer() "interface flux" begin
        calc_interface_flux!(cache_parabolic.elements.surface_flux_values, mesh,
                             equations_parabolic, dg, cache_parabolic)
    end

    # Prolong solution to boundaries
    @trixi_timeit timer() "prolong2boundaries" begin
        prolong2boundaries!(cache_parabolic, flux_viscous, mesh, equations_parabolic,
                            dg.surface_integral, dg, cache)
    end

    # Calculate boundary fluxes
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux_divergence!(cache_parabolic, t,
                                       boundary_conditions_parabolic, mesh,
                                       equations_parabolic,
                                       dg.surface_integral, dg)
    end

    # Calculate surface integrals
    @trixi_timeit timer() "surface integral" begin
        calc_surface_integral!(du, u, mesh, equations_parabolic,
                               dg.surface_integral, dg, cache_parabolic)
    end

    # Apply Jacobian from mapping to reference element
    @trixi_timeit timer() "Jacobian" begin
        apply_jacobian_parabolic!(du, mesh, equations_parabolic, dg, cache_parabolic)
    end

    return nothing
end

# Transform solution variables prior to taking the gradient
# (e.g., conservative to primitive variables). Defaults to doing nothing.
# TODO: can we avoid copying data?
function transform_variables!(u_transformed, u, mesh::TreeMesh{1},
                              equations_parabolic::AbstractEquationsParabolic,
                              dg::DG, parabolic_scheme, cache, cache_parabolic)
    transformation = gradient_variable_transformation(equations_parabolic)

    @threaded for element in eachelement(dg, cache)
        # Calculate volume terms in one element
        for i in eachnode(dg)
            u_node = get_node_vars(u, equations_parabolic, dg, i, element)
            u_transformed_node = transformation(u_node, equations_parabolic)
            set_node_vars!(u_transformed, u_transformed_node, equations_parabolic, dg,
                           i, element)
        end
    end
end

# This is the version used when calculating the divergence of the viscous fluxes
function calc_volume_integral!(du, flux_viscous,
                               mesh::TreeMesh{1},
                               equations_parabolic::AbstractEquationsParabolic,
                               dg::DGSEM, cache)
    @unpack derivative_dhat = dg.basis

    @threaded for element in eachelement(dg, cache)
        # Calculate volume terms in one element
        for i in eachnode(dg)
            flux_1_node = get_node_vars(flux_viscous, equations_parabolic, dg, i,
                                        element)

            for ii in eachnode(dg)
                multiply_add_to_node_vars!(du, derivative_dhat[ii, i], flux_1_node,
                                           equations_parabolic, dg, ii, element)
            end
        end
    end

    return nothing
end

# This is the version used when calculating the divergence of the viscous fluxes
# We pass the `surface_integral` argument solely for dispatch
function prolong2interfaces!(cache_parabolic, flux_viscous,
                             mesh::TreeMesh{1},
                             equations_parabolic::AbstractEquationsParabolic,
                             surface_integral, dg::DG, cache)
    @unpack interfaces = cache_parabolic
    @unpack neighbor_ids = interfaces
    interfaces_u = interfaces.u

    @threaded for interface in eachinterface(dg, cache)
        left_element = neighbor_ids[1, interface]
        right_element = neighbor_ids[2, interface]

        # interface in x-direction
        for v in eachvariable(equations_parabolic)
            # OBS! `interfaces_u` stores the interpolated *fluxes* and *not the solution*!
            interfaces_u[1, v, interface] = flux_viscous[v, nnodes(dg), left_element]
            interfaces_u[2, v, interface] = flux_viscous[v, 1, right_element]
        end
    end

    return nothing
end

# This is the version used when calculating the divergence of the viscous fluxes
function calc_interface_flux!(surface_flux_values,
                              mesh::TreeMesh{1}, equations_parabolic,
                              dg::DG, cache_parabolic)
    @unpack neighbor_ids, orientations = cache_parabolic.interfaces

    @threaded for interface in eachinterface(dg, cache_parabolic)
        # Get neighboring elements
        left_id = neighbor_ids[1, interface]
        right_id = neighbor_ids[2, interface]

        # Determine interface direction with respect to elements:
        # orientation = 1: left -> 2, right -> 1
        left_direction = 2 * orientations[interface]
        right_direction = 2 * orientations[interface] - 1

        # Get precomputed fluxes at interfaces
        flux_ll, flux_rr = get_surface_node_vars(cache_parabolic.interfaces.u,
                                                 equations_parabolic,
                                                 dg, interface)

        # Compute interface flux as mean of left and right viscous fluxes
        # TODO: parabolic; only BR1 at the moment
        flux = 0.5f0 * (flux_ll + flux_rr)

        # Copy flux to left and right element storage
        for v in eachvariable(equations_parabolic)
            surface_flux_values[v, left_direction, left_id] = flux[v]
            surface_flux_values[v, right_direction, right_id] = flux[v]
        end
    end

    return nothing
end

# This is the version used when calculating the divergence of the viscous fluxes
function prolong2boundaries!(cache_parabolic, flux_viscous,
                             mesh::TreeMesh{1},
                             equations_parabolic::AbstractEquationsParabolic,
                             surface_integral, dg::DG, cache)
    @unpack boundaries = cache_parabolic
    @unpack neighbor_sides, neighbor_ids = boundaries
    boundaries_u = boundaries.u

    @threaded for boundary in eachboundary(dg, cache_parabolic)
        element = neighbor_ids[boundary]

        if neighbor_sides[boundary] == 1
            # element in -x direction of boundary
            for v in eachvariable(equations_parabolic)
                # OBS! `boundaries_u` stores the interpolated *fluxes* and *not the solution*!
                boundaries_u[1, v, boundary] = flux_viscous[v, nnodes(dg), element]
            end
        else # Element in +x direction of boundary
            for v in eachvariable(equations_parabolic)
                # OBS! `boundaries_u` stores the interpolated *fluxes* and *not the solution*!
                boundaries_u[2, v, boundary] = flux_viscous[v, 1, element]
            end
        end
    end

    return nothing
end

function calc_viscous_fluxes!(flux_viscous, gradients, u_transformed, mesh::TreeMesh{1},
                              equations_parabolic::AbstractEquationsParabolic,
                              dg::DG, cache, cache_parabolic)
    @threaded for element in eachelement(dg, cache)
        for i in eachnode(dg)
            # Get solution and gradients
            u_node = get_node_vars(u_transformed, equations_parabolic, dg, i, element)
            gradients_1_node = get_node_vars(gradients, equations_parabolic, dg, i,
                                             element)

            # Calculate viscous flux and store each component for later use
            flux_viscous_node = flux(u_node, gradients_1_node, 1, equations_parabolic)
            set_node_vars!(flux_viscous, flux_viscous_node, equations_parabolic, dg, i,
                           element)
        end
    end
end

function calc_boundary_flux_gradients!(cache, t,
                                       boundary_conditions_parabolic::BoundaryConditionPeriodic,
                                       mesh::TreeMesh{1},
                                       equations_parabolic::AbstractEquationsParabolic,
                                       surface_integral, dg::DG)
    return nothing
end

function calc_boundary_flux_divergence!(cache, t,
                                        boundary_conditions_parabolic::BoundaryConditionPeriodic,
                                        mesh::TreeMesh{1},
                                        equations_parabolic::AbstractEquationsParabolic,
                                        surface_integral, dg::DG)
    return nothing
end

function calc_boundary_flux_gradients!(cache, t,
                                       boundary_conditions_parabolic::NamedTuple,
                                       mesh::TreeMesh{1},
                                       equations_parabolic::AbstractEquationsParabolic,
                                       surface_integral, dg::DG)
    @unpack surface_flux_values = cache.elements
    @unpack n_boundaries_per_direction = cache.boundaries

    # Calculate indices
    lasts = accumulate(+, n_boundaries_per_direction)
    firsts = lasts - n_boundaries_per_direction .+ 1

    # Calc boundary fluxes in each direction
    calc_boundary_flux_by_direction_gradient!(surface_flux_values, t,
                                              boundary_conditions_parabolic[1],
                                              equations_parabolic, surface_integral, dg,
                                              cache,
                                              1, firsts[1], lasts[1])
    calc_boundary_flux_by_direction_gradient!(surface_flux_values, t,
                                              boundary_conditions_parabolic[2],
                                              equations_parabolic, surface_integral, dg,
                                              cache,
                                              2, firsts[2], lasts[2])
end

function calc_boundary_flux_by_direction_gradient!(surface_flux_values::AbstractArray{<:Any,
                                                                                      3},
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

        # Get boundary flux
        u_ll, u_rr = get_surface_node_vars(u, equations_parabolic, dg, boundary)
        if neighbor_sides[boundary] == 1 # Element is on the left, boundary on the right
            u_inner = u_ll
        else # Element is on the right, boundary on the left
            u_inner = u_rr
        end

        # TODO: revisit if we want more general boundary treatments.
        # This assumes the gradient numerical flux at the boundary is the gradient variable,
        # which is consistent with BR1, LDG.
        flux_inner = u_inner

        x = get_node_coords(node_coordinates, equations_parabolic, dg, boundary)
        flux = boundary_condition(flux_inner, u_inner, orientations[boundary],
                                  direction,
                                  x, t, Gradient(), equations_parabolic)

        # Copy flux to left and right element storage
        for v in eachvariable(equations_parabolic)
            surface_flux_values[v, direction, neighbor] = flux[v]
        end
    end

    return nothing
end

function calc_boundary_flux_divergence!(cache, t,
                                        boundary_conditions_parabolic::NamedTuple,
                                        mesh::TreeMesh{1},
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
end
function calc_boundary_flux_by_direction_divergence!(surface_flux_values::AbstractArray{<:Any,
                                                                                        3},
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

        # Get viscous boundary fluxes
        flux_ll, flux_rr = get_surface_node_vars(u, equations_parabolic, dg, boundary)
        if neighbor_sides[boundary] == 1 # Element is on the left, boundary on the right
            flux_inner = flux_ll
        else # Element is on the right, boundary on the left
            flux_inner = flux_rr
        end

        x = get_node_coords(node_coordinates, equations_parabolic, dg, boundary)

        # TODO: add a field in `cache.boundaries` for gradient information.
        # Here, we pass in `u_inner = nothing` since we overwrite cache.boundaries.u with gradient information.
        # This currently works with Dirichlet/Neuman boundary conditions for LaplaceDiffusion2D and
        # NoSlipWall/Adiabatic boundary conditions for CompressibleNavierStokesDiffusion2D as of 2022-6-27.
        # It will not work with implementations which utilize `u_inner` to impose boundary conditions.
        flux = boundary_condition(flux_inner, nothing, orientations[boundary],
                                  direction,
                                  x, t, Divergence(), equations_parabolic)

        # Copy flux to left and right element storage
        for v in eachvariable(equations_parabolic)
            surface_flux_values[v, direction, neighbor] = flux[v]
        end
    end

    return nothing
end

# Calculate the gradient of the transformed variables
function calc_gradient!(gradients, u_transformed, t,
                        mesh::TreeMesh{1}, equations_parabolic,
                        boundary_conditions_parabolic, dg::DG, cache, cache_parabolic)

    # Reset du
    @trixi_timeit timer() "reset gradients" begin
        reset_du!(gradients, dg, cache)
    end

    # Calculate volume integral
    @trixi_timeit timer() "volume integral" begin
        @unpack derivative_dhat = dg.basis
        @threaded for element in eachelement(dg, cache)

            # Calculate volume terms in one element
            for i in eachnode(dg)
                u_node = get_node_vars(u_transformed, equations_parabolic, dg, i,
                                       element)

                for ii in eachnode(dg)
                    multiply_add_to_node_vars!(gradients, derivative_dhat[ii, i],
                                               u_node, equations_parabolic, dg, ii,
                                               element)
                end
            end
        end
    end

    # Prolong solution to interfaces
    @trixi_timeit timer() "prolong2interfaces" prolong2interfaces!(cache_parabolic,
                                                                   u_transformed, mesh,
                                                                   equations_parabolic,
                                                                   dg.surface_integral,
                                                                   dg)

    # Calculate interface fluxes
    @trixi_timeit timer() "interface flux" begin
        @unpack surface_flux_values = cache_parabolic.elements
        @unpack neighbor_ids, orientations = cache_parabolic.interfaces

        @threaded for interface in eachinterface(dg, cache_parabolic)
            # Get neighboring elements
            left_id = neighbor_ids[1, interface]
            right_id = neighbor_ids[2, interface]

            # Determine interface direction with respect to elements:
            # orientation = 1: left -> 2, right -> 1
            left_direction = 2 * orientations[interface]
            right_direction = 2 * orientations[interface] - 1

            # Call pointwise Riemann solver
            u_ll, u_rr = get_surface_node_vars(cache_parabolic.interfaces.u,
                                               equations_parabolic, dg, interface)
            flux = 0.5f0 * (u_ll + u_rr)

            # Copy flux to left and right element storage
            for v in eachvariable(equations_parabolic)
                surface_flux_values[v, left_direction, left_id] = flux[v]
                surface_flux_values[v, right_direction, right_id] = flux[v]
            end
        end
    end

    # Prolong solution to boundaries
    @trixi_timeit timer() "prolong2boundaries" prolong2boundaries!(cache_parabolic,
                                                                   u_transformed, mesh,
                                                                   equations_parabolic,
                                                                   dg.surface_integral,
                                                                   dg)

    # Calculate boundary fluxes
    @trixi_timeit timer() "boundary flux" calc_boundary_flux_gradients!(cache_parabolic,
                                                                        t,
                                                                        boundary_conditions_parabolic,
                                                                        mesh,
                                                                        equations_parabolic,
                                                                        dg.surface_integral,
                                                                        dg)

    # Calculate surface integrals
    @trixi_timeit timer() "surface integral" begin
        @unpack boundary_interpolation = dg.basis
        @unpack surface_flux_values = cache_parabolic.elements

        # Note that all fluxes have been computed with outward-pointing normal vectors.
        # Access the factors only once before beginning the loop to increase performance.
        # We also use explicit assignments instead of `+=` to let `@muladd` turn these
        # into FMAs (see comment at the top of the file).
        factor_1 = boundary_interpolation[1, 1]
        factor_2 = boundary_interpolation[nnodes(dg), 2]
        @threaded for element in eachelement(dg, cache)
            for v in eachvariable(equations_parabolic)
                # surface at -x
                gradients[v, 1, element] = (gradients[v, 1, element] -
                                            surface_flux_values[v, 1, element] *
                                            factor_1)

                # surface at +x
                gradients[v, nnodes(dg), element] = (gradients[v, nnodes(dg), element] +
                                                     surface_flux_values[v, 2,
                                                                         element] *
                                                     factor_2)
            end
        end
    end

    # Apply Jacobian from mapping to reference element
    @trixi_timeit timer() "Jacobian" begin
        apply_jacobian_parabolic!(gradients, mesh, equations_parabolic, dg,
                                  cache_parabolic)
    end

    return nothing
end

# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache_parabolic(mesh::TreeMesh{1},
                                equations_hyperbolic::AbstractEquations,
                                equations_parabolic::AbstractEquationsParabolic,
                                dg::DG, parabolic_scheme, RealT, uEltype)
    # Get cells for which an element needs to be created (i.e. all leaf cells)
    leaf_cell_ids = local_leaf_cells(mesh.tree)

    elements = init_elements(leaf_cell_ids, mesh, equations_hyperbolic, dg.basis, RealT,
                             uEltype)

    interfaces = init_interfaces(leaf_cell_ids, mesh, elements)

    boundaries = init_boundaries(leaf_cell_ids, mesh, elements)

    viscous_container = init_viscous_container_1d(nvariables(equations_hyperbolic),
                                                  nnodes(elements), nelements(elements),
                                                  uEltype)

    cache = (; elements, interfaces, boundaries, viscous_container)

    return cache
end

# Needed to *not* flip the sign of the inverse Jacobian.
# This is because the parabolic fluxes are assumed to be of the form
#   `du/dt + df/dx = dg/dx + source(x,t)`,
# where f(u) is the inviscid flux and g(u) is the viscous flux.
function apply_jacobian_parabolic!(du, mesh::TreeMesh{1},
                                   equations::AbstractEquationsParabolic, dg::DG, cache)
    @unpack inverse_jacobian = cache.elements

    @threaded for element in eachelement(dg, cache)
        factor = inverse_jacobian[element]

        for i in eachnode(dg)
            for v in eachvariable(equations)
                du[v, i, element] *= factor
            end
        end
    end

    return nothing
end
end # @muladd
