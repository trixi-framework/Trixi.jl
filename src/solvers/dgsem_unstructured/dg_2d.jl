# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::UnstructuredMesh2D, equations,
                      dg::DG, RealT, uEltype)
    elements = init_elements(mesh, equations, dg.basis, RealT, uEltype)

    interfaces = init_interfaces(mesh, elements)

    boundaries = init_boundaries(mesh, elements)

    cache = (; elements, interfaces, boundaries)

    # perform a check on the sufficient metric identities condition for free-stream preservation
    # and halt computation if it fails
    # For `Float64`, this gives 1.8189894035458565e-12
    # For `Float32`, this gives 1.1920929f-5
    atol = max(100 * eps(RealT), eps(RealT)^convert(RealT, 0.75f0))
    if !isapprox(max_discrete_metric_identities(dg, cache), 0, atol = atol)
        error("metric terms fail free-stream preservation check with maximum error $(max_discrete_metric_identities(dg, cache))")
    end

    # Add specialized parts of the cache required to compute the flux differencing volume integral
    cache = (; cache...,
             create_cache(mesh, equations, dg.volume_integral, dg, uEltype)...)

    return cache
end

function rhs!(du, u, t,
              mesh::UnstructuredMesh2D, equations,
              boundary_conditions, source_terms::Source,
              dg::DG, cache) where {Source}
    # Reset du
    @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

    # Calculate volume integral
    @trixi_timeit timer() "volume integral" begin
        calc_volume_integral!(du, u, mesh,
                              have_nonconservative_terms(equations), equations,
                              dg.volume_integral, dg, cache)
    end

    # Prolong solution to interfaces
    @trixi_timeit timer() "prolong2interfaces" begin
        prolong2interfaces!(cache, u, mesh, equations,
                            dg.surface_integral, dg)
    end

    # Calculate interface fluxes
    @trixi_timeit timer() "interface flux" begin
        calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                             have_nonconservative_terms(equations), equations,
                             dg.surface_integral, dg, cache)
    end

    # Prolong solution to boundaries
    @trixi_timeit timer() "prolong2boundaries" begin
        prolong2boundaries!(cache, u, mesh, equations,
                            dg.surface_integral, dg)
    end

    # Calculate boundary fluxes
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations,
                            dg.surface_integral, dg)
    end

    # Calculate surface integrals
    @trixi_timeit timer() "surface integral" begin
        calc_surface_integral!(du, u, mesh, equations,
                               dg.surface_integral, dg, cache)
    end

    # Apply Jacobian from mapping to reference element
    #  Note! this routine is reused from dgsem_structured/dg_2d.jl
    @trixi_timeit timer() "Jacobian" apply_jacobian!(du, mesh, equations, dg, cache)

    # Calculate source terms
    @trixi_timeit timer() "source terms" begin
        calc_sources!(du, u, t, source_terms, equations, dg, cache)
    end

    return nothing
end

# prolong the solution into the convenience array in the interior interface container
# We pass the `surface_integral` argument solely for dispatch
# Note! this routine is for quadrilateral elements with "right-handed" orientation
function prolong2interfaces!(cache, u,
                             mesh::UnstructuredMesh2D,
                             equations, surface_integral, dg::DG)
    @unpack interfaces = cache
    @unpack element_ids, element_side_ids = interfaces
    interfaces_u = interfaces.u

    @threaded for interface in eachinterface(dg, cache)
        primary_element = element_ids[1, interface]
        secondary_element = element_ids[2, interface]

        primary_side = element_side_ids[1, interface]
        secondary_side = element_side_ids[2, interface]

        if primary_side == 1
            for i in eachnode(dg), v in eachvariable(equations)
                interfaces_u[1, v, i, interface] = u[v, i, 1, primary_element]
            end
        elseif primary_side == 2
            for i in eachnode(dg), v in eachvariable(equations)
                interfaces_u[1, v, i, interface] = u[v, nnodes(dg), i, primary_element]
            end
        elseif primary_side == 3
            for i in eachnode(dg), v in eachvariable(equations)
                interfaces_u[1, v, i, interface] = u[v, i, nnodes(dg), primary_element]
            end
        else # primary_side == 4
            for i in eachnode(dg), v in eachvariable(equations)
                interfaces_u[1, v, i, interface] = u[v, 1, i, primary_element]
            end
        end

        if secondary_side == 1
            for i in eachnode(dg), v in eachvariable(equations)
                interfaces_u[2, v, i, interface] = u[v, i, 1, secondary_element]
            end
        elseif secondary_side == 2
            for i in eachnode(dg), v in eachvariable(equations)
                interfaces_u[2, v, i, interface] = u[v, nnodes(dg), i,
                                                     secondary_element]
            end
        elseif secondary_side == 3
            for i in eachnode(dg), v in eachvariable(equations)
                interfaces_u[2, v, i, interface] = u[v, i, nnodes(dg),
                                                     secondary_element]
            end
        else # secondary_side == 4
            for i in eachnode(dg), v in eachvariable(equations)
                interfaces_u[2, v, i, interface] = u[v, 1, i, secondary_element]
            end
        end
    end

    return nothing
end

# compute the numerical flux interface coupling between two elements on an unstructured
# quadrilateral mesh
function calc_interface_flux!(surface_flux_values,
                              mesh::UnstructuredMesh2D,
                              nonconservative_terms::False, equations,
                              surface_integral, dg::DG, cache)
    @unpack surface_flux = surface_integral
    @unpack u, start_index, index_increment, element_ids, element_side_ids = cache.interfaces
    @unpack normal_directions = cache.elements

    @threaded for interface in eachinterface(dg, cache)
        # Get neighboring elements
        primary_element = element_ids[1, interface]
        secondary_element = element_ids[2, interface]

        # Get the local side id on which to compute the flux
        primary_side = element_side_ids[1, interface]
        secondary_side = element_side_ids[2, interface]

        # initial index for the coordinate system on the secondary element
        secondary_index = start_index[interface]

        # loop through the primary element coordinate system and compute the interface coupling
        for primary_index in eachnode(dg)
            # pull the primary and secondary states from the boundary u values
            u_ll = get_one_sided_surface_node_vars(u, equations, dg, 1, primary_index,
                                                   interface)
            u_rr = get_one_sided_surface_node_vars(u, equations, dg, 2, secondary_index,
                                                   interface)

            # pull the outward pointing (normal) directional vector
            #   Note! this assumes a conforming approximation, more must be done in terms of the normals
            #         for hanging nodes and other non-conforming approximation spaces
            outward_direction = get_surface_normal(normal_directions, primary_index,
                                                   primary_side,
                                                   primary_element)

            # Call pointwise numerical flux with rotation. Direction is normalized inside this function
            flux = surface_flux(u_ll, u_rr, outward_direction, equations)

            # Copy flux back to primary/secondary element storage
            # Note the sign change for the normal flux in the secondary element!
            for v in eachvariable(equations)
                surface_flux_values[v, primary_index, primary_side, primary_element] = flux[v]
                surface_flux_values[v, secondary_index, secondary_side, secondary_element] = -flux[v]
            end

            # increment the index of the coordinate system in the secondary element
            secondary_index += index_increment[interface]
        end
    end

    return nothing
end

# compute the numerical flux interface with nonconservative terms coupling between two elements
# on an unstructured quadrilateral mesh
function calc_interface_flux!(surface_flux_values,
                              mesh::UnstructuredMesh2D,
                              nonconservative_terms::True, equations,
                              surface_integral, dg::DG, cache)
    surface_flux, nonconservative_flux = surface_integral.surface_flux
    @unpack u, start_index, index_increment, element_ids, element_side_ids = cache.interfaces
    @unpack normal_directions = cache.elements

    @threaded for interface in eachinterface(dg, cache)
        # Get the primary element index and local side index
        primary_element = element_ids[1, interface]
        primary_side = element_side_ids[1, interface]

        # Get neighboring element, local side index, and index increment on the
        # secondary element
        secondary_element = element_ids[2, interface]
        secondary_side = element_side_ids[2, interface]
        secondary_index_increment = index_increment[interface]

        secondary_index = start_index[interface]
        for primary_index in eachnode(dg)
            # pull the primary and secondary states from the boundary u values
            u_ll = get_one_sided_surface_node_vars(u, equations, dg, 1, primary_index,
                                                   interface)
            u_rr = get_one_sided_surface_node_vars(u, equations, dg, 2, secondary_index,
                                                   interface)

            # pull the outward pointing (normal) directional vector
            # Note! This assumes a conforming approximation, more must be done in terms
            # of the normals for hanging nodes and other non-conforming approximation spaces
            outward_direction = get_surface_normal(normal_directions, primary_index,
                                                   primary_side,
                                                   primary_element)

            # Calculate the conservative portion of the numerical flux
            # Call pointwise numerical flux with rotation. Direction is normalized
            # inside this function
            flux = surface_flux(u_ll, u_rr, outward_direction, equations)

            # Compute both nonconservative fluxes
            noncons_primary = nonconservative_flux(u_ll, u_rr, outward_direction,
                                                   equations)
            noncons_secondary = nonconservative_flux(u_rr, u_ll, outward_direction,
                                                     equations)

            # Copy flux to primary and secondary element storage
            # Note the sign change for the components in the secondary element!
            for v in eachvariable(equations)
                # Note the factor 0.5 necessary for the nonconservative fluxes based on
                # the interpretation of global SBP operators coupled discontinuously via
                # central fluxes/SATs
                surface_flux_values[v, primary_index, primary_side, primary_element] = (flux[v] +
                                                                                        0.5f0 *
                                                                                        noncons_primary[v])
                surface_flux_values[v, secondary_index, secondary_side, secondary_element] = -(flux[v] +
                                                                                               0.5f0 *
                                                                                               noncons_secondary[v])
            end

            # increment the index of the coordinate system in the secondary element
            secondary_index += secondary_index_increment
        end
    end

    return nothing
end

# move the approximate solution onto physical boundaries within a "right-handed" element
function prolong2boundaries!(cache, u,
                             mesh::UnstructuredMesh2D,
                             equations, surface_integral, dg::DG)
    @unpack boundaries = cache
    @unpack element_id, element_side_id = boundaries
    boundaries_u = boundaries.u

    @threaded for boundary in eachboundary(dg, cache)
        element = element_id[boundary]
        side = element_side_id[boundary]

        if side == 1
            for l in eachnode(dg), v in eachvariable(equations)
                boundaries_u[v, l, boundary] = u[v, l, 1, element]
            end
        elseif side == 2
            for l in eachnode(dg), v in eachvariable(equations)
                boundaries_u[v, l, boundary] = u[v, nnodes(dg), l, element]
            end
        elseif side == 3
            for l in eachnode(dg), v in eachvariable(equations)
                boundaries_u[v, l, boundary] = u[v, l, nnodes(dg), element]
            end
        else # side == 4
            for l in eachnode(dg), v in eachvariable(equations)
                boundaries_u[v, l, boundary] = u[v, 1, l, element]
            end
        end
    end

    return nothing
end

# TODO: Taal dimension agnostic
function calc_boundary_flux!(cache, t, boundary_condition::BoundaryConditionPeriodic,
                             mesh::Union{UnstructuredMesh2D, P4estMesh, T8codeMesh},
                             equations, surface_integral, dg::DG)
    @assert isempty(eachboundary(dg, cache))
end

# Function barrier for type stability
function calc_boundary_flux!(cache, t, boundary_conditions,
                             mesh::Union{UnstructuredMesh2D, P4estMesh, T8codeMesh},
                             equations, surface_integral, dg::DG)
    @unpack boundary_condition_types, boundary_indices = boundary_conditions

    calc_boundary_flux_by_type!(cache, t, boundary_condition_types, boundary_indices,
                                mesh, equations, surface_integral, dg)
    return nothing
end

# Iterate over tuples of boundary condition types and associated indices
# in a type-stable way using "lispy tuple programming".
function calc_boundary_flux_by_type!(cache, t, BCs::NTuple{N, Any},
                                     BC_indices::NTuple{N, Vector{Int}},
                                     mesh::Union{UnstructuredMesh2D, P4estMesh,
                                                 T8codeMesh},
                                     equations, surface_integral, dg::DG) where {N}
    # Extract the boundary condition type and index vector
    boundary_condition = first(BCs)
    boundary_condition_indices = first(BC_indices)
    # Extract the remaining types and indices to be processed later
    remaining_boundary_conditions = Base.tail(BCs)
    remaining_boundary_condition_indices = Base.tail(BC_indices)

    # process the first boundary condition type
    calc_boundary_flux!(cache, t, boundary_condition, boundary_condition_indices,
                        mesh, equations, surface_integral, dg)

    # recursively call this method with the unprocessed boundary types
    calc_boundary_flux_by_type!(cache, t, remaining_boundary_conditions,
                                remaining_boundary_condition_indices,
                                mesh, equations, surface_integral, dg)

    return nothing
end

# terminate the type-stable iteration over tuples
function calc_boundary_flux_by_type!(cache, t, BCs::Tuple{}, BC_indices::Tuple{},
                                     mesh::Union{UnstructuredMesh2D, P4estMesh,
                                                 T8codeMesh},
                                     equations, surface_integral, dg::DG)
    nothing
end

function calc_boundary_flux!(cache, t, boundary_condition::BC, boundary_indexing,
                             mesh::UnstructuredMesh2D, equations,
                             surface_integral, dg::DG) where {BC}
    @unpack surface_flux_values = cache.elements
    @unpack element_id, element_side_id = cache.boundaries

    @threaded for local_index in eachindex(boundary_indexing)
        # use the local index to get the global boundary index from the pre-sorted list
        boundary = boundary_indexing[local_index]

        # get the element and side IDs on the boundary element
        element = element_id[boundary]
        side = element_side_id[boundary]

        # calc boundary flux on the current boundary interface
        for node in eachnode(dg)
            calc_boundary_flux!(surface_flux_values, t, boundary_condition,
                                mesh, have_nonconservative_terms(equations),
                                equations, surface_integral, dg, cache,
                                node, side, element, boundary)
        end
    end
end

# inlined version of the boundary flux calculation along a physical interface where the
# boundary flux values are set according to a particular `boundary_condition` function
@inline function calc_boundary_flux!(surface_flux_values, t, boundary_condition,
                                     mesh::UnstructuredMesh2D,
                                     nonconservative_terms::False, equations,
                                     surface_integral, dg::DG, cache,
                                     node_index, side_index, element_index,
                                     boundary_index)
    @unpack normal_directions = cache.elements
    @unpack u, node_coordinates = cache.boundaries
    @unpack surface_flux = surface_integral

    # pull the inner solution state from the boundary u values on the boundary element
    u_inner = get_node_vars(u, equations, dg, node_index, boundary_index)

    # pull the outward pointing (normal) directional vector
    outward_direction = get_surface_normal(normal_directions, node_index, side_index,
                                           element_index)

    # get the external solution values from the prescribed external state
    x = get_node_coords(node_coordinates, equations, dg, node_index, boundary_index)

    # Call pointwise numerical flux function in the normal direction on the boundary
    flux = boundary_condition(u_inner, outward_direction, x, t, surface_flux, equations)

    for v in eachvariable(equations)
        surface_flux_values[v, node_index, side_index, element_index] = flux[v]
    end
end

# inlined version of the boundary flux and nonconseravtive terms calculation along a
# physical interface. The conservative portion of the boundary flux values
# are set according to a particular `boundary_condition` function
# Note, it is necessary to set and add in the nonconservative values because
# the upper left/lower right diagonal terms have been peeled off due to the use of
# `derivative_split` from `dg.basis` in [`flux_differencing_kernel!`](@ref)
@inline function calc_boundary_flux!(surface_flux_values, t, boundary_condition,
                                     mesh::UnstructuredMesh2D,
                                     nonconservative_terms::True, equations,
                                     surface_integral, dg::DG, cache,
                                     node_index, side_index, element_index,
                                     boundary_index)
    surface_flux, nonconservative_flux = surface_integral.surface_flux
    @unpack normal_directions = cache.elements
    @unpack u, node_coordinates = cache.boundaries

    # pull the inner solution state from the boundary u values on the boundary element
    u_inner = get_node_vars(u, equations, dg, node_index, boundary_index)

    # pull the outward pointing (normal) directional vector
    outward_direction = get_surface_normal(normal_directions, node_index, side_index,
                                           element_index)

    # get the external solution values from the prescribed external state
    x = get_node_coords(node_coordinates, equations, dg, node_index, boundary_index)

    # Call pointwise numerical flux function for the conservative part
    # in the normal direction on the boundary
    flux = boundary_condition(u_inner, outward_direction, x, t, surface_flux, equations)

    # Compute pointwise nonconservative numerical flux at the boundary.
    noncons_flux = boundary_condition(u_inner, outward_direction, x, t,
                                      nonconservative_flux, equations)

    for v in eachvariable(equations)
        # Note the factor 0.5 necessary for the nonconservative fluxes based on
        # the interpretation of global SBP operators coupled discontinuously via
        # central fluxes/SATs
        surface_flux_values[v, node_index, side_index, element_index] = flux[v] +
                                                                        0.5f0 *
                                                                        noncons_flux[v]
    end
end

# Note! The local side numbering for the unstructured quadrilateral element implementation differs
#       from the structured TreeMesh or StructuredMesh local side numbering:
#
#      TreeMesh/StructuredMesh sides   versus   UnstructuredMesh sides
#                  4                                  3
#          -----------------                  -----------------
#          |               |                  |               |
#          | ^ eta         |                  | ^ eta         |
#        1 | |             | 2              4 | |             | 2
#          | |             |                  | |             |
#          | ---> xi       |                  | ---> xi       |
#          -----------------                  -----------------
#                  3                                  1
# Therefore, we require a different surface integral routine here despite their similar structure.
function calc_surface_integral!(du, u, mesh::UnstructuredMesh2D,
                                equations, surface_integral, dg::DGSEM, cache)
    @unpack boundary_interpolation = dg.basis
    @unpack surface_flux_values = cache.elements

    @threaded for element in eachelement(dg, cache)
        for l in eachnode(dg), v in eachvariable(equations)
            # surface contribution along local sides 2 and 4 (fixed x and y varies)
            du[v, 1, l, element] += (surface_flux_values[v, l, 4, element]
                                     *
                                     boundary_interpolation[1, 1])
            du[v, nnodes(dg), l, element] += (surface_flux_values[v, l, 2, element]
                                              *
                                              boundary_interpolation[nnodes(dg), 2])
            # surface contribution along local sides 1 and 3 (fixed y and x varies)
            du[v, l, 1, element] += (surface_flux_values[v, l, 1, element]
                                     *
                                     boundary_interpolation[1, 1])
            du[v, l, nnodes(dg), element] += (surface_flux_values[v, l, 3, element]
                                              *
                                              boundary_interpolation[nnodes(dg), 2])
        end
    end

    return nothing
end

# This routine computes the maximum value of the discrete metric identities necessary to ensure
# that the approxmiation will be free-stream preserving (i.e. a constant solution remains constant)
# on a curvilinear mesh.
#   Note! Independent of the equation system and is only a check on the discrete mapping terms.
#         Can be used for a metric identities check on StructuredMesh{2} or UnstructuredMesh2D
function max_discrete_metric_identities(dg::DGSEM, cache)
    @unpack derivative_matrix = dg.basis
    @unpack contravariant_vectors = cache.elements

    ndims_ = size(contravariant_vectors, 1)

    metric_id_dx = zeros(eltype(contravariant_vectors), nnodes(dg), nnodes(dg))
    metric_id_dy = zeros(eltype(contravariant_vectors), nnodes(dg), nnodes(dg))

    max_metric_ids = zero(eltype(contravariant_vectors))

    for i in 1:ndims_, element in eachelement(dg, cache)
        # compute D*Ja_1^i + Ja_2^i*D^T
        @views mul!(metric_id_dx, derivative_matrix,
                    contravariant_vectors[i, 1, :, :, element])
        @views mul!(metric_id_dy, contravariant_vectors[i, 2, :, :, element],
                    derivative_matrix')
        local_max_metric_ids = maximum(abs.(metric_id_dx + metric_id_dy))

        max_metric_ids = max(max_metric_ids, local_max_metric_ids)
    end

    return max_metric_ids
end
end # @muladd
