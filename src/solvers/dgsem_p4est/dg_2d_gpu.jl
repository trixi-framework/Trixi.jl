# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function rhs_hyperbolic!(backend::Backend,
                         du, u, t,
                         mesh::Union{P4estMesh{2}, P4estMeshView{2}, T8codeMesh{2},
                                     P4estMesh{3}, T8codeMesh{3}},
                         equations,
                         boundary_conditions, source_terms::Source,
                         dg::DG, cache) where {Source}

    # Reset du
    @trixi_timeit_ext backend timer() "reset ∂u/∂t" begin
        set_zero!(du, dg, cache)
    end

    # Calculate volume integral
    @trixi_timeit_ext backend timer() "volume integral" begin
        calc_volume_integral!(backend, du, u, mesh,
                              have_nonconservative_terms(equations), equations,
                              dg.volume_integral, dg, cache)
    end

    # Calculate interface fluxes
    @trixi_timeit_ext backend timer() "prolong2interfaces + flux" begin
        prolong2interfaces_and_calc_interface_flux!(backend,
                                                    cache.elements.surface_flux_values,
                                                    u, mesh,
                                                    have_nonconservative_terms(equations),
                                                    equations,
                                                    dg.surface_integral, dg, cache)
    end

    # Prolong solution to boundaries
    @trixi_timeit_ext backend timer() "prolong2boundaries" begin
        prolong2boundaries!(backend, cache, u, mesh, equations, dg)
    end

    # Calculate boundary fluxes
    @trixi_timeit_ext backend timer() "boundary flux" begin
        calc_boundary_flux!(backend, cache, t, boundary_conditions, mesh, equations,
                            dg.surface_integral, dg)
    end

    # Prolong solution to mortars and calculate flux
    @trixi_timeit_ext backend timer() "prolong2mortars + flux" begin
        prolong2mortars_and_calc_mortar_flux!(backend,
                                              cache.elements.surface_flux_values,
                                              u, mesh,
                                              have_nonconservative_terms(equations),
                                              equations, dg.mortar, dg, cache)
    end

    # Calculate surface integrals, apply Jacobian from mapping to reference element
    # and calculate source terms
    @trixi_timeit_ext backend timer() "surface, Jacobian + source terms" begin
        calc_surface_integral_and_apply_jacobian_and_calc_sources!(backend, du, u, t,
                                                                   source_terms, mesh,
                                                                   equations,
                                                                   dg.surface_integral,
                                                                   dg, cache)
    end

    return nothing
end

function prolong2interfaces_and_calc_interface_flux!(backend::Backend,
                                                     surface_flux_values, u,
                                                     mesh::Union{P4estMesh{2},
                                                                 P4estMeshView{2},
                                                                 T8codeMesh{2}},
                                                     have_nonconservative_terms,
                                                     equations,
                                                     surface_integral,
                                                     dg::DGSEM{<:LobattoLegendreBasis},
                                                     cache)
    @unpack neighbor_ids, node_indices = cache.interfaces
    @unpack contravariant_vectors = cache.elements
    ninterfaces(cache.interfaces) == 0 && return nothing
    index_range = eachnode(dg)
    kernel! = prolong2interfaces_and_calc_interface_flux_KAkernel!(backend)
    kernel!(surface_flux_values, u, typeof(mesh), have_nonconservative_terms, equations,
            surface_integral, dg, neighbor_ids, node_indices, contravariant_vectors,
            index_range,
            ndrange = (nnodes(dg), ninterfaces(cache.interfaces)))

    return nothing
end

@kernel function prolong2interfaces_and_calc_interface_flux_KAkernel!(surface_flux_values,
                                                                      u,
                                                                      MeshT::Type{<:Union{P4estMesh{2},
                                                                                          P4estMeshView{2},
                                                                                          T8codeMesh{2}}},
                                                                      have_nonconservative_terms,
                                                                      equations,
                                                                      surface_integral,
                                                                      dg,
                                                                      neighbor_ids,
                                                                      node_indices,
                                                                      contravariant_vectors,
                                                                      index_range)
    i, interface = @index(Global, NTuple)
    prolong2interfaces_and_calc_interface_flux_per_node!(surface_flux_values, u, MeshT,
                                                         have_nonconservative_terms,
                                                         equations,
                                                         surface_integral, dg,
                                                         neighbor_ids,
                                                         node_indices,
                                                         contravariant_vectors,
                                                         index_range, i,
                                                         interface)
end

@inline function delayed_index_2d(start, step, i)
    return start + (i - 1) * step
end

@inline function get_interface_values(u, equations, dg, neighbor_ids,
                                      node_indices, contravariant_vectors,
                                      index_range, i, interface)
    primary_element = neighbor_ids[1, interface]
    primary_indices = node_indices[1, interface]
    primary_direction = indices2direction(primary_indices)

    i_primary_start, i_primary_step = index_to_start_step_2d(primary_indices[1],
                                                             index_range)
    j_primary_start, j_primary_step = index_to_start_step_2d(primary_indices[2],
                                                             index_range)

    i_primary = delayed_index_2d(i_primary_start, i_primary_step, i)
    j_primary = delayed_index_2d(j_primary_start, j_primary_step, i)

    secondary_element = neighbor_ids[2, interface]
    secondary_indices = node_indices[2, interface]
    secondary_direction = indices2direction(secondary_indices)

    i_secondary_start, i_secondary_step = index_to_start_step_2d(secondary_indices[1],
                                                                 index_range)
    j_secondary_start, j_secondary_step = index_to_start_step_2d(secondary_indices[2],
                                                                 index_range)

    i_secondary_node = delayed_index_2d(i_secondary_start, i_secondary_step, i)
    j_secondary_node = delayed_index_2d(j_secondary_start, j_secondary_step, i)

    if i_secondary_step == 0
        i_secondary = j_secondary_node
    else
        i_secondary = i_secondary_node
    end

    u_ll = get_node_vars(u, equations, dg, i_primary, j_primary, primary_element)
    u_rr = get_node_vars(u, equations, dg, i_secondary_node, j_secondary_node,
                         secondary_element)

    normal_direction = get_normal_direction(primary_direction, contravariant_vectors,
                                            i_primary, j_primary, primary_element)

    return (u_ll, u_rr, normal_direction, primary_direction, primary_element,
            i_secondary, secondary_direction, secondary_element)
end

@inline function prolong2interfaces_and_calc_interface_flux_per_node!(surface_flux_values,
                                                                      u,
                                                                      MeshT::Type{<:Union{P4estMesh{2},
                                                                                          P4estMeshView{2},
                                                                                          T8codeMesh{2}}},
                                                                      have_nonconservative_terms::False,
                                                                      equations,
                                                                      surface_integral,
                                                                      dg,
                                                                      neighbor_ids,
                                                                      node_indices,
                                                                      contravariant_vectors,
                                                                      index_range,
                                                                      i, interface)
    @unpack surface_flux = surface_integral

    u_ll, u_rr, normal_direction, primary_direction, primary_element,
    i_secondary, secondary_direction, secondary_element = get_interface_values(u,
                                                                               equations,
                                                                               dg,
                                                                               neighbor_ids,
                                                                               node_indices,
                                                                               contravariant_vectors,
                                                                               index_range,
                                                                               i,
                                                                               interface)

    flux_ = surface_flux(u_ll, u_rr, normal_direction, equations)
    for v in eachvariable(equations)
        surface_flux_values[v, i, primary_direction, primary_element] = flux_[v]
        surface_flux_values[v, i_secondary, secondary_direction,
        secondary_element] = -flux_[v]
    end
    return nothing
end

@inline function prolong2interfaces_and_calc_interface_flux_per_node!(surface_flux_values,
                                                                      u,
                                                                      MeshT::Type{<:Union{P4estMesh{2},
                                                                                          P4estMeshView{2},
                                                                                          T8codeMesh{2}}},
                                                                      have_nonconservative_terms::True,
                                                                      equations,
                                                                      surface_integral,
                                                                      dg,
                                                                      neighbor_ids,
                                                                      node_indices,
                                                                      contravariant_vectors,
                                                                      index_range,
                                                                      i, interface)
    prolong2interfaces_and_calc_interface_flux_per_node!(surface_flux_values, u, MeshT,
                                                         have_nonconservative_terms,
                                                         combine_conservative_and_nonconservative_fluxes(surface_integral.surface_flux,
                                                                                                         equations),
                                                         equations, surface_integral,
                                                         dg,
                                                         neighbor_ids,
                                                         node_indices,
                                                         contravariant_vectors,
                                                         index_range, i, interface)
    return nothing
end

@inline function prolong2interfaces_and_calc_interface_flux_per_node!(surface_flux_values,
                                                                      u,
                                                                      MeshT::Type{<:Union{P4estMesh{2},
                                                                                          P4estMeshView{2},
                                                                                          T8codeMesh{2}}},
                                                                      have_nonconservative_terms::True,
                                                                      combine_conservative_and_nonconservative_fluxes::True,
                                                                      equations,
                                                                      surface_integral,
                                                                      dg,
                                                                      neighbor_ids,
                                                                      node_indices,
                                                                      contravariant_vectors,
                                                                      index_range,
                                                                      i, interface)
    @unpack surface_flux = surface_integral

    u_ll, u_rr, normal_direction, primary_direction, primary_element,
    i_secondary, secondary_direction, secondary_element = get_interface_values(u,
                                                                               equations,
                                                                               dg,
                                                                               neighbor_ids,
                                                                               node_indices,
                                                                               contravariant_vectors,
                                                                               index_range,
                                                                               i,
                                                                               interface)

    flux_left, flux_right = surface_flux(u_ll, u_rr, normal_direction, equations)
    for v in eachvariable(equations)
        surface_flux_values[v, i, primary_direction, primary_element] = flux_left[v]
        surface_flux_values[v, i_secondary, secondary_direction,
        secondary_element] = -flux_right[v]
    end
    return nothing
end

@kernel function prolong2boundaries_kernel!(u,
                                            MeshT::Type{<:Union{P4estMesh{2},
                                                                P4estMeshView{2},
                                                                T8codeMesh{2}}},
                                            equations, dg, index_range,
                                            u_boundaries, neighbor_ids, node_indices)
    i, boundary = @index(Global, NTuple)
    prolong2boundaries_per_node!(u, MeshT, equations, dg, index_range, u_boundaries,
                                 neighbor_ids, node_indices, i, boundary)
end

@inline function boundary_node_ndrange(mesh::Union{P4estMesh{2}, P4estMeshView{2},
                                                   T8codeMesh{2}}, dg)
    return (nnodes(dg),)
end

@inline function prolong2boundaries_per_node!(u,
                                              MeshT::Type{<:Union{P4estMesh{2},
                                                                  P4estMeshView{2},
                                                                  T8codeMesh{2}}},
                                              equations, dg::DG, index_range,
                                              u_boundaries,
                                              neighbor_ids, node_indices, i, boundary)
    # Copy solution data from the element using "delayed indexing" with
    # a start value and a step size to get the correct face and orientation.
    element = neighbor_ids[boundary]
    node_index = node_indices[boundary]

    i_node_start, i_node_step = index_to_start_step_2d(node_index[1], index_range)
    j_node_start, j_node_step = index_to_start_step_2d(node_index[2], index_range)

    i_node = delayed_index_2d(i_node_start, i_node_step, i)
    j_node = delayed_index_2d(j_node_start, j_node_step, i)

    u_node = get_node_vars(u, equations, dg, i_node, j_node, element)
    set_node_vars!(u_boundaries, u_node, equations, dg, i, boundary)

    return nothing
end

@kernel function calc_boundary_flux_kernel!(u,
                                            surface_flux_values,
                                            boundary_condition_indices,
                                            neighbor_ids,
                                            node_indices_arr,
                                            t,
                                            boundary_condition,
                                            index_range,
                                            MeshT::Type{<:Union{P4estMesh{2},
                                                                P4estMeshView{2},
                                                                T8codeMesh{2}}},
                                            equations,
                                            surface_integral,
                                            dg,
                                            cache, node_coordinates,
                                            contravariant_vectors)
    i, local_index = @index(Global, NTuple)

    if local_index <= length(boundary_condition_indices)
        boundary = boundary_condition_indices[local_index]

        calc_boundary_flux_per_node!(u,
                                     surface_flux_values, t, boundary_condition,
                                     MeshT, equations, surface_integral, dg, cache,
                                     boundary, neighbor_ids, node_indices_arr,
                                     index_range, node_coordinates,
                                     contravariant_vectors, i)
    end
end

@inline function calc_boundary_flux_per_node!(u,
                                              surface_flux_values, t,
                                              boundary_condition,
                                              MeshT::Type{<:Union{P4estMesh{2},
                                                                  P4estMeshView{2},
                                                                  T8codeMesh{2}}},
                                              equations, surface_integral, dg,
                                              cache,
                                              boundary, neighbor_ids,
                                              node_indices_arr,
                                              index_range, node_coordinates,
                                              contravariant_vectors, i)

    # Get information on the adjacent element, compute the surface fluxes,
    # and store them
    element = neighbor_ids[boundary]
    node_indices = node_indices_arr[boundary]
    direction = indices2direction(node_indices)

    i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
    j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

    i_node = delayed_index_2d(i_node_start, i_node_step, i)
    j_node = delayed_index_2d(j_node_start, j_node_step, i)

    calc_boundary_flux!(u, surface_flux_values, t, boundary_condition, MeshT,
                        have_nonconservative_terms(equations), equations,
                        surface_integral, dg, cache, i_node, j_node,
                        i, direction, element, boundary, node_coordinates,
                        contravariant_vectors)
end

# inlined version of the boundary flux calculation along a physical interface
@inline function calc_boundary_flux!(u, surface_flux_values, t, boundary_condition,
                                     MeshT::Type{<:Union{P4estMesh{2},
                                                         P4estMeshView{2},
                                                         T8codeMesh{2}}},
                                     have_nonconservative_terms::False, equations,
                                     surface_integral, dg, cache,
                                     i_index, j_index, node_index,
                                     direction_index, element_index,
                                     boundary_index, node_coordinates,
                                     contravariant_vectors)
    @unpack surface_flux = surface_integral

    # Extract solution data from boundary container
    u_inner = get_node_vars(u, equations, dg, node_index, boundary_index)

    # Outward-pointing normal direction (not normalized)
    normal_direction = get_normal_direction(direction_index, contravariant_vectors,
                                            i_index, j_index, element_index)

    # Coordinates at boundary node
    x = get_node_coords(node_coordinates, equations, dg,
                        i_index, j_index, element_index)

    flux_ = boundary_condition(u_inner, normal_direction, x, t, surface_flux, equations)

    # Copy flux to element storage in the correct orientation
    for v in eachvariable(equations)
        surface_flux_values[v, node_index, direction_index, element_index] = flux_[v]
    end
end

@inline function calc_boundary_flux!(u, surface_flux_values, t, boundary_condition,
                                     MeshT::Type{<:Union{P4estMesh{2},
                                                         P4estMeshView{2},
                                                         T8codeMesh{2}}},
                                     have_nonconservative_terms::True, equations,
                                     surface_integral, dg, cache,
                                     i_index, j_index, node_index,
                                     direction_index, element_index,
                                     boundary_index, node_coordinates,
                                     contravariant_vectors)
    calc_boundary_flux!(u, surface_flux_values, t, boundary_condition, MeshT,
                        have_nonconservative_terms,
                        combine_conservative_and_nonconservative_fluxes(surface_integral.surface_flux,
                                                                        equations),
                        equations,
                        surface_integral, dg, cache,
                        i_index, j_index, node_index,
                        direction_index, element_index, boundary_index,
                        node_coordinates, contravariant_vectors)
    return nothing
end

@inline function calc_boundary_flux!(u, surface_flux_values, t, boundary_condition,
                                     MeshT::Type{<:Union{P4estMesh{2},
                                                         P4estMeshView{2},
                                                         T8codeMesh{2}}},
                                     have_nonconservative_terms::True,
                                     combine_conservative_and_nonconservative_fluxes::True,
                                     equations,
                                     surface_integral, dg::DG, cache,
                                     i_index, j_index, node_index,
                                     direction_index, element_index,
                                     boundary_index, node_coordinates,
                                     contravariant_vectors)
    @unpack surface_flux = surface_integral

    # Extract solution data from boundary container
    u_inner = get_node_vars(u, equations, dg, node_index, boundary_index)

    # Outward-pointing normal direction (not normalized)
    normal_direction = get_normal_direction(direction_index, contravariant_vectors,
                                            i_index, j_index, element_index)

    # Coordinates at boundary node
    x = get_node_coords(node_coordinates, equations, dg,
                        i_index, j_index, element_index)

    # Call pointwise numerical flux functions for the conservative and nonconservative part
    # in the normal direction on the boundary
    flux = boundary_condition(u_inner, normal_direction, x, t,
                              surface_flux, equations)

    # Copy flux to element storage in the correct orientation
    for v in eachvariable(equations)
        surface_flux_values[v, node_index,
        direction_index, element_index] = flux[v]
    end

    return nothing
end

function prolong2mortars_and_calc_mortar_flux!(backend::Backend, surface_flux_values, u,
                                               mesh::Union{P4estMesh{2}, T8codeMesh{2}},
                                               have_nonconservative_terms, equations,
                                               mortar_l2::LobattoLegendreMortarL2,
                                               dg::DGSEM{<:LobattoLegendreBasis}, cache)
    nmortars(dg, cache) == 0 && return nothing

    @unpack neighbor_ids, node_indices = cache.mortars
    @unpack contravariant_vectors = cache.elements
    @unpack surface_flux = dg.surface_integral

    index_range = eachnode(dg)
    nvars = nvariables(equations)

    kernel! = prolong2mortars_and_calc_mortar_flux_KAkernel!(backend)
    kernel!(cache.mortars.u, surface_flux_values, u, typeof(mesh),
            have_nonconservative_terms, equations,
            surface_flux, typeof(dg),
            neighbor_ids, node_indices, contravariant_vectors,
            mortar_l2.forward_lower, mortar_l2.forward_upper,
            mortar_l2.reverse_lower, mortar_l2.reverse_upper,
            index_range, Val(nvars),
            ndrange = (nnodes(dg), nmortars(dg, cache)))

    return nothing
end

@kernel function prolong2mortars_and_calc_mortar_flux_KAkernel!(mortars_u,
                                                                surface_flux_values, u,
                                                                MeshT::Type{<:Union{P4estMesh{2},
                                                                                    T8codeMesh{2}}},
                                                                have_nonconservative_terms,
                                                                equations,
                                                                surface_flux, SolverT,
                                                                neighbor_ids,
                                                                node_indices,
                                                                contravariant_vectors,
                                                                forward_lower,
                                                                forward_upper,
                                                                reverse_lower,
                                                                reverse_upper,
                                                                index_range,
                                                                val_vars::Val{nvars}) where {nvars}
    node, mortar = @index(Global, NTuple)

    prolong2mortars_per_node!(mortars_u, u, neighbor_ids, node_indices, forward_lower,
                              forward_upper, node, mortar, index_range, val_vars)

    calc_mortar_flux_small_per_node!(surface_flux_values, MeshT,
                                     have_nonconservative_terms, equations,
                                     surface_flux, SolverT, neighbor_ids, node_indices,
                                     contravariant_vectors, mortars_u,
                                     index_range, mortar, node, val_vars)

    calc_mortar_flux_large_per_node!(surface_flux_values, MeshT,
                                     have_nonconservative_terms, equations,
                                     surface_flux, SolverT, neighbor_ids, node_indices,
                                     contravariant_vectors, mortars_u,
                                     reverse_lower, reverse_upper,
                                     index_range, mortar, node, val_vars)
end

@inline function prolong2mortars_per_node!(mortars_u,
                                           u, neighbor_ids, node_indices,
                                           forward_lower, forward_upper,
                                           node, mortar, index_range,
                                           ::Val{nvars}) where {nvars}

    # Copy solution data from the small elements using "delayed indexing" with
    # a start value and a step size to get the correct face and orientation.
    small_indices = node_indices[1, mortar]

    i_small_start, i_small_step = index_to_start_step_2d(small_indices[1], index_range)
    j_small_start, j_small_step = index_to_start_step_2d(small_indices[2], index_range)

    i_small = i_small_start + (node - 1) * i_small_step
    j_small = j_small_start + (node - 1) * j_small_step

    for position in 1:2
        element = neighbor_ids[position, mortar]
        for v in 1:nvars
            mortars_u[1, v, position, node, mortar] = u[v, i_small, j_small, element]
        end
    end

    # Interpolate large element face data to small face locations
    # In contrast to the CPU version, we do not buffer the data
    large_indices = node_indices[2, mortar]
    element = neighbor_ids[3, mortar]

    i_large_start, i_large_step = index_to_start_step_2d(large_indices[1], index_range)
    j_large_start, j_large_step = index_to_start_step_2d(large_indices[2], index_range)

    # This is what the two multiply_dimensionwise! calls do in the CPU version
    for v in 1:nvars
        res_lower = zero(eltype(u))
        res_upper = zero(eltype(u))
        for ii in index_range
            i_large = i_large_start + (ii - 1) * i_large_step
            j_large = j_large_start + (ii - 1) * j_large_step
            res_lower += forward_lower[node, ii] * u[v, i_large, j_large, element]
            res_lower += forward_upper[node, ii] * u[v, i_large, j_large, element]
        end
        mortars_u[2, v, 1, node, mortar] = res_lower
        mortars_u[2, v, 2, node, mortar] = res_upper
    end
    return nothing
end

@inline function calc_mortar_flux_small_per_node!(surface_flux_values,
                                                  MeshT::Type{<:Union{P4estMesh{2},
                                                                      T8codeMesh{2}}},
                                                  have_nonconservative_terms, equations,
                                                  surface_flux, SolverT,
                                                  neighbor_ids, node_indices,
                                                  contravariant_vectors, mortars_u,
                                                  index_range, mortar, node,
                                                  ::Val{nvars}) where {nvars}

    # Get index information on the small elements
    small_indices = node_indices[1, mortar]
    small_direction = indices2direction(small_indices)

    i_small_start, i_small_step = index_to_start_step_2d(small_indices[1], index_range)
    j_small_start, j_small_step = index_to_start_step_2d(small_indices[2], index_range)

    i_small = i_small_start + (node - 1) * i_small_step
    j_small = j_small_start + (node - 1) * j_small_step

    for position in 1:2
        element = neighbor_ids[position, mortar]
        # Get the normal direction on the small element.
        # Note, contravariant vectors at interfaces in negative coordinate direction
        # are pointing inwards. This is handled by `get_normal_direction`.
        normal_direction = get_normal_direction(small_direction,
                                                contravariant_vectors,
                                                i_small, j_small, element)

        fstar_primary, _ = calc_mortar_flux!(MeshT,
                                             have_nonconservative_terms,
                                             combine_conservative_and_nonconservative_fluxes(surface_flux,
                                                                                             equations),
                                             equations,
                                             surface_flux, SolverT, mortars_u,
                                             mortar, position, normal_direction, node)

        # Write to small sides
        for v in 1:nvars
            surface_flux_values[v, node,
            small_direction, element] = fstar_primary[v]
        end
    end
end

@inline function calc_mortar_flux_large_per_node!(surface_flux_values,
                                                  MeshT::Type{<:Union{P4estMesh{2},
                                                                      T8codeMesh{2}}},
                                                  have_nonconservative_terms, equations,
                                                  surface_flux, SolverT,
                                                  neighbor_ids, node_indices,
                                                  contravariant_vectors, mortars_u,
                                                  reverse_lower, reverse_upper,
                                                  index_range, mortar, node,
                                                  ::Val{nvars}) where {nvars}
    large_element = neighbor_ids[3, mortar]
    large_indices = node_indices[2, mortar]
    large_direction = indices2direction(large_indices)

    # Note that the index of the small sides will always run forward but
    # the index of the large side might need to run backwards for flipped sides.
    large_node_index = :i_backward in large_indices ? last(index_range) + 1 - node :
                       node

    # Get index information on the small elements
    small_element_lower = neighbor_ids[1, mortar]
    small_element_upper = neighbor_ids[2, mortar]
    small_indices = node_indices[1, mortar]
    small_direction = indices2direction(small_indices)

    i_small_start, i_small_step = index_to_start_step_2d(small_indices[1], index_range)
    j_small_start, j_small_step = index_to_start_step_2d(small_indices[2], index_range)

    for v in 1:nvars
        surface_flux_values[v, large_node_index, large_direction, large_element] = 0
    end

    i_small = i_small_start
    j_small = j_small_start

    # TODO node dimes reverse upper lower?
    for ii in index_range
        # Get the normal direction on the small element.
        # Note, contravariant vectors at interfaces in negative coordinate direction
        # are pointing inwards. This is handled by `get_normal_direction`.
        normal_direction_lower = get_normal_direction(small_direction,
                                                      contravariant_vectors,
                                                      i_small, j_small,
                                                      small_element_lower)

        normal_direction_upper = get_normal_direction(small_direction,
                                                      contravariant_vectors,
                                                      i_small, j_small,
                                                      small_element_upper)

        _, fstar_secondary_lower = calc_mortar_flux!(MeshT,
                                                     have_nonconservative_terms,
                                                     combine_conservative_and_nonconservative_fluxes(surface_flux,
                                                                                                     equations),
                                                     equations,
                                                     surface_flux,
                                                     SolverT, mortars_u,
                                                     mortar, 1,
                                                     normal_direction_lower,
                                                     node)
        _, fstar_secondary_upper = calc_mortar_flux!(MeshT,
                                                     have_nonconservative_terms,
                                                     combine_conservative_and_nonconservative_fluxes(surface_flux,
                                                                                                     equations),
                                                     equations,
                                                     surface_flux,
                                                     SolverT, mortars_u,
                                                     mortar, 2,
                                                     normal_direction_upper,
                                                     node)
        # The flux is calculated in the outward direction of the small elements,
        # so the sign must be switched to get the flux in outward direction
        # of the large element.
        # The contravariant vectors of the large element (and therefore the normal
        # vectors of the large element as well) are twice as large as the
        # contravariant vectors of the small elements. Therefore, the flux needs
        # to be scaled by a factor of 2 to obtain the flux of the large element.
        for v in 1:nvars
            surface_flux_values[v, large_node_index,
            large_direction,
            large_element] += -2 * (reverse_upper[node, ii] * fstar_secondary_upper[v] +
                               reverse_lower[node, ii] * fstar_secondary_lower[v])
        end

        i_small += i_small_step
        j_small += j_small_step
    end
end

# Mortar flux computation on small elements for conservation laws
# In contrast to the CPU version fluxes are not stored in buffers
@inline function calc_mortar_flux!(MeshT::Type{<:Union{P4estMesh{2}, T8codeMesh{2}}},
                                   have_nonconservative_terms::False,
                                   combine_conservative_and_nonconservative_fluxes,
                                   equations,
                                   surface_flux, SolverT::Type{<:DG}, mortars_u,
                                   mortar_index, position_index, normal_direction,
                                   node_index)
    u_ll, u_rr = get_surface_node_vars(mortars_u, equations, SolverT, position_index,
                                       node_index, mortar_index)

    flux = surface_flux(u_ll, u_rr, normal_direction, equations)

    return flux, flux
end

# Mortar flux computation on small elements for equations with conservative and
# nonconservative terms
# In contrast to the CPU version fluxes are not stored in buffers

# TODO GPU: untested!
@inline function calc_mortar_flux!(MeshT::Type{<:Union{P4estMesh{2}, T8codeMesh{2}}},
                                   have_nonconservative_terms::True,
                                   combine_conservative_and_nonconservative_fluxes::True,
                                   equations,
                                   surface_flux, SolverT::Type{<:DG}, mortars_u,
                                   mortar_index, position_index, normal_direction,
                                   node_index)
    u_ll, u_rr = get_surface_node_vars(mortars_u, equations, SolverT, position_index,
                                       node_index, mortar_index)

    # Compute combined fluxes
    flux_left, flux_right = surface_flux(u_ll, u_rr, normal_direction, equations)
    return flux_left, -flux_right
end

function calc_surface_integral_and_apply_jacobian_and_calc_sources!(backend::Backend,
                                                                    du, u, t,
                                                                    source_terms,
                                                                    mesh::Union{P4estMesh{2},
                                                                                T8codeMesh{2},
                                                                                P4estMeshView{2}},
                                                                    equations,
                                                                    surface_integral::SurfaceIntegralWeakForm,
                                                                    dg::DGSEM{<:LobattoLegendreBasis},
                                                                    cache)
    nelements(dg, cache) == 0 && return nothing
    @unpack inverse_weights = dg.basis
    @unpack surface_flux_values, inverse_jacobian, node_coordinates = cache.elements
    kernel_cache = kernel_filter_cache(cache)
    NNODES = nnodes(dg)
    kernel! = calc_surface_integral_and_apply_jacobian_and_calc_sources_KAkernel!(backend)
    kernel!(du, u, t, source_terms, node_coordinates, typeof(mesh), equations,
            inverse_weights[1], Val(NNODES), surface_flux_values, dg, inverse_jacobian,
            kernel_cache,
            ndrange = (NNODES, NNODES, nelements(dg, cache)))

    return nothing
end

@kernel function calc_surface_integral_and_apply_jacobian_and_calc_sources_KAkernel!(du,
                                                                                     u,
                                                                                     t,
                                                                                     source_terms,
                                                                                     node_coordinates,
                                                                                     MeshT::Type{<:Union{P4estMesh{2},
                                                                                                         P4estMeshView{2},
                                                                                                         T8codeMesh{2}}},
                                                                                     equations::AbstractEquations{2},
                                                                                     factor,
                                                                                     ::Val{NNODES},
                                                                                     surface_flux_values,
                                                                                     dg::DGSEM,
                                                                                     inverse_jacobian,
                                                                                     cache) where {NNODES}
    i, j, element = @index(Global, NTuple)
    # Note that all fluxes have been computed with outward-pointing normal vectors.
    # This computes the **negative** surface integral contribution,
    # i.e., M^{-1} * boundary_interpolation^T (which is for Gauss-Lobatto DGSEM just M^{-1} * B)
    # and the missing "-" is taken care of by the Jacobian factor below.
    #
    # We also use explicit assignments instead of `+=` to let `@muladd` turn these
    # into FMAs (see comment at the top of the file).
    #
    # factor = inverse_weights[1]
    # For LGL basis: Identical to weighted boundary interpolation at x = ±1
    x_node_interface = (i == 1) | (i == NNODES)
    y_node_interface = (j == 1) | (j == NNODES)
    x_face = ifelse(i == 1, 1, 2)
    y_face = ifelse(j == 1, 3, 4)
    _zero = zero(eltype(du))
    surface_node = SVector(ntuple(@inline(v->ifelse(x_node_interface,
                                                    surface_flux_values[v, j, x_face,
                                                                        element],
                                                    _zero) +
                                             ifelse(y_node_interface,
                                                    surface_flux_values[v, i, y_face,
                                                                        element], _zero)),
                                  Val(nvariables(equations))))
    source_node = calc_source_terms_node(u, t, source_terms, node_coordinates,
                                         equations, dg, i, j, element)
    jacobian_factor = inverse_jacobian[i, j, element]
    du_local = get_node_vars(du, equations, dg, i, j, element) + factor * surface_node
    du_node = source_node - jacobian_factor * du_local
    set_node_vars!(du, du_node, equations, dg, i, j, element)
end

@inline function calc_source_terms_node(u, t, source_terms, node_coordinates,
                                        equations, dg::DG, indices...)
    u_local = get_node_vars(u, equations, dg, indices...)
    x_local = get_node_coords(node_coordinates, equations, dg, indices...)

    return source_terms(u_local, x_local, t, equations)
end

@inline function calc_source_terms_node(u, t, source_terms::Nothing, node_coordinates,
                                        equations, dg::DG, indices...)
    return zero(SVector{nvariables(equations), eltype(u)})
end
end #muladd
