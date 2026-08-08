using KernelAbstractions: @synchronize, @localmem, @uniform

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

    # Prolong solution to mortars
    @trixi_timeit_ext backend timer() "prolong2mortars" begin
        prolong2mortars!(backend, cache, u, mesh, equations,
                         dg.mortar, dg)
    end

    # Calculate mortar fluxes
    @trixi_timeit_ext backend timer() "mortar flux" begin
        calc_mortar_flux!(backend, cache.elements.surface_flux_values, mesh,
                          have_nonconservative_terms(equations), equations,
                          dg.mortar, dg.surface_integral, dg, cache)
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

# Number of surface nodes of a single boundary. Together with the number of
# boundaries this makes up the `ndrange` of the boundary kernels below, i.e.,
# every surface node is handled by its own GPU thread.
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

    # Get information on the adjacent element, compute the surface flux,
    # and store it
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

# For GPU backends mortars are not yet implemented
function prolong2mortars!(backend::Backend, cache, u, mesh, equations, mortar, dg)
    @assert isempty(eachmortar(dg, cache))
    return nothing
end

# For GPU backends mortars are not yet implemented
function calc_mortar_flux!(backend::Backend, surface_flux_values, mesh,
                           have_nonconservative_terms, equations, mortar,
                           surface_integral, dg, cache)
    @assert isempty(eachmortar(dg, cache))
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
