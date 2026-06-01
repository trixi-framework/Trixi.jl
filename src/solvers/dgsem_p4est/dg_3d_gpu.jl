# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function calc_volume_integral!(backend::Backend, du, u,
                                       mesh::P4estMesh{3},
                                       have_nonconservative_terms, equations,
                                       volume_integral::VolumeIntegralFluxDifferencing,
                                       dg::DGSEM, cache)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    nodes = eachnode(dg)
    kernel! = flux_differencing_kernel_gpu!(backend)
    _nnodes = nnodes(dg)
    kernel!(du, u, equations,
            typeof(mesh),
            have_nonconservative_terms,
            combine_conservative_and_nonconservative_fluxes(volume_flux, equations),
            dg,
            volume_integral, _nnodes,
            derivative_split,
            contravariant_vectors,
            ndrange = (_nnodes, _nnodes, _nnodes, nelements(dg, cache)))
    return nothing
end

@kernel function flux_differencing_kernel!(du, u, equations,
                                           MeshT::Type{<:P4estMesh{3}},
                                           have_nonconservative_terms::False,
                                           combine_conservative_and_nonconservative_fluxes::False,
                                           dg::DGSEM,
                                           volume_integral,
                                           num_nodes,
                                           derivative_split,
                                           contravariant_vectors,
                                           alpha = true)
    # true * [some floating point value] == [exactly the same floating point value]
    # This can (hopefully) be optimized away due to constant propagation.
    i, j, k, element = @index(Global, NTuple)

    @unpack volume_flux = volume_integral

    # Calculate volume integral in one element
    u_node = get_node_vars(u, equations, dg, i, j, k, element)

    # pull the contravariant vectors in each coordinate direction
    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)

    # All diagonal entries of `derivative_split` are zero. Thus, we can skip
    # the computation of the diagonal terms. In addition, we use the symmetry
    # of the `volume_flux` to save half of the possible two-point flux
    # computations.

    KernelAbstractions.Extras.@unroll for other in min(i, j, k):num_nodes
        if other > i
            u_node_ii = get_node_vars(u, equations, dg, other, j, k, element)

            # pull the contravariant vectors and compute the average
            Ja1_node_ii = get_contravariant_vector(1, contravariant_vectors,
                                                   other, j, k, element)
            Ja1_avg = 0.5f0 * (Ja1_node + Ja1_node_ii)
            # compute the contravariant sharp flux in the direction of the
            # averaged contravariant vector
            fluxtilde1 = volume_flux(u_node, u_node_ii, Ja1_avg, equations)
            multiply_add_to_first_axis_atomic!(du, alpha * derivative_split[i, other],
                                               fluxtilde1,
                                               i, j, k, element)
            multiply_add_to_first_axis_atomic!(du, alpha * derivative_split[other, i],
                                               fluxtilde1,
                                               other, j, k, element)
        end
        if other > j
            u_node_jj = get_node_vars(u, equations, dg, i, other, k, element)
            # pull the contravariant vectors and compute the average
            Ja2_node_jj = get_contravariant_vector(2, contravariant_vectors,
                                                   i, other, k, element)
            Ja2_avg = 0.5f0 * (Ja2_node + Ja2_node_jj)
            # compute the contravariant sharp flux in the direction of the
            # averaged contravariant vector
            fluxtilde2 = volume_flux(u_node, u_node_jj, Ja2_avg, equations)
            multiply_add_to_first_axis_atomic!(du, alpha * derivative_split[j, other],
                                               fluxtilde2,
                                               i, j, k, element)
            multiply_add_to_first_axis_atomic!(du, alpha * derivative_split[other, j],
                                               fluxtilde2,
                                               i, other, k, element)
        end
        if other > k
            u_node_kk = get_node_vars(u, equations, dg, i, j, other, element)
            # pull the contravariant vectors and compute the average
            Ja3_node_kk = get_contravariant_vector(3, contravariant_vectors,
                                                   i, j, other, element)
            Ja3_avg = 0.5f0 * (Ja3_node + Ja3_node_kk)
            # compute the contravariant sharp flux in the direction of the
            # averaged contravariant vector
            fluxtilde3 = volume_flux(u_node, u_node_kk, Ja3_avg, equations)
            multiply_add_to_first_axis_atomic!(du, alpha * derivative_split[k, other],
                                               fluxtilde3,
                                               i, j, k, element)
            multiply_add_to_first_axis_atomic!(du, alpha * derivative_split[other, k],
                                               fluxtilde3,
                                               i, j, other, element)
        end
    end
end

@kernel function volume_integral_KAkernel!(du, u, MeshT,
                                           have_nonconservative_terms, equations,
                                           volume_integral, dg::DGSEM, cache)
    element = @index(Global)
    volume_integral_kernel!(du, u, element, MeshT, have_nonconservative_terms,
                            equations, volume_integral, dg, cache)
end

function prolong2interfaces!(backend::Backend, cache, u,
                             mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                             equations, dg::DG)
    @unpack interfaces = cache
    @unpack neighbor_ids, node_indices = cache.interfaces
    index_range = eachnode(dg)

    kernel! = prolong2interfaces_KAkernel!(backend)
    kernel!(interfaces.u, u, typeof(mesh), equations, neighbor_ids, node_indices,
            index_range,
            ndrange = ninterfaces(interfaces))
    return nothing
end

@kernel function prolong2interfaces_KAkernel!(interface_u, u, MeshT, equations,
                                              neighbor_ids, node_indices, index_range)
    interface = @index(Global)
    prolong2interfaces_per_interface!(interface_u, u, MeshT, equations, neighbor_ids,
                                      node_indices, index_range, interface)
end

function calc_surface_integral!(backend::Backend, du, u,
                                mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                equations,
                                surface_integral::SurfaceIntegralWeakForm,
                                dg::DGSEM, cache)
    @unpack inverse_weights = dg.basis
    @unpack surface_flux_values = cache.elements

    kernel! = calc_surface_integral_KAkernel!(backend)
    kernel!(du, typeof(mesh), equations, surface_integral, dg, inverse_weights[1],
            surface_flux_values, ndrange = nelements(cache.elements))
    return nothing
end

@kernel function calc_surface_integral_KAkernel!(du, MeshT, equations,
                                                 surface_integral, dg, factor,
                                                 surface_flux_values)
    element = @index(Global)
    calc_surface_integral_per_element!(du, MeshT,
                                       equations, surface_integral, dg, factor,
                                       surface_flux_values, element)
end

function calc_interface_flux!(backend::Backend, surface_flux_values,
                              mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                              have_nonconservative_terms,
                              equations, surface_integral, dg::DG, cache)
    @unpack neighbor_ids, node_indices = cache.interfaces
    @unpack contravariant_vectors = cache.elements
    index_range = eachnode(dg)

    kernel! = calc_interface_flux_KAkernel!(backend)
    kernel!(surface_flux_values, typeof(mesh), have_nonconservative_terms, equations,
            surface_integral, typeof(dg), cache.interfaces.u,
            neighbor_ids, node_indices, contravariant_vectors, index_range,
            ndrange = ninterfaces(cache.interfaces))
    return nothing
end

@kernel function calc_interface_flux_KAkernel!(surface_flux_values, MeshT,
                                               have_nonconservative_terms, equations,
                                               surface_integral, SolverT, u_interface,
                                               neighbor_ids, node_indices,
                                               contravariant_vectors, index_range)
    interface = @index(Global)
    calc_interface_flux_per_interface!(surface_flux_values,
                                       MeshT,
                                       have_nonconservative_terms,
                                       equations, surface_integral, SolverT,
                                       u_interface,
                                       neighbor_ids, node_indices,
                                       contravariant_vectors,
                                       index_range, interface)
end

function prolong2boundaries!(backend::Backend, cache, u,
                             mesh::P4estMesh,
                             equations, dg::DG)
    @unpack boundaries = cache
    @unpack neighbor_ids, node_indices = boundaries
    nboundaries = length(eachboundary(dg, cache))
    nboundaries == 0 && return nothing
    index_range = eachnode(dg)
    kernel! = prolong2boundaries_kernel!(backend)
    kernel!(u, typeof(mesh), equations, dg, index_range, boundaries.u, neighbor_ids,
            node_indices, ndrange = nboundaries)
    return nothing
end

@kernel function prolong2boundaries_kernel!(u, MeshT, equations, dg, index_range,
                                            u_boundaries, neighbor_ids, node_indices)
    boundary = @index(Global)
    prolong2boundaries_per_boundary!(u, MeshT, equations, dg, index_range, u_boundaries,
                                     neighbor_ids, node_indices, boundary)
end

function calc_boundary_flux!(backend::Backend, cache, t::Real,
                             boundary_condition::BoundaryConditionPeriodic,
                             mesh::P4estMesh,
                             equations, surface_integral, dg::DG)
    @assert isempty(eachboundary(dg, cache))

    return nothing
end

function calc_boundary_flux!(backend::Backend, cache, t, boundary_conditions,
                             mesh::Union{UnstructuredMesh2D, P4estMesh, T8codeMesh},
                             equations, surface_integral, dg::DG)
    @unpack boundary_condition_types, boundary_indices = boundary_conditions
    @unpack node_coordinates, contravariant_vectors = cache.elements
    calc_boundary_flux_by_type!(backend, cache, t,
                                boundary_condition_types, boundary_indices,
                                mesh, equations, surface_integral, dg,
                                node_coordinates, contravariant_vectors)
    return nothing
end

function calc_boundary_flux_by_type!(backend::Backend, cache, t,
                                     BCs::Tuple{},
                                     BC_indices::Tuple{},
                                     mesh, equations, surface_integral, dg,
                                     node_coordinates, contravariant_vectors)
    return nothing
end

function calc_boundary_flux_by_type!(backend::Backend, cache, t,
                                     BCs::Tuple{Any, Vararg{Any}},
                                     BC_indices::Tuple{AbstractVector{Int},
                                                       Vararg{AbstractVector{Int}}},
                                     mesh::Union{UnstructuredMesh2D, P4estMesh,
                                                 T8codeMesh},
                                     equations, surface_integral, dg::DG,
                                     node_coordinates, contravariant_vectors)
    boundary_condition = first(BCs)
    boundary_condition_indices = first(BC_indices)
    length(boundary_condition_indices) == 0 && return nothing
    @unpack boundaries = cache
    @unpack neighbor_ids, node_indices = boundaries

    index_range = eachnode(dg)
    n_boundaries = length(boundary_condition_indices)
    kernel_cache = kernel_filter_cache(cache)
    kernel! = calc_boundary_flux_kernel!(backend)
    kernel!(boundaries.u,
            cache.elements.surface_flux_values,
            boundary_condition_indices,
            neighbor_ids,
            node_indices,
            t,
            boundary_condition,
            index_range,
            typeof(mesh),
            equations,
            surface_integral,
            dg,
            kernel_cache, node_coordinates, contravariant_vectors;
            ndrange = n_boundaries)

    calc_boundary_flux_by_type!(backend, cache, t,
                                Base.tail(BCs),
                                Base.tail(BC_indices),
                                mesh, equations, surface_integral, dg,
                                node_coordinates, contravariant_vectors)
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
                                            mesh,
                                            equations,
                                            surface_integral,
                                            dg,
                                            cache, node_coordinates,
                                            contravariant_vectors)
    local_index = @index(Global, Linear)

    if local_index <= length(boundary_condition_indices)
        boundary = boundary_condition_indices[local_index]

        calc_boundary_flux_per_boundary!(u,
                                         surface_flux_values, t, boundary_condition,
                                         mesh, equations, surface_integral, dg, cache,
                                         boundary, neighbor_ids, node_indices_arr,
                                         index_range, node_coordinates,
                                         contravariant_vectors)
    end
end

function calc_boundary_flux_per_boundary!(u,
                                          surface_flux_values, t, boundary_condition,
                                          MeshT::Type{<:P4estMesh{3}},
                                          equations, surface_integral, dg, cache,
                                          boundary, neighbor_ids, node_indices_arr,
                                          index_range, node_coordinates,
                                          contravariant_vectors)

    # Get information on the adjacent element, compute the surface fluxes,
    # and store them
    element = neighbor_ids[boundary]
    node_indices = node_indices_arr[boundary]
    direction = indices2direction(node_indices)

    i_node_start, i_node_step_i, i_node_step_j = index_to_start_step_3d(node_indices[1],
                                                                        index_range)
    j_node_start, j_node_step_i, j_node_step_j = index_to_start_step_3d(node_indices[2],
                                                                        index_range)
    k_node_start, k_node_step_i, k_node_step_j = index_to_start_step_3d(node_indices[3],
                                                                        index_range)

    i_node = i_node_start
    j_node = j_node_start
    k_node = k_node_start
    for j in eachnode(dg)
        for i in eachnode(dg)
            calc_boundary_flux!(u, surface_flux_values, t, boundary_condition, MeshT,
                                have_nonconservative_terms(equations), equations,
                                surface_integral, dg, cache, i_node, j_node, k_node,
                                i, j, direction, element, boundary, node_coordinates,
                                contravariant_vectors)
            i_node += i_node_step_i
            j_node += j_node_step_i
            k_node += k_node_step_i
        end
        i_node += i_node_step_j
        j_node += j_node_step_j
        k_node += k_node_step_j
    end
end

# inlined version of the boundary flux calculation along a physical interface
@inline function calc_boundary_flux!(u, surface_flux_values, t, boundary_condition,
                                     MeshT,
                                     have_nonconservative_terms::False, equations,
                                     surface_integral, dg, cache,
                                     i_index, j_index, k_index, i_node_index,
                                     j_node_index,
                                     direction_index, element,
                                     boundary, node_coordinates,
                                     contravariant_vectors)
    @unpack surface_flux = surface_integral

    # Extract solution data from boundary container
    u_inner = get_node_vars(u, equations, dg, i_node_index, j_node_index, boundary)

    # Outward-pointing normal direction (not normalized)
    normal_direction = get_normal_direction(direction_index, contravariant_vectors,
                                            i_index, j_index, k_index, element)

    # Coordinates at boundary node
    x = get_node_coords(node_coordinates, equations, dg,
                        i_index, j_index, k_index, element)

    flux_ = boundary_condition(u_inner, normal_direction, x, t, surface_flux, equations)

    # Copy flux to element storage in the correct orientation
    for v in eachvariable(equations)
        surface_flux_values[v, i_node_index, j_node_index, direction_index, element] = flux_[v]
    end
end

function apply_jacobian!(backend::Backend, du,
                         mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                         equations, dg::DG, cache)
    @unpack inverse_jacobian = cache.elements

    kernel! = apply_jacobian_KAkernel!(backend)
    kernel!(du, typeof(mesh), equations, dg, inverse_jacobian,
            ndrange = (nnodes(dg), nnodes(dg), nnodes(dg), nelements(dg, cache)))
    return nothing
end

@kernel function apply_jacobian_KAkernel!(du, MeshT, equations, dg::DG,
                                          inverse_jacobian)
    i, j, k, element = @index(Global, NTuple)
    apply_jacobian_per_quadrature_node!(du, MeshT, equations, dg, inverse_jacobian,
                                        i, j, k, element)
end

@kernel function calc_sources_KAkernel!(du, u, t, source_terms,
                                        node_coordinates,
                                        equations::AbstractEquations{3}, dg, cache)
    i, j, k, element = @index(Global, NTuple)
    u_local = get_node_vars(u, equations, dg, i, j, k, element)
    x_local = get_node_coords(node_coordinates, equations, dg, i, j, k, element)

    du_local = source_terms(u_local, x_local, t, equations)

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

function calc_sources!(backend::Backend, du, u, t, source_terms,
                       equations::AbstractEquations{3}, dg::DG, cache)
    nelements(dg, cache) == 0 && return nothing
    @unpack node_coordinates = cache.elements
    kernel_cache = kernel_filter_cache(cache)
    kernel! = calc_sources_KAkernel!(backend)
    kernel!(du, u, t, source_terms, node_coordinates, equations, dg, kernel_cache,
            ndrange = (nnodes(dg), nnodes(dg), nnodes(dg), nelements(dg, cache)))

    return nothing
end

function calc_sources!(backend::Backend, du, u, t, source_terms::Nothing,
                       equations::AbstractEquations{3}, dg::DG, cache)
    return nothing
end
end #muladd
