# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

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
                             mesh::Union{P4estMesh{2}, P4estMeshView{2}, T8codeMesh{2}},
                             equations, dg::DG)
    @unpack boundaries = cache
		@unpack neighbor_ids, node_indices = boundaries
    index_range = eachnode(dg)
    nboundaries = length(eachboundary(dg, cache))  
	kernel_cache = kernel_filter_cache(cache)
		kernel! = prolong2boundaries_kernel!(backend)
		kernel!(u, typeof(mesh), equations, dg, index_range, boundaries.u, neighbor_ids, node_indices, ndrange = nboundaries)
    return nothing
end

@kernel function prolong2boundaries_kernel!(u, MeshT, equations, dg, index_range, u_boundaries, neighbor_ids, node_indices)

boundary = @index(Global)
prolong2boundaries_per_boundary!(u, MeshT, equations, dg, index_range, u_boundaries, neighbor_ids, node_indices, boundary)
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
