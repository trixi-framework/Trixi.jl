# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function prolong2interfaces!(backend::Backend, cache, u,
                             mesh::Union{P4estMesh{2}, P4estMeshView{2},
                                         T8codeMesh{2}},
                             equations, dg::DGSEM{<:LobattoLegendreBasis})
    @unpack interfaces = cache
    ninterfaces(interfaces) == 0 && return nothing
    @unpack neighbor_ids, node_indices = cache.interfaces
    index_range = eachnode(dg)

    kernel! = prolong2interfaces_KAkernel!(backend)
    kernel!(interfaces.u, u, typeof(mesh), equations, neighbor_ids, node_indices,
            index_range, ndrange = ninterfaces(interfaces))
    return nothing
end

@kernel function prolong2interfaces_KAkernel!(interfaces_u, u,
                                              MeshT::Type{<:Union{P4estMesh{2},
                                                                  P4estMeshView{2},
                                                                  T8codeMesh{2}}},
                                              equations, neighbor_ids,
                                              node_indices, index_range)
    interface = @index(Global)
    prolong2interfaces_per_interface!(interfaces_u, u, interface, MeshT, equations,
                                      neighbor_ids, node_indices, index_range)
end

function calc_surface_integral!(backend::Backend, du, u,
                                mesh::Union{P4estMesh{2}, P4estMeshView{2},
                                            T8codeMesh{2}},
                                equations,
                                surface_integral::SurfaceIntegralWeakForm,
                                dg::DGSEM{<:LobattoLegendreBasis}, cache)
    nelements(dg, cache) == 0 && return nothing
    @unpack inverse_weights = dg.basis
    @unpack surface_flux_values = cache.elements

    kernel! = calc_surface_integral_KAkernel!(backend)
    kernel!(du, typeof(mesh), equations, surface_integral, dg, inverse_weights[1],
            surface_flux_values, ndrange = nelements(dg, cache))
    return nothing
end

@kernel function calc_surface_integral_KAkernel!(du,
                                                 MeshT::Type{<:Union{P4estMesh{2},
                                                                     P4estMeshView{2},
                                                                     T8codeMesh{2}}},
                                                 equations,
                                                 surface_integral::SurfaceIntegralWeakForm,
                                                 dg::DGSEM{<:LobattoLegendreBasis},
                                                 factor,
                                                 surface_flux_values)
    element = @index(Global)
    calc_surface_integral_per_element!(du, MeshT, equations, surface_integral,
                                       dg, factor, surface_flux_values, element)
end

function calc_interface_flux!(backend::Backend, surface_flux_values,
                              mesh::Union{P4estMesh{2}, P4estMeshView{2},
                                          T8codeMesh{2}},
                              have_nonconservative_terms, have_aux_node_vars,
                              equations, surface_integral,
                              dg::DGSEM{<:LobattoLegendreBasis}, cache)
    ninterfaces(cache.interfaces) == 0 && return nothing
    @unpack neighbor_ids, node_indices = cache.interfaces
    @unpack contravariant_vectors = cache.elements
    index_range = eachnode(dg)

    kernel! = calc_interface_flux_KAkernel!(backend)
    kernel!(surface_flux_values, typeof(mesh), have_nonconservative_terms,
            equations, surface_integral, typeof(dg), cache.interfaces.u,
            neighbor_ids, node_indices, contravariant_vectors, index_range,
            ndrange = ninterfaces(cache.interfaces))

    return nothing
end

@kernel function calc_interface_flux_KAkernel!(surface_flux_values,
                                               MeshT::Type{<:Union{P4estMesh{2},
                                                                   P4estMeshView{2},
                                                                   T8codeMesh{2}}},
                                               have_nonconservative_terms,
                                               equations, surface_integral,
                                               SolverT::Type{<:DG}, u_interface,
                                               neighbor_ids, node_indices,
                                               contravariant_vectors, index_range)
    interface = @index(Global)
    calc_interface_flux_per_interface!(surface_flux_values, MeshT,
                                       have_nonconservative_terms, equations,
                                       surface_integral, SolverT, u_interface,
                                       interface, neighbor_ids, node_indices,
                                       contravariant_vectors, index_range)
end

function apply_jacobian!(backend::Backend, du,
                         mesh::Union{P4estMesh{2}, P4estMeshView{2},
                                     T8codeMesh{2}},
                         equations, dg::DG, cache)
    nelements(dg, cache) == 0 && return nothing
    @unpack inverse_jacobian = cache.elements
    kernel! = apply_jacobian_KAkernel!(backend)
    kernel!(du, typeof(mesh), equations, dg, inverse_jacobian,
            ndrange = (nnodes(dg), nnodes(dg), nelements(dg, cache)))
end

@kernel function apply_jacobian_KAkernel!(du,
                                          MeshT::Type{<:Union{P4estMesh{2},
                                                              P4estMeshView{2},
                                                              T8codeMesh{2}}},
                                          equations, dg::DG, inverse_jacobian)
    i, j, element = @index(Global, NTuple)
    apply_jacobian_per_quadrature_node!(du, MeshT, equations, dg, inverse_jacobian,
                                        i, j, element)
end

@kernel function calc_sources_KAkernel!(du, u, t, source_terms,
                                        node_coordinates,
                                        equations::AbstractEquations{2}, dg, cache)
    i, j, element = @index(Global, NTuple)
    u_local = get_node_vars(u, equations, dg, i, j, element)
    x_local = get_node_coords(node_coordinates, equations, dg, i, j, element)

    du_local = source_terms(u_local, x_local, t, equations)

    add_to_node_vars!(du, du_local, equations, dg, i, j, element)
end

function calc_sources!(backend::Backend, du, u, t, source_terms,
                       equations::AbstractEquations{2}, dg::DG, cache)
    nelements(dg, cache) == 0 && return nothing
    @unpack node_coordinates = cache.elements
    kernel_cache = kernel_filter_cache(cache)
    kernel! = calc_sources_KAkernel!(backend)
    kernel!(du, u, t, source_terms, node_coordinates, equations, dg, kernel_cache,
            ndrange = (nnodes(dg), nnodes(dg), nelements(dg, cache)))

    return nothing
end

function calc_sources!(backend::Backend, du, u, t, source_terms::Nothing,
                       equations::AbstractEquations{2}, dg::DG, cache)
    return nothing
end
end #muladd
