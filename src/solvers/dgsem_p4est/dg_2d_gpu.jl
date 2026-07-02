# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function rhs!(backend::Backend, du, u, t,
              mesh::Union{P4estMesh{2}, P4estMeshView{2}, T8codeMesh{2}, P4estMesh{3},
                          T8codeMesh{3}},
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

    # Prolong solution to interfaces
    @trixi_timeit_ext backend timer() "prolong2interfaces" begin
        prolong2interfaces!(backend, cache, u, mesh, equations, dg)
    end

    # Calculate interface fluxes
    @trixi_timeit_ext backend timer() "interface flux" begin
        calc_interface_flux!(backend, cache.elements.surface_flux_values, mesh,
                             have_nonconservative_terms(equations), equations,
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

    # Calculate surface integrals
    @trixi_timeit_ext backend timer() "surface integral" begin
        calc_surface_integral!(backend, du, u, mesh, equations,
                               dg.surface_integral, dg, cache)
    end

    # Apply Jacobian from mapping to reference element
    @trixi_timeit_ext backend timer() "Jacobian" begin
        apply_jacobian!(backend, du, mesh, equations, dg, cache)
    end

    # Calculate source terms
    @trixi_timeit_ext backend timer() "source terms" begin
        calc_sources!(backend, du, u, t, source_terms, equations, dg, cache)
    end

    return nothing
end

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

function calc_interface_flux!(backend::Backend, surface_flux_values,
                              mesh::Union{P4estMesh{2}, P4estMeshView{2},
                                          T8codeMesh{2}},
                              have_nonconservative_terms,
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

function prolong2boundaries_per_boundary!(u,
                                          MeshT::Type{<:Union{P4estMesh{2},
                                                              P4estMeshView{2},
                                                              T8codeMesh{2}}},
                                          equations, dg::DG, index_range, u_boundaries,
                                          neighbor_ids, node_indices, boundary)
    # Copy solution data from the element using "delayed indexing" with
    # a start value and a step size to get the correct face and orientation.
    element = neighbor_ids[boundary]
    node_index = node_indices[boundary]

    i_node_start, i_node_step = index_to_start_step_2d(node_index[1], index_range)
    j_node_start, j_node_step = index_to_start_step_2d(node_index[2], index_range)

    i_node = i_node_start
    j_node = j_node_start
    for i in eachnode(dg)
        for v in eachvariable(equations)
            u_boundaries[v, i, boundary] = u[v, i_node, j_node, element]
        end
        i_node += i_node_step
        j_node += j_node_step
    end

    return nothing
end

function calc_surface_integral!(backend::Backend, du, u,
                                mesh::Union{P4estMesh{2}, T8codeMesh{2},
                                            P4estMeshView{2}},
                                equations,
                                surface_integral::SurfaceIntegralWeakForm,
                                dg::DGSEM{<:LobattoLegendreBasis},
                                cache)
    @unpack inverse_weights = dg.basis
    @unpack surface_flux_values = cache.elements
    NNODES = nnodes(dg)
    kernel! = calc_surface_integral_KAkernel!(backend)
    kernel!(du, typeof(mesh), equations, inverse_weights[1],
            Val(NNODES),
            surface_flux_values,
            ndrange = (NNODES, NNODES, nelements(dg, cache)))

    return nothing
end

@kernel function calc_surface_integral_KAkernel!(du,
                                                 MeshT::Type{<:Union{P4estMesh{2},
                                                                     P4estMeshView{2},
                                                                     T8codeMesh{2}}},
                                                 equations, factor, ::Val{NNODES},
                                                 surface_flux_values) where {NNODES}
    i, j, element = @index(Global, NTuple)
    # Note that all fluxes have been computed with outward-pointing normal vectors.
    # This computes the **negative** surface integral contribution,
    # i.e., M^{-1} * boundary_interpolation^T (which is for Gauss-Lobatto DGSEM just M^{-1} * B)
    # and the missing "-" is taken care of by `apply_jacobian!`.
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
    for v in eachvariable(equations)
        x_contribution = ifelse(x_node_interface,
                                surface_flux_values[v, j, x_face, element], _zero)
        y_contribution = ifelse(y_node_interface,
                                surface_flux_values[v, i, y_face, element], _zero)
        du_node = x_contribution + y_contribution
        du[v, i, j, element] = du[v, i, j, element] + du_node * factor
    end
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
