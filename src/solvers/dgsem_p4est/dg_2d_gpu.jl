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

@inline function prolong2mortars_per_mortar!(mortars_u, u, mortar,
                                             MeshT::Type{<:Union{Trixi.P4estMesh{2},
                                                                 Trixi.P4estMeshView{2},
                                                                 Trixi.T8codeMesh{2}}},
                                             equations,
                                             neighbor_ids, node_indices,
                                             index_range,
                                             forward_lower,
                                             forward_upper,
                                             ::Val{N}, ::Val{NVARS}, ::Val{T},
                                             ::Val{L}) where {N, NVARS, T, L}
    @inbounds begin
        small_indices = node_indices[1, mortar]
        i_small_start, i_small_step = Trixi.index_to_start_step_2d(small_indices[1],
                                                                   index_range)
        j_small_start, j_small_step = Trixi.index_to_start_step_2d(small_indices[2],
                                                                   index_range)

        for position in 1:2
            i_small = i_small_start
            j_small = j_small_start
            element = neighbor_ids[position, mortar]
            for i in eachindex(index_range)
                for v in Base.OneTo(NVARS)
                    mortars_u[1, v, position, i, mortar] = u[v, i_small, j_small,
                                                             element]
                end
                i_small += i_small_step
                j_small += j_small_step
            end
        end

        u_buffer = MArray{Tuple{NVARS, N}, T, 2, NVARS * N}(undef)

        large_indices = node_indices[2, mortar]
        i_large_start, i_large_step = Trixi.index_to_start_step_2d(large_indices[1],
                                                                   index_range)
        j_large_start, j_large_step = Trixi.index_to_start_step_2d(large_indices[2],
                                                                   index_range)

        i_large = i_large_start
        j_large = j_large_start
        element = neighbor_ids[3, mortar]
        for i in eachindex(index_range)
            for v in Base.OneTo(NVARS)
                u_buffer[v, i] = u[v, i_large, j_large, element]
            end
            i_large += i_large_step
            j_large += j_large_step
        end

        val_lower = StaticArrays.MArray{Tuple{NVARS, N}, T, 2, L}(undef)
        val_upper = StaticArrays.MArray{Tuple{NVARS, N}, T, 2, L}(undef)

        for i in 1:N, v in Base.OneTo(NVARS)
            val_lower[v, i] = zero(T)
            val_upper[v, i] = zero(T)
        end

        Trixi.multiply_dimensionwise!(val_lower, forward_lower, u_buffer)
        Trixi.multiply_dimensionwise!(val_upper, forward_upper, u_buffer)

        for i in 1:N
            for v in Base.OneTo(NVARS)
                mortars_u[2, v, 1, i, mortar] = val_lower[v, i]
                mortars_u[2, v, 2, i, mortar] = val_upper[v, i]
            end
        end
    end # @inbounds

    return nothing
end

@kernel function prolong2mortars_KAkernel!(mortars_u, u,
                                           MeshT::Type{<:Union{Trixi.P4estMesh{2},
                                                               Trixi.P4estMeshView{2},
                                                               Trixi.T8codeMesh{2}}},
                                           equations,
                                           neighbor_ids, node_indices,
                                           index_range,
                                           forward_lower,
                                           forward_upper,
                                           ::Val{N}, ::Val{NVARS}, ::Val{T},
                                           ::Val{L}) where {N, NVARS, T, L}
    mortar = @index(Global)
    prolong2mortars_per_mortar!(mortars_u, u, mortar, MeshT, equations,
                                neighbor_ids, node_indices, index_range,
                                forward_lower, forward_upper, Val(N), Val(NVARS),
                                Val(T), Val(L))
end

function Trixi.prolong2mortars!(backend::KernelAbstractions.Backend, cache, u,
                                mesh::Union{Trixi.P4estMesh{2}, Trixi.P4estMeshView{2},
                                            Trixi.T8codeMesh{2}},
                                equations,
                                mortar_l2::Trixi.LobattoLegendreMortarL2,
                                dg::Trixi.DGSEM{<:Trixi.LobattoLegendreBasis})
    Trixi.nmortars(dg, cache) == 0 && return nothing

    @unpack mortars = cache
    @unpack neighbor_ids, node_indices = cache.mortars

    index_range = Trixi.eachnode(dg)

    N = Trixi.nnodes(dg)
    T = eltype(u)
    NVARS = Trixi.nvariables(equations)
    L = N * NVARS

    kernel! = prolong2mortars_KAkernel!(backend)
    kernel!(mortars.u, u, typeof(mesh), equations,
            neighbor_ids, node_indices, index_range,
            mortar_l2.forward_lower, mortar_l2.forward_upper,
            Val(N), Val(NVARS), Val(T), Val(L);
            ndrange = Trixi.nmortars(dg, cache))

    return nothing
end

@inline function mortar_fluxes_to_elements!(surface_flux_values,
                                            neighbor_ids, node_indices,
                                            reverse_lower, reverse_upper,
                                            mortar,
                                            fstar_primary_1, fstar_primary_2,
                                            fstar_s_lower, fstar_s_upper,
                                            u_buffer, N, NVARS)
    small_indices = node_indices[1, mortar]
    small_direction = Trixi.indices2direction(small_indices)

    for position in 1:2
        element = neighbor_ids[position, mortar]
        for i in 1:N
            for v in Base.OneTo(NVARS)
                surface_flux_values[v, i, small_direction, element] = position == 1 ?
                                                                      fstar_primary_1[v,
                                                                                      i] :
                                                                      fstar_primary_2[v,
                                                                                      i]
            end
        end
    end

    multiply_dimensionwise!(u_buffer,
                            reverse_upper, fstar_s_upper,
                            reverse_lower, fstar_s_lower)

    u_buffer .*= -2

    large_element = neighbor_ids[3, mortar]
    large_indices = node_indices[2, mortar]
    large_direction = Trixi.indices2direction(large_indices)

    if :i_backward in large_indices
        for i in 1:N
            for v in Base.OneTo(NVARS)
                surface_flux_values[v, N + 1 - i, large_direction,
                                    large_element] = u_buffer[v, i]
            end
        end
    else
        for i in 1:N
            for v in Base.OneTo(NVARS)
                surface_flux_values[v, i, large_direction, large_element] = u_buffer[v,
                                                                                     i]
            end
        end
    end

    return nothing
end

@inline function gpu_calc_mortar_flux!(fstar_p_1, fstar_p_2, fstar_s_1, fstar_s_2,
                                       MeshT,
                                       have_nonconservative_terms::Trixi.False,
                                       equations,
                                       pure_surface_flux, dg::Trixi.DGSEM, mortar_u,
                                       mortar_index, position_index, normal_direction,
                                       node_index)
    u_ll, u_rr = Trixi.get_surface_node_vars(mortar_u, equations, dg, position_index,
                                             node_index, mortar_index)

    flux = pure_surface_flux(u_ll, u_rr, normal_direction, equations)

    if position_index == 1
        Trixi.set_node_vars!(fstar_p_1, flux, equations, dg, node_index)
        Trixi.set_node_vars!(fstar_s_1, flux, equations, dg, node_index)
    else
        Trixi.set_node_vars!(fstar_p_2, flux, equations, dg, node_index)
        Trixi.set_node_vars!(fstar_s_2, flux, equations, dg, node_index)
    end
    return nothing
end

@inline function gpu_calc_mortar_flux!(fstar_p_1, fstar_p_2, fstar_s_1, fstar_s_2,
                                       MeshT,
                                       have_nonconservative_terms::Trixi.True,
                                       equations,
                                       pure_surface_flux, dg::Trixi.DGSEM, mortar_u,
                                       mortar_index, position_index, normal_direction,
                                       node_index)
    surface_flux, nonconservative_flux = pure_surface_flux

    u_ll, u_rr = Trixi.get_surface_node_vars(mortar_u, equations, dg, position_index,
                                             node_index, mortar_index)

    flux = surface_flux(u_ll, u_rr, normal_direction, equations)

    noncons_primary = nonconservative_flux(u_ll, u_rr, normal_direction, equations)
    noncons_secondary = nonconservative_flux(u_rr, u_ll, normal_direction, equations)

    flux_plus_noncons_primary = flux + 0.5f0 * noncons_primary
    flux_plus_noncons_secondary = flux + 0.5f0 * noncons_secondary

    if position_index == 1
        Trixi.set_node_vars!(fstar_p_1, flux_plus_noncons_primary, equations, dg,
                             node_index)
        Trixi.set_node_vars!(fstar_s_1, flux_plus_noncons_secondary, equations, dg,
                             node_index)
    else
        Trixi.set_node_vars!(fstar_p_2, flux_plus_noncons_primary, equations, dg,
                             node_index)
        Trixi.set_node_vars!(fstar_s_2, flux_plus_noncons_secondary, equations, dg,
                             node_index)
    end

    return nothing
end

@kernel function calc_mortar_flux_KAkernel!(surface_flux_values,
                                            MeshT::Type{<:Union{Trixi.P4estMesh{2},
                                                                Trixi.P4estMeshView{2},
                                                                Trixi.T8codeMesh{2}}},
                                            have_nonconservative_terms, equations,
                                            pure_surface_flux, dg::Trixi.DGSEM,
                                            mortars_u, neighbor_ids, node_indices,
                                            contravariant_vectors,
                                            reverse_lower, reverse_upper, index_range,
                                            ::Val{N}, ::Val{NVARS}, ::Val{T},
                                            ::Val{L}) where {N, NVARS, T, L}
    mortar = @index(Global)

    @inbounds begin
        fstar_p_1 = MArray{Tuple{NVARS, N}, T, 2, L}(undef)
        fstar_p_2 = MArray{Tuple{NVARS, N}, T, 2, L}(undef)
        fstar_s_1 = MArray{Tuple{NVARS, N}, T, 2, L}(undef)
        fstar_s_2 = MArray{Tuple{NVARS, N}, T, 2, L}(undef)
        u_buffer = MArray{Tuple{NVARS, N}, T, 2, L}(undef)

        small_indices = node_indices[1, mortar]
        small_direction = Trixi.indices2direction(small_indices)

        i_small_start, i_small_step = Trixi.index_to_start_step_2d(small_indices[1],
                                                                   index_range)
        j_small_start, j_small_step = Trixi.index_to_start_step_2d(small_indices[2],
                                                                   index_range)

        for position in 1:2
            i_small = i_small_start
            j_small = j_small_start
            element = neighbor_ids[position, mortar]

            for node in 1:N
                normal_direction = Trixi.get_normal_direction(small_direction,
                                                              contravariant_vectors,
                                                              i_small, j_small, element)

                gpu_calc_mortar_flux!(fstar_p_1, fstar_p_2, fstar_s_1, fstar_s_2,
                                      MeshT, have_nonconservative_terms, equations,
                                      pure_surface_flux, dg, mortars_u,
                                      mortar, position, normal_direction, node)

                i_small += i_small_step
                j_small += j_small_step
            end
        end

        mortar_fluxes_to_elements!(surface_flux_values,
                                   neighbor_ids, node_indices,
                                   reverse_lower, reverse_upper,
                                   mortar,
                                   fstar_p_1, fstar_p_2,
                                   fstar_s_1, fstar_s_2,
                                   u_buffer, N, NVARS)
    end
end

function Trixi.calc_mortar_flux!(backend::KernelAbstractions.Backend,
                                 surface_flux_values,
                                 mesh::Union{Trixi.P4estMesh{2}, Trixi.P4estMeshView{2},
                                             Trixi.T8codeMesh{2}},
                                 have_nonconservative_terms, equations,
                                 mortar_l2::Trixi.LobattoLegendreMortarL2,
                                 surface_integral, dg::Trixi.DGSEM, cache)
    Trixi.nmortars(dg, cache) == 0 && return nothing

    @unpack neighbor_ids, node_indices = cache.mortars
    @unpack contravariant_vectors = cache.elements
    mortars_u = cache.mortars.u
    pure_surface_flux = surface_integral.surface_flux
    index_range = Trixi.eachnode(dg)

    N = Trixi.nnodes(dg)
    NVARS = Trixi.nvariables(equations)
    T = eltype(surface_flux_values)
    L = N * NVARS

    kernel! = calc_mortar_flux_KAkernel!(backend)

    kernel!(surface_flux_values, typeof(mesh), have_nonconservative_terms,
            equations, pure_surface_flux, dg,
            mortars_u, neighbor_ids, node_indices, contravariant_vectors,
            mortar_l2.reverse_lower, mortar_l2.reverse_upper, index_range,
            Val(N), Val(NVARS), Val(T), Val(L);
            ndrange = Trixi.nmortars(dg, cache))

    return nothing
end
end #muladd
