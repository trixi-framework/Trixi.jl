# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function calc_volume_integral!(backend::Backend, du, u,
                                       mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                       have_nonconservative_terms, equations,
                                       volume_integral::VolumeIntegralFluxDifferencing,
                                       dg::DGSEM, cache)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    kernel! = flux_differencing_KAkernel!(backend)
    NNODES = nnodes(dg)
    kernel!(du, u, equations,
            typeof(mesh),
            have_nonconservative_terms,
            combine_conservative_and_nonconservative_fluxes(volume_integral.volume_flux,
                                                            equations),
            dg,
            volume_integral, Val(NNODES),
            derivative_split,
            contravariant_vectors,
            ndrange = (NNODES, NNODES, NNODES, nelements(dg, cache)))
    return nothing
end

@kernel function flux_differencing_KAkernel!(du, u, equations,
                                             MeshT::Type{<:Union{P4estMesh{3},
                                                                 T8codeMesh{3}}},
                                             have_nonconservative_terms::False,
                                             combine_conservative_and_nonconservative_fluxes::False,
                                             dg::DGSEM,
                                             volume_integral,
                                             ::Val{NNODES},
                                             derivative_split,
                                             contravariant_vectors,
                                             alpha = true) where {NNODES}
    # `true * [some floating point value] == [exactly the same floating point value]`
    # This can (hopefully) be optimized away due to constant propagation.
    i, j, k, element = @index(Global, NTuple)

    @unpack volume_flux = volume_integral

    u_node = get_node_vars(u, equations, dg, i, j, k, element)

    # pull the contravariant vectors in each coordinate direction
    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)

    # All diagonal entries of `derivative_split` are zero. Thus, we can skip
    # the computation of the diagonal terms. In addition, we use the symmetry
    # of the `volume_flux` to save half of the possible two-point flux
    # computations.
    #
    # Instead of assigning thread i the partners i+1, …, N,
    # we distribute the half-sweep cyclically: each thread visits
    # half = div(N,2) partners at a fixed rotating offset.
    # Every unordered pair is still covered exactly
    # once, but now every thread performs the same number of loop iterations.
    # When N is even (odd polynomial degree) the antipodal pair at
    # offset half is shared by two threads, so its contribution is weighted by
    # 1/2 to avoid double counting.
    #
    # See Section 4.1 (Eq. 6) of
    # - Waterhouse, Waruszewski, Wilcox, Giraldo (2026)
    #   GPU Performance of an Entropy-Stable Discontinuous Galerkin Euler Solver
    #   with Non-Conservative Terms.
    #   arXiv (pre-print): https://arxiv.org/abs/2605.16684

    half_nnodes = div(NNODES, 2)
    even_nodes = iseven(NNODES)

    KernelAbstractions.Extras.@unroll for offset in 1:half_nnodes
        # weight the antipodal pair by 1/2 only when the number of nodes is even
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # first coordinate direction: rotate the partner index along `i`
        ii = mod(i - 1 + offset, NNODES) + 1
        u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
        # pull the contravariant vectors and compute the average
        Ja1_node_ii = get_contravariant_vector(1, contravariant_vectors,
                                               ii, j, k, element)
        Ja1_avg = 0.5f0 * (Ja1_node + Ja1_node_ii)
        # compute the contravariant volume flux in the direction of the
        # averaged contravariant vector
        fluxtilde1 = volume_flux(u_node, u_node_ii, Ja1_avg, equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[i, ii],
                                           fluxtilde1,
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[ii, i],
                                           fluxtilde1,
                                           ii, j, k, element)

        # second coordinate direction: rotate the partner index along `j`
        jj = mod(j - 1 + offset, NNODES) + 1
        u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
        # pull the contravariant vectors and compute the average
        Ja2_node_jj = get_contravariant_vector(2, contravariant_vectors,
                                               i, jj, k, element)
        Ja2_avg = 0.5f0 * (Ja2_node + Ja2_node_jj)
        # compute the contravariant volume flux in the direction of the
        # averaged contravariant vector
        fluxtilde2 = volume_flux(u_node, u_node_jj, Ja2_avg, equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[j, jj],
                                           fluxtilde2,
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[jj, j],
                                           fluxtilde2,
                                           i, jj, k, element)

        # third coordinate direction: rotate the partner index along `k`
        kk = mod(k - 1 + offset, NNODES) + 1
        u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
        # pull the contravariant vectors and compute the average
        Ja3_node_kk = get_contravariant_vector(3, contravariant_vectors,
                                               i, j, kk, element)
        Ja3_avg = 0.5f0 * (Ja3_node + Ja3_node_kk)
        # compute the contravariant volume flux in the direction of the
        # averaged contravariant vector
        fluxtilde3 = volume_flux(u_node, u_node_kk, Ja3_avg, equations)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[k, kk],
                                           fluxtilde3,
                                           i, j, k, element)
        multiply_add_to_first_axis_atomic!(du,
                                           weight * alpha * derivative_split[kk, k],
                                           fluxtilde3,
                                           i, j, kk, element)
    end
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

@kernel function prolong2interfaces_KAkernel!(interface_u, u,
                                              MeshT::Type{<:Union{P4estMesh{3},
                                                                  T8codeMesh{3}}},
                                              equations,
                                              neighbor_ids, node_indices, index_range)
    interface = @index(Global)
    prolong2interfaces_per_interface!(interface_u, u,
                                      MeshT::Type{<:Union{P4estMesh{3},
                                                          T8codeMesh{3}}},
                                      equations, neighbor_ids,
                                      node_indices, index_range, interface)
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

@kernel function calc_interface_flux_KAkernel!(surface_flux_values,
                                               MeshT::Type{<:Union{P4estMesh{3},
                                                                   T8codeMesh{3}}},
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
                             mesh::Union{P4estMesh, T8codeMesh},
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

@kernel function prolong2boundaries_kernel!(u,
                                            MeshT::Type{<:Union{P4estMesh{3},
                                                                T8codeMesh{3}}},
                                            equations, dg, index_range,
                                            u_boundaries, neighbor_ids, node_indices)
    boundary = @index(Global)
    prolong2boundaries_per_boundary!(u, MeshT, equations, dg, index_range, u_boundaries,
                                     neighbor_ids, node_indices, boundary)
end

function calc_boundary_flux!(backend::Backend, cache, t::Real,
                             boundary_condition::BoundaryConditionPeriodic,
                             mesh::Union{P4estMesh, T8codeMesh},
                             equations, surface_integral, dg::DG)
    @assert isempty(eachboundary(dg, cache))

    return nothing
end

function calc_boundary_flux!(backend::Backend, cache, t, boundary_conditions,
                             mesh::Union{P4estMesh, T8codeMesh},
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
                                     mesh::Union{P4estMesh, T8codeMesh},
                                     equations, surface_integral, dg,
                                     node_coordinates, contravariant_vectors)
    return nothing
end

function calc_boundary_flux_by_type!(backend::Backend, cache, t,
                                     BCs::Tuple{Any, Vararg{Any}},
                                     BC_indices::Tuple{AbstractVector{Int},
                                                       Vararg{AbstractVector{Int}}},
                                     mesh::Union{P4estMesh,
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
                                            MeshT::Type{<:Union{P4estMesh,
                                                                T8codeMesh}},
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
                                         MeshT, equations, surface_integral, dg, cache,
                                         boundary, neighbor_ids, node_indices_arr,
                                         index_range, node_coordinates,
                                         contravariant_vectors)
    end
end

function calc_boundary_flux_per_boundary!(u,
                                          surface_flux_values, t, boundary_condition,
                                          MeshT::Type{<:Union{P4estMesh{3},
                                                              T8codeMesh{3}}},
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
                                     MeshT::Type{<:Union{P4estMesh{3},
                                                         T8codeMesh{3}}},
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
        surface_flux_values[v, i_node_index, j_node_index, direction_index,
                            element] = flux_[v]
    end
end

function calc_surface_integral!(backend::Backend, du, u,
                                mesh::Union{P4estMesh{3}, T8codeMesh{3}},
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
            ndrange = (NNODES, NNODES, NNODES, nelements(dg, cache)))

    return nothing
end

@kernel function calc_surface_integral_KAkernel!(du,
                                                 MeshT::Type{<:Union{P4estMesh{3},
                                                                     T8codeMesh{3}}},
                                                 equations, factor, ::Val{NNODES},
                                                 surface_flux_values) where {NNODES}
    i, j, k, element = @index(Global, NTuple)
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
    z_node_interface = (k == 1) | (k == NNODES)
    x_face = ifelse(i == 1, 1, 2)
    y_face = ifelse(j == 1, 3, 4)
    z_face = ifelse(k == 1, 5, 6)
    _zero = zero(eltype(du))
    for v in eachvariable(equations)
        x_contribution = ifelse(x_node_interface,
                                surface_flux_values[v, j, k, x_face, element], _zero)
        y_contribution = ifelse(y_node_interface,
                                surface_flux_values[v, i, k, y_face, element], _zero)
        z_contribution = ifelse(z_node_interface,
                                surface_flux_values[v, i, j, z_face, element], _zero)
        du_node = x_contribution + y_contribution + z_contribution
        du[v, i, j, k, element] = du[v, i, j, k, element] + du_node * factor
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

@kernel function apply_jacobian_KAkernel!(du,
                                          MeshT::Type{<:Union{P4estMesh{3},
                                                              T8codeMesh{3}}},
                                          equations, dg::DG,
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

function prolong2mortars!(backend::KernelAbstractions.Backend, cache, u,
                          mesh::Union{Trixi.P4estMesh{3}, P4estMeshView{3},
                                      Trixi.T8codeMesh{3}},
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

    L = NVARS * N * N

    kernel! = prolong2mortars_KAkernel!(backend)
    kernel!(mortars.u, u, typeof(mesh), equations,
            neighbor_ids, node_indices, index_range,
            mortar_l2.forward_lower, mortar_l2.forward_upper,
            Val(N), Val(NVARS), Val(T), Val(L);
            ndrange = (Trixi.nmortars(dg, cache)))

    return nothing
end

@kernel function prolong2mortars_KAkernel!(mortars_u, u,
                                           MeshT::Type{<:Union{Trixi.P4estMesh{3},
                                                               P4estMeshView{3},
                                                               Trixi.T8codeMesh{3}}},
                                           equations,
                                           neighbor_ids, node_indices,
                                           index_range,
                                           forward_lower, forward_upper,
                                           ::Val{N}, ::Val{NVARS}, ::Val{T},
                                           ::Val{L}) where {N, NVARS, T, L}
    mortar = @index(Global)
    prolong2mortars_per_mortar!(mortars_u, u, mortar, MeshT, equations,
                                neighbor_ids, node_indices, index_range,
                                forward_lower, forward_upper,
                                Val(N), Val(NVARS), Val(T), Val(L))
end

@inline function prolong2mortars_per_mortar!(mortars_u, u, mortar,
                                             MeshT, equations,
                                             neighbor_ids, node_indices,
                                             index_range,
                                             forward_lower, forward_upper,
                                             ::Val{N}, ::Val{NVARS}, ::Val{T},
                                             ::Val{L}) where {N, NVARS, T, L}
    @inbounds begin

        # Copy solution data from the small elements using "delayed indexing" with
        # a start value and two step sizes to get the correct face and orientation.
        small_indices = node_indices[1, mortar]

        i_small_start, i_small_step_i,
        i_small_step_j = index_to_start_step_3d(small_indices[1],
                                                index_range)
        j_small_start, j_small_step_i,
        j_small_step_j = index_to_start_step_3d(small_indices[2],
                                                index_range)
        k_small_start, k_small_step_i,
        k_small_step_j = index_to_start_step_3d(small_indices[3],
                                                index_range)

        for position in 1:4
            i_small = i_small_start
            j_small = j_small_start
            k_small = k_small_start
            element = neighbor_ids[position, mortar]
            for j in 1:N
                for i in 1:N
                    for v in Base.OneTo(NVARS)
                        mortars_u[1, v, position, i, j, mortar] = u[v, i_small,
                                                                    j_small,
                                                                    k_small,
                                                                    element]
                    end
                    i_small += i_small_step_i
                    j_small += j_small_step_i
                    k_small += k_small_step_i
                end
                i_small += i_small_step_j
                j_small += j_small_step_j
                k_small += k_small_step_j
            end
        end

        # Buffer to copy solution values of the large element in the correct orientation
        # before interpolating
        u_buffer = MArray{Tuple{NVARS, N, N}, T, 3, L}(undef)
        # temporary buffer for projections
        fstar_tmp = MArray{Tuple{NVARS, N, N}, T, 3, L}(undef)

        #buffer for output
        val_out = MArray{Tuple{NVARS, N, N}, T, 3, L}(undef)

        # Copy solution of large element face to buffer in the
        # correct orientation
        large_indices = node_indices[2, mortar]

        i_large_start, i_large_step_i,
        i_large_step_j = index_to_start_step_3d(large_indices[1],
                                                index_range)
        j_large_start, j_large_step_i,
        j_large_step_j = index_to_start_step_3d(large_indices[2],
                                                index_range)
        k_large_start, k_large_step_i,
        k_large_step_j = index_to_start_step_3d(large_indices[3],
                                                index_range)

        i_large = i_large_start
        j_large = j_large_start
        k_large = k_large_start
        element = neighbor_ids[5, mortar]
        for j in 1:N
            for i in 1:N
                for v in Base.OneTo(NVARS)
                    u_buffer[v, i, j] = u[v, i_large, j_large, k_large, element]
                end
                i_large += i_large_step_i
                j_large += j_large_step_i
                k_large += k_large_step_i
            end
            i_large += i_large_step_j
            j_large += j_large_step_j
            k_large += k_large_step_j
        end

        # Interpolate large element face data from buffer to small face locations
        multiply_dimensionwise!(val_out,
                                forward_lower,
                                forward_lower,
                                u_buffer,
                                fstar_tmp)

        for j in 1:N, i in 1:N, v in Base.OneTo(NVARS)
            mortars_u[2, v, 1, i, j, mortar] = val_out[v, i, j]
        end

        multiply_dimensionwise!(val_out,
                                forward_upper,
                                forward_lower,
                                u_buffer,
                                fstar_tmp)

        for j in 1:N, i in 1:N, v in Base.OneTo(NVARS)
            mortars_u[2, v, 2, i, j, mortar] = val_out[v, i, j]
        end

        multiply_dimensionwise!(val_out,
                                forward_lower,
                                forward_upper,
                                u_buffer,
                                fstar_tmp)
        for j in 1:N, i in 1:N, v in Base.OneTo(NVARS)
            mortars_u[2, v, 3, i, j, mortar] = val_out[v, i, j]
        end

        multiply_dimensionwise!(val_out,
                                forward_upper,
                                forward_upper,
                                u_buffer,
                                fstar_tmp)

        for j in 1:N, i in 1:N, v in Base.OneTo(NVARS)
            mortars_u[2, v, 4, i, j, mortar] = val_out[v, i, j]
        end
    end #@inbounds
    return nothing
end

function calc_mortar_flux!(backend::KernelAbstractions.Backend, surface_flux_values,
                           mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                           have_nonconservative_terms, equations,
                           mortar_l2::LobattoLegendreMortarL2,
                           surface_integral, dg::DGSEM, cache)
    nmortars(dg, cache) == 0 && return nothing

    @unpack neighbor_ids, node_indices = cache.mortars
    @unpack contravariant_vectors = cache.elements
    mortars_u = cache.mortars.u
    pure_surface_flux = surface_integral.surface_flux
    index_range = eachnode(dg)

    N = nnodes(dg)
    NVARS = nvariables(equations)
    T = eltype(surface_flux_values)
    L = N * N * NVARS

    kernel! = calc_mortar_flux_KAkernel!(backend)

    kernel!(surface_flux_values, typeof(mesh), have_nonconservative_terms,
            equations, pure_surface_flux, dg,
            mortars_u, neighbor_ids, node_indices, contravariant_vectors,
            mortar_l2.reverse_lower, mortar_l2.reverse_upper, index_range,
            Val(N), Val(NVARS), Val(T), Val(L);
            ndrange = nmortars(dg, cache))

    return nothing
end

@kernel function calc_mortar_flux_KAkernel!(surface_flux_values,
                                            MeshT::Type{<:Union{P4estMesh{3},
                                                                T8codeMesh{3}}},
                                            have_nonconservative_terms, equations,
                                            pure_surface_flux, dg::DGSEM,
                                            mortars_u, neighbor_ids, node_indices,
                                            contravariant_vectors,
                                            reverse_lower, reverse_upper, index_range,
                                            ::Val{N}, ::Val{NVARS}, ::Val{T},
                                            ::Val{L}) where {N, NVARS, T, L}
    mortar = @index(Global, Linear)

    @inbounds begin
        fstar_primary1 = MArray{Tuple{NVARS, N, N}, T, 3, L}(undef)
        fstar_primary2 = MArray{Tuple{NVARS, N, N}, T, 3, L}(undef)
        fstar_primary3 = MArray{Tuple{NVARS, N, N}, T, 3, L}(undef)
        fstar_primary4 = MArray{Tuple{NVARS, N, N}, T, 3, L}(undef)

        fstar_secondary1 = MArray{Tuple{NVARS, N, N}, T, 3, L}(undef)
        fstar_secondary2 = MArray{Tuple{NVARS, N, N}, T, 3, L}(undef)
        fstar_secondary3 = MArray{Tuple{NVARS, N, N}, T, 3, L}(undef)
        fstar_secondary4 = MArray{Tuple{NVARS, N, N}, T, 3, L}(undef)

        fstar_tmp = MArray{Tuple{NVARS, N, N}, T, 3, L}(undef)

        small_indices = node_indices[1, mortar]
        small_direction = indices2direction(small_indices)

        i_small_start, i_small_step_i,
        i_small_step_j = index_to_start_step_3d(small_indices[1],
                                                index_range)
        j_small_start, j_small_step_i,
        j_small_step_j = index_to_start_step_3d(small_indices[2],
                                                index_range)
        k_small_start, k_small_step_i,
        k_small_step_j = index_to_start_step_3d(small_indices[3],
                                                index_range)

        KernelAbstractions.Extras.@unroll for position in 1:4
            i_small = i_small_start
            j_small = j_small_start
            k_small = k_small_start
            element = neighbor_ids[position, mortar]
            for j in 1:N
                for i in 1:N
                    normal_direction = get_normal_direction(small_direction,
                                                            contravariant_vectors,
                                                            i_small, j_small, k_small,
                                                            element)

                    gpu_calc_mortar_flux!(fstar_primary1, fstar_primary2,
                                          fstar_primary3, fstar_primary4,
                                          fstar_secondary1, fstar_secondary2,
                                          fstar_secondary3, fstar_secondary4,
                                          MeshT, have_nonconservative_terms, equations,
                                          pure_surface_flux, dg, mortars_u, mortar,
                                          position, normal_direction, i, j)

                    i_small += i_small_step_i
                    j_small += j_small_step_i
                    k_small += k_small_step_i
                end
                i_small += i_small_step_j
                j_small += j_small_step_j
                k_small += k_small_step_j
            end
        end

        u_buffer = MArray{Tuple{NVARS, N, N}, T, 3, L}(undef)

        mortar_fluxes_to_elements!(surface_flux_values, neighbor_ids, node_indices,
                                   index_range,
                                   reverse_lower, reverse_upper, mortar,
                                   fstar_primary1, fstar_primary2, fstar_primary3,
                                   fstar_primary4,
                                   fstar_secondary1, fstar_secondary2, fstar_secondary3,
                                   fstar_secondary4,
                                   u_buffer, fstar_tmp, N, NVARS)
    end
end

@inline function gpu_calc_mortar_flux!(fstar_primary1, fstar_primary2, fstar_primary3,
                                       fstar_primary4,
                                       fstar_secondary1, fstar_secondary2,
                                       fstar_secondary3, fstar_secondary4,
                                       MeshT,
                                       have_nonconservative_terms::False, equations,
                                       pure_surface_flux, dg::DGSEM, mortar_u,
                                       mortar_index, position_index, normal_direction,
                                       i_node_index, j_node_index)
    u_ll, u_rr = get_surface_node_vars(mortar_u, equations, dg, position_index,
                                       i_node_index, j_node_index, mortar_index)

    flux = pure_surface_flux(u_ll, u_rr, normal_direction, equations)

    # Copy flux to buffer
    if (position_index == 1)
        set_node_vars!(fstar_primary1, flux, equations, dg,
                       i_node_index, j_node_index)
        set_node_vars!(fstar_secondary1, flux, equations, dg,
                       i_node_index, j_node_index)
    elseif (position_index == 2)
        set_node_vars!(fstar_primary2, flux, equations, dg,
                       i_node_index, j_node_index)
        set_node_vars!(fstar_secondary2, flux, equations, dg,
                       i_node_index, j_node_index)
    elseif (position_index == 3)
        set_node_vars!(fstar_primary3, flux, equations, dg,
                       i_node_index, j_node_index)
        set_node_vars!(fstar_secondary3, flux, equations, dg,
                       i_node_index, j_node_index)
    elseif (position_index == 4)
        set_node_vars!(fstar_primary4, flux, equations, dg,
                       i_node_index, j_node_index)
        set_node_vars!(fstar_secondary4, flux, equations, dg,
                       i_node_index, j_node_index)
    end
    return nothing
end

@inline function gpu_calc_mortar_flux!(fstar_primary, fstar_secondary,
                                       MeshT,
                                       have_nonconservative_terms::True, equations,
                                       pure_surface_flux, dg::DGSEM, mortar_u,
                                       mortar_index, position_index, normal_direction,
                                       i_node_index, j_node_index)
    surface_flux, nonconservative_flux = pure_surface_flux

    u_ll, u_rr = get_surface_node_vars(mortar_u, equations, dg, position_index,
                                       i_node_index,
                                       j_node_index, mortar_index)

    # Compute conservative flux
    flux = surface_flux(u_ll, u_rr, normal_direction, equations)

    # Compute nonconservative flux and add it to the flux scaled by a factor of 0.5 based on
    # the interpretation of global SBP operators coupled discontinuously via
    # central fluxes/SATs
    noncons_primary = nonconservative_flux(u_ll, u_rr, normal_direction, equations)
    noncons_secondary = nonconservative_flux(u_rr, u_ll, normal_direction, equations)
    flux_plus_noncons_primary = flux + 0.5f0 * noncons_primary
    flux_plus_noncons_secondary = flux + 0.5f0 * noncons_secondary

    # Copy to buffer
    if (position_index == 1)
        set_node_vars!(fstar_primary, flux_plus_noncons_primary, equations, dg,
                       i_node_index, j_node_index)
        set_node_vars!(fstar_secondary, flux_plus_noncons_secondary, equations, dg,
                       i_node_index, j_node_index)
    elseif (position_index == 2)
        set_node_vars!(fstar_primary, flux_plus_noncons_primary, equations, dg,
                       i_node_index, j_node_index)
        set_node_vars!(fstar_secondary, flux_plus_noncons_secondary, equations, dg,
                       i_node_index, j_node_index)
    elseif (position_index == 3)
        set_node_vars!(fstar_primary, flux_plus_noncons_primary, equations, dg,
                       i_node_index, j_node_index)
        set_node_vars!(fstar_secondary, flux_plus_noncons_secondary, equations, dg,
                       i_node_index, j_node_index)
    elseif (position_index == 4)
        set_node_vars!(fstar_primary, flux_plus_noncons_primary, equations, dg,
                       i_node_index, j_node_index)
        set_node_vars!(fstar_secondary, flux_plus_noncons_secondary, equations, dg,
                       i_node_index, j_node_index)
    end
    return nothing
end

@inline function mortar_fluxes_to_elements!(surface_flux_values,
                                            neighbor_ids, node_indices, index_range,
                                            reverse_lower, reverse_upper,
                                            mortar,
                                            fstar_p_1, fstar_p_2, fstar_p_3, fstar_p_4,
                                            fstar_s_1, fstar_s_2, fstar_s_3, fstar_s_4,
                                            u_buffer, fstar_tmp, N, NVARS)

    # Copy solution small to small
    small_indices = node_indices[1, mortar]
    small_direction = indices2direction(small_indices)

    for position in 1:4
        element = neighbor_ids[position, mortar]
        for j in 1:N, i in 1:N
            for v in Base.OneTo(NVARS)
                if (position == 1)
                    surface_flux_values[v, i, j, small_direction,
                                        element] = fstar_p_1[v,
                                                             i,
                                                             j]
                elseif (position == 2)
                    surface_flux_values[v, i, j, small_direction,
                                        element] = fstar_p_2[v,
                                                             i,
                                                             j]
                elseif (position == 3)
                    surface_flux_values[v, i, j, small_direction,
                                        element] = fstar_p_3[v,
                                                             i,
                                                             j]
                elseif (position == 4)
                    surface_flux_values[v, i, j, small_direction,
                                        element] = fstar_p_4[v,
                                                             i,
                                                             j]
                end
            end
        end
    end

    # Project small fluxes to large element.
    multiply_dimensionwise!(u_buffer,
                            reverse_lower, reverse_lower,
                            fstar_s_1,
                            fstar_tmp)
    add_multiply_dimensionwise!(u_buffer,
                                reverse_upper, reverse_lower,
                                fstar_s_2,
                                fstar_tmp)
    add_multiply_dimensionwise!(u_buffer,
                                reverse_lower, reverse_upper,
                                fstar_s_3,
                                fstar_tmp)
    add_multiply_dimensionwise!(u_buffer,
                                reverse_upper, reverse_upper,
                                fstar_s_4,
                                fstar_tmp)

    # The flux is calculated in the outward direction of the small elements,
    # so the sign must be switched to get the flux in outward direction
    # of the large element.
    # The contravariant vectors of the large element (and therefore the normal
    # vectors of the large element as well) are four times as large as the
    # contravariant vectors of the small elements. Therefore, the flux needs
    # to be scaled by a factor of 4 to obtain the flux of the large element.
    u_buffer .*= -4

    # Copy interpolated flux values from buffer to large element face in the
    # correct orientation.
    # Note that the index of the small sides will always run forward but
    # the index of the large side might need to run backwards for flipped sides.
    large_element = neighbor_ids[5, mortar]
    large_indices = node_indices[2, mortar]
    large_direction = indices2direction(large_indices)
    large_surface_indices = surface_indices(large_indices)

    i_large_start, i_large_step_i,
    i_large_step_j = index_to_start_step_3d(large_surface_indices[1],
                                            index_range)
    j_large_start, j_large_step_i,
    j_large_step_j = index_to_start_step_3d(large_surface_indices[2],
                                            index_range)

    # Note that the indices of the small sides will always run forward but
    # the large indices might need to run backwards for flipped sides.
    i_large = i_large_start
    j_large = j_large_start
    for j in 1:N
        for i in 1:N
            for v in Base.OneTo(NVARS)
                surface_flux_values[v, i_large, j_large, large_direction,
                                    large_element] = u_buffer[v,
                                                              i,
                                                              j]
            end
            i_large += i_large_step_i
            j_large += j_large_step_i
        end
        i_large += i_large_step_j
        j_large += j_large_step_j
    end

    return nothing
end
end #muladd
