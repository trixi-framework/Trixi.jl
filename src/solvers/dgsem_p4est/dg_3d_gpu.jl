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
            Ja1_avg = 0.5 * (Ja1_node + Ja1_node_ii)
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
            Ja2_avg = 0.5 * (Ja2_node + Ja2_node_jj)
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
            Ja3_avg = 0.5 * (Ja3_node + Ja3_node_kk)
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
