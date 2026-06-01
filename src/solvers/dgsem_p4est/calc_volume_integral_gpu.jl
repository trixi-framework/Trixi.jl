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
    kernel! = flux_differencing_KAkernel!(backend)
    _nnodes = nnodes(dg)
    kernel!(du, u, equations,
            typeof(mesh),
            have_nonconservative_terms,
            combine_conservative_and_nonconservative_fluxes(volume_integral.volume_flux,
                                                            equations),
            dg,
            volume_integral, Val(_nnodes),
            derivative_split,
            contravariant_vectors,
            ndrange = (_nnodes, _nnodes, _nnodes, nelements(dg, cache)))
    return nothing
end

@kernel function flux_differencing_KAkernel!(du, u, equations,
                                             MeshT::Type{<:P4estMesh{3}},
                                             have_nonconservative_terms::False,
                                             combine_conservative_and_nonconservative_fluxes::False,
                                             dg::DGSEM,
                                             volume_integral,
                                             ::Val{_nnodes},
                                             derivative_split,
                                             contravariant_vectors,
                                             alpha = true) where {_nnodes}
    # `true * [some floating point value] == [exactly the same floating point value]`
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
    #
    # Instead of assigning thread `i` the partners `i+1, …, _nnodes` (which
    # leaves low-index threads with much more work than high-index ones and
    # causes load imbalance within a warp), we distribute the half-sweep
    # cyclically: each thread visits `half = div(_nnodes,2)` partners at a fixed
    # rotating offset. Every unordered pair `{i, m}` is still covered exactly
    # once, but now every thread performs the same number of loop iterations.
    # When `_nnodes` is even (odd polynomial degree) the antipodal pair at
    # offset `half` is shared by two threads, so its contribution is weighted by
    # `1/2` to avoid double counting.
    #
    # See Section 4.1 (Eq. 6) of
    # - Waterhouse, Waruszewski, Wilcox, Warburton, Giraldo (2026)
    #   GPU Performance of an Entropy-Stable Discontinuous Galerkin Euler Solver
    #   with Non-Conservative Terms.
    #   arXiv (pre-print): https://arxiv.org/abs/2605.16684

    half_nnodes = div(_nnodes, 2)
    even_nodes = iseven(_nnodes)

    KernelAbstractions.Extras.@unroll for offset in 1:half_nnodes
        # weight the antipodal pair by 1/2 only when `_nnodes` is even
        weight = (even_nodes && offset == half_nnodes) ? 0.5f0 : 1.0f0

        # first coordinate direction: rotate the partner index along `i`
        ii = mod(i - 1 + offset, _nnodes) + 1
        u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
        # pull the contravariant vectors and compute the average
        Ja1_node_ii = get_contravariant_vector(1, contravariant_vectors,
                                               ii, j, k, element)
        Ja1_avg = 0.5f0 * (Ja1_node + Ja1_node_ii)
        # compute the contravariant sharp flux in the direction of the
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
        jj = mod(j - 1 + offset, _nnodes) + 1
        u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
        # pull the contravariant vectors and compute the average
        Ja2_node_jj = get_contravariant_vector(2, contravariant_vectors,
                                               i, jj, k, element)
        Ja2_avg = 0.5f0 * (Ja2_node + Ja2_node_jj)
        # compute the contravariant sharp flux in the direction of the
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
        kk = mod(k - 1 + offset, _nnodes) + 1
        u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
        # pull the contravariant vectors and compute the average
        Ja3_node_kk = get_contravariant_vector(3, contravariant_vectors,
                                               i, j, kk, element)
        Ja3_avg = 0.5f0 * (Ja3_node + Ja3_node_kk)
        # compute the contravariant sharp flux in the direction of the
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

@kernel function volume_integral_KAkernel!(du, u, MeshT,
                                           have_nonconservative_terms, equations,
                                           volume_integral, dg::DGSEM, cache)
    element = @index(Global)
    volume_integral_kernel!(du, u, element, MeshT, have_nonconservative_terms,
                            equations, volume_integral, dg, cache)
end
end #muladd
