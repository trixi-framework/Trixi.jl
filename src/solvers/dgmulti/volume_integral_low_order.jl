# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Algebraic low order method on the nodes of diagonal-norm SBP operators (`DGMultiSBP`,
# e.g., traditional SBP operators with LGL-type nodes). It is used as the low order part
# of shock capturing (`VolumeIntegralShockCapturingHG`) and as the entropy stable volume
# integral of entropy correction (`VolumeIntegralAdaptive` with
# `IndicatorEntropyCorrection`).
#
# Unlike the `GaussSBP` version in `shock_capturing.jl`, the solution lives at nodes that
# include the face nodes (at positions `rd.Fmask`), so no entropy projection is needed.
# Only the skew-symmetric parts of the sparse low order operators enter the volume
# integral; the boundary terms are accounted for by the surface integral, as for flux
# differencing with `Qrst_skew`.

# Creates the sparse low order SBP operators and their joint sparsity pattern. The kernel
# below additionally reads `cache.dxidxhatj`, which every `DGMulti` cache provides.
function create_cache(mesh::DGMultiMesh{NDIMS}, equations,
                      volume_integral::Union{VolumeIntegralShockCapturingHGType,
                                             VolumeIntegralPureLGLFiniteVolume},
                      dg::DGMultiSBP, RealT, uEltype) where {NDIMS}
    element_to_element_connectivity = build_element_to_element_connectivity(mesh, dg)

    # create skew-symmetric parts of sparse hybridized operators for low order scheme.
    sparse_SBP_operators, _ = StartUpDG.sparse_low_order_SBP_operators(dg.basis)
    sparse_SBP_operators = map(A -> 0.5f0 * (A - A'), sparse_SBP_operators)

    # Find the joint sparsity pattern of the entire matrix. We store the sparsity pattern as
    # an adjoint for faster iteration through the rows.
    sparsity_pattern = sum(map(A -> abs.(A)', sparse_SBP_operators)) .> 100 * eps()

    # inverse quadrature weights, i.e., the inverse of the diagonal mass matrix
    inv_wq = inv.(dg.basis.wq)

    return (; sparse_SBP_operators, sparsity_pattern,
            element_to_element_connectivity, inv_wq)
end

# Calculates the volume integral corresponding to the algebraic low order method.
function volume_integral_kernel!(du, u, element, mesh::DGMultiMesh,
                                 have_nonconservative_terms::False, equations,
                                 volume_integral::VolumeIntegralPureLGLFiniteVolume,
                                 dg::DGMultiSBP, cache, alpha = true)
    (; volume_flux_fv) = volume_integral

    (; inv_wq, sparsity_pattern) = cache
    A_base, row_ids, rows, _ = sparse_operator_data(sparsity_pattern)
    for i in row_ids
        u_i = u[i, element]
        du_i = zero(u_i)
        for id in nzrange(A_base, i)
            # nonzero column indices for row i of the sparse operator.
            # note that because Julia uses SparseMatrixCSC, rows[id]
            # are efficient to access. We assume here that `sparsity_pattern`
            # is symmetric (which is true since A_base is skew-symmetric),
            # so nonzero row indices are the same as nonzero column indices.
            j = rows[id]
            u_j = u[j, element]

            # compute (Q_1[i,j], Q_2[i,j], ...) where Q_i = ∑_j dxidxhatj * Q̂_j
            geometric_matrix = get_low_order_geometric_matrix(i, j, element, mesh,
                                                              cache)
            reference_operator_entries = get_sparse_operator_entries(i, j, mesh,
                                                                     cache)
            normal_direction_ij = geometric_matrix * reference_operator_entries

            # note that we do not need to normalize `normal_direction_ij` since
            # it is typically normalized within the flux computation.
            f_ij = volume_flux_fv(u_i, u_j, normal_direction_ij, equations)

            # the factor of 2 is for consistency; for example, if f_ij is the central
            # flux, flux differencing with a differentiation matrix should recover the
            # flux derivative via
            #   \sum_j 2 * D_ij * f_ij = \sum_j 2 * D_ij * 0.5 * (f(u_i) + f(u_j))
            #                          = f(u_i) \sum_j D_ij + \sum_j D_ij f(u_j)
            #                          = 0 (since \sum_j D_ij = 0) + (D * f(u))_i
            du_i = du_i + 2 * f_ij
        end
        du[i, element] = du[i, element] + alpha * du_i * inv_wq[i]
    end

    return nothing
end
end # @muladd
