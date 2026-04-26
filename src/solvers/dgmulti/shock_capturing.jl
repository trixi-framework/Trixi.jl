# Since `@muladd` can fuse multiply-add operations and thus improve performance in
# the flux differencing loops, we opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# by default, return an empty tuple for volume integral caches
function create_cache(mesh::DGMultiMesh{NDIMS}, equations,
                      volume_integral::VolumeIntegralShockCapturingHGType,
                      dg::DGMultiFluxDiff{<:GaussSBP}, RealT, uEltype) where {NDIMS}
    (; volume_integral_default, volume_integral_blend_high_order) = volume_integral
    @assert volume_integral_default isa VolumeIntegralFluxDifferencing "DGMulti is currently only compatible with `VolumeIntegralFluxDifferencing` as `volume_integral_default`"
    @assert volume_integral_blend_high_order isa VolumeIntegralFluxDifferencing "DGMulti is currently only compatible with `VolumeIntegralFluxDifferencing` as `volume_integral_blend_high_order`"
    # `volume_integral_blend_low_order` limited to finite-volume on Gauss-node subcells

    element_to_element_connectivity = build_element_to_element_connectivity(mesh, dg)

    # create sparse hybridized operators for low order scheme
    Qrst, E = StartUpDG.sparse_low_order_SBP_operators(dg.basis)
    Brst = map(n -> Diagonal(n .* dg.basis.wf), dg.basis.nrstJ)
    sparse_SBP_operators = map((Q, B) -> 0.5f0 * [Q-Q' E'*B; -B*E zeros(size(B))],
                               Qrst, Brst)

    # Find the joint sparsity pattern of the entire matrix. We store the sparsity pattern as
    # an adjoint for faster iteration through the rows.
    sparsity_pattern = sum(map(A -> abs.(A)', sparse_SBP_operators)) .>
                       100 * eps()

    return (; sparse_SBP_operators, sparsity_pattern,
            element_to_element_connectivity)
end

function create_cache(mesh::DGMultiMesh{NDIMS}, equations,
                      volume_integral::Union{VolumeIntegralShockCapturingHGType,
                                             VolumeIntegralPureLGLFiniteVolume},
                      dg::DGMultiFluxDiffSBP, RealT, uEltype) where {NDIMS}
    element_to_element_connectivity = build_element_to_element_connectivity(mesh, dg)

    # create skew-symmetric parts of sparse hybridized operators for low order scheme.
    sparse_SBP_operators, _ = StartUpDG.sparse_low_order_SBP_operators(dg.basis)
    sparse_SBP_operators = map(A -> 0.5f0 * (A - A'), sparse_SBP_operators)

    # Find the joint sparsity pattern of the entire matrix. We store the sparsity pattern as
    # an adjoint for faster iteration through the rows.
    sparsity_pattern = sum(map(A -> abs.(A)', sparse_SBP_operators)) .> 100 * eps()

    return (; sparse_SBP_operators, sparsity_pattern,
            element_to_element_connectivity)
end

# this method is used when the indicator is constructed as for shock-capturing volume integrals
function create_cache(::Type{IndicatorHennemannGassner}, equations::AbstractEquations,
                      basis::DGMultiBasis{NDIMS}) where {NDIMS}
    uEltype = real(basis)
    alpha = Vector{uEltype}()
    alpha_tmp = similar(alpha)

    MVec_nodes = MVector{nnodes(basis), uEltype}
    indicator_threaded = MVec_nodes[MVec_nodes(undef) for _ in 1:Threads.maxthreadid()]
    MVec_modes = MVector{nmodes(basis.N, basis.element_type), uEltype}
    modal_threaded = MVec_modes[MVec_modes(undef) for _ in 1:Threads.maxthreadid()]

    inverse_vandermonde = calc_inverse_vandermonde(basis)

    return (; alpha, alpha_tmp, indicator_threaded, modal_threaded, inverse_vandermonde)
end

# calculates the inverse of the Vandermonde matrix for shock capturing purposes.
# This version is for tensor product elements (Line, Quad, Hex)
function calc_inverse_vandermonde(basis::DGMultiBasis{NDIMS, <:Union{Line, Quad, Hex}}) where {NDIMS}
    # initialize inverse Vandermonde matrices at Gauss-Legendre nodes
    (; N) = basis
    lobatto_node_coordinates_1D, _ = StartUpDG.gauss_lobatto_quad(0, 0, N)
    VDM_1D = StartUpDG.vandermonde(Line(), N, lobatto_node_coordinates_1D)
    inverse_vandermonde = SimpleKronecker(NDIMS, inv(VDM_1D))

    return inverse_vandermonde
end

function (indicator_hg::IndicatorHennemannGassner)(u, mesh::DGMultiMesh,
                                                   equations,
                                                   dg::DGMulti{NDIMS,
                                                               <:Union{Line, Quad, Hex}},
                                                   cache;
                                                   kwargs...) where {NDIMS}
    (; alpha_max, alpha_min, alpha_smooth, variable) = indicator_hg
    (; alpha, alpha_tmp, indicator_threaded, modal_threaded, inverse_vandermonde) = indicator_hg.cache

    resize!(alpha, nelements(mesh, dg))
    if alpha_smooth
        resize!(alpha_tmp, nelements(mesh, dg))
    end

    # magic parameters
    threshold = 0.5f0 * 10^(-1.8 * (dg.basis.N + 1)^0.25)
    parameter_s = log((1 - 0.0001) / 0.0001)

    @threaded for element in eachelement(mesh, dg)
        indicator = indicator_threaded[Threads.threadid()]
        modal_ = modal_threaded[Threads.threadid()]

        # Calculate indicator variable at interpolation (Lobatto) nodes.
        # TODO: calculate indicator variables at Gauss nodes or using `cache.entropy_projected_u_values`
        for i in eachnode(dg)
            indicator[i] = variable(u[i, element], equations)
        end

        # multiply by invVDM::SimpleKronecker
        LinearAlgebra.mul!(modal_, inverse_vandermonde, indicator)

        # Create Returns functors to return the constructor args (e.g., Base.OneTo(dg.basis.N)) no matter what
        # Returns(Base.OneTo(dg.basis.N)) equiv to _ -> Base.OneTo(dg.basis.N), with possibly fewer allocs
        return_N_plus_one = Returns(dg.basis.N + 1)
        return_to_N_minus_one = Returns(Base.OneTo(dg.basis.N - 1))
        return_to_N = Returns(Base.OneTo(dg.basis.N))

        # As of Julia 1.9, Base.ReshapedArray does not produce allocations when setting values.
        # Thus, Base.ReshapedArray should be used if you are setting values in the array.
        # `reshape` is fine if you are only accessing values.
        # Here, we reshape modal coefficients to expose the tensor product structure.

        modal = Base.ReshapedArray(modal_, ntuple(return_N_plus_one, NDIMS), ())

        # Calculate total energies for all modes, all modes minus the highest mode, and
        # all modes without the two highest modes
        total_energy = sum(abs2, modal)
        clip_1_ranges = ntuple(return_to_N, NDIMS)
        clip_2_ranges = ntuple(return_to_N_minus_one, NDIMS)
        # These splattings do not seem to allocate as of Julia 1.9.0?
        total_energy_clip1 = sum(abs2, view(modal, clip_1_ranges...))
        total_energy_clip2 = sum(abs2, view(modal, clip_2_ranges...))

        # Calculate energy in higher modes
        if !(iszero(total_energy))
            energy_frac_1 = (total_energy - total_energy_clip1) / total_energy
        else
            energy_frac_1 = zero(total_energy)
        end
        if !(iszero(total_energy_clip1))
            energy_frac_2 = (total_energy_clip1 - total_energy_clip2) /
                            total_energy_clip1
        else
            energy_frac_2 = zero(total_energy_clip1)
        end
        energy = max(energy_frac_1, energy_frac_2)

        alpha_element = 1 / (1 + exp(-parameter_s / threshold * (energy - threshold)))

        # Take care of the case close to pure DG
        if alpha_element < alpha_min
            alpha_element = zero(alpha_element)
        end

        # Take care of the case close to pure FV
        if alpha_element > 1 - alpha_min
            alpha_element = one(alpha_element)
        end

        # Clip the maximum amount of FV allowed
        alpha[element] = min(alpha_max, alpha_element)
    end

    # smooth element indices after they're all computed
    if alpha_smooth
        apply_smoothing!(mesh, alpha, alpha_tmp, dg, cache)
    end

    return alpha
end

# Diffuse alpha values by setting each alpha to at least 50% of neighboring elements' alpha
function apply_smoothing!(mesh::DGMultiMesh, alpha, alpha_tmp, dg::DGMulti, cache)

    # Copy alpha values such that smoothing is independent of the element access order
    alpha_tmp .= alpha

    # smooth alpha with its neighboring value
    @threaded for element in eachelement(mesh, dg)
        for face in Base.OneTo(StartUpDG.num_faces(dg.basis.element_type))
            neighboring_element = cache.element_to_element_connectivity[face, element]
            alpha_neighbor = alpha_tmp[neighboring_element]
            alpha[element] = max(alpha[element], 0.5f0 * alpha_neighbor)
        end
    end

    return nothing
end

function calc_volume_integral!(du, u, mesh::DGMultiMesh,
                               have_nonconservative_terms, equations,
                               volume_integral::VolumeIntegralShockCapturingHGType,
                               dg::DGMultiFluxDiff, cache)
    (; indicator, volume_integral_default,
    volume_integral_blend_high_order, volume_integral_blend_low_order) = volume_integral

    # Calculate blending factors α: u = u_DG * (1 - α) + u_FV * α
    alpha = @trixi_timeit timer() "blending factors" indicator(u, mesh, equations, dg,
                                                               cache)

    # For `Float64`, this gives 1.8189894035458565e-12
    # For `Float32`, this gives 1.1920929f-5
    RealT = eltype(alpha)
    atol = max(100 * eps(RealT), eps(RealT)^convert(RealT, 0.75f0))

    @threaded for element in eachelement(mesh, dg)
        alpha_element = alpha[element]
        # Clip blending factor for values close to zero (-> pure DG)
        dg_only = isapprox(alpha_element, 0, atol = atol)

        if dg_only
            volume_integral_kernel!(du, u, element, mesh,
                                    have_nonconservative_terms, equations,
                                    volume_integral_default,
                                    dg, cache)
        else
            # Calculate DG volume integral contribution
            volume_integral_kernel!(du, u, element, mesh,
                                    have_nonconservative_terms, equations,
                                    volume_integral_blend_high_order,
                                    dg, cache, 1 - alpha_element)

            # Calculate "FV" low order volume integral contribution
            volume_integral_kernel!(du, u, element, mesh,
                                    have_nonconservative_terms, equations,
                                    volume_integral_blend_low_order,
                                    dg, cache, alpha_element)
        end
    end

    return nothing
end

function get_sparse_operator_entries(i, j, mesh::DGMultiMesh{1}, cache)
    return SVector(cache.sparse_SBP_operators[1][i, j])
end

function get_sparse_operator_entries(i, j, mesh::DGMultiMesh{2}, cache)
    Qr, Qs = cache.sparse_SBP_operators
    return SVector(Qr[i, j], Qs[i, j])
end

function get_sparse_operator_entries(i, j, mesh::DGMultiMesh{3}, cache)
    Qr, Qs, Qt = cache.sparse_SBP_operators
    return SVector(Qr[i, j], Qs[i, j], Qt[i, j])
end

function get_contravariant_matrix(element, mesh::DGMultiMesh{1}, cache)
    return SMatrix{1, 1}(cache.dxidxhatj[1, 1][1, element])
end

function get_contravariant_matrix(element, mesh::DGMultiMesh{2, <:Affine}, cache)
    (; dxidxhatj) = cache
    return SMatrix{2, 2}(dxidxhatj[1, 1][1, element], dxidxhatj[2, 1][1, element],
                         dxidxhatj[1, 2][1, element], dxidxhatj[2, 2][1, element])
end

function get_contravariant_matrix(element, mesh::DGMultiMesh{3, <:Affine}, cache)
    (; dxidxhatj) = cache
    return SMatrix{3, 3}(dxidxhatj[1, 1][1, element], dxidxhatj[2, 1][1, element],
                         dxidxhatj[3, 1][1, element],
                         dxidxhatj[1, 2][1, element], dxidxhatj[2, 2][1, element],
                         dxidxhatj[3, 2][1, element],
                         dxidxhatj[1, 3][1, element], dxidxhatj[2, 3][1, element],
                         dxidxhatj[3, 3][1, element])
end

function get_contravariant_matrix(i, element, mesh::DGMultiMesh{2}, cache)
    (; dxidxhatj) = cache
    return SMatrix{2, 2}(dxidxhatj[1, 1][i, element], dxidxhatj[2, 1][i, element],
                         dxidxhatj[1, 2][i, element], dxidxhatj[2, 2][i, element])
end

function get_contravariant_matrix(i, element, mesh::DGMultiMesh{3}, cache)
    (; dxidxhatj) = cache
    return SMatrix{3, 3}(dxidxhatj[1, 1][i, element], dxidxhatj[2, 1][i, element],
                         dxidxhatj[3, 1][i, element],
                         dxidxhatj[1, 2][i, element], dxidxhatj[2, 2][i, element],
                         dxidxhatj[3, 2][i, element],
                         dxidxhatj[1, 3][i, element], dxidxhatj[2, 3][i, element],
                         dxidxhatj[3, 3][i, element])
end

function get_avg_contravariant_matrix(i, j, element, mesh::DGMultiMesh, cache)
    return 0.5f0 * (get_contravariant_matrix(i, element, mesh, cache) +
            get_contravariant_matrix(j, element, mesh, cache))
end

# On affine meshes, the geometric matrix is constant over the element, so we compute it
# once and reuse it for all node pairs (i, j). The compiler is expected to hoist this
# out of the inner loop after inlining.
@inline function get_low_order_geometric_matrix(i, j, element,
                                                mesh::DGMultiMesh{NDIMS, <:Affine},
                                                cache) where {NDIMS}
    return get_contravariant_matrix(element, mesh, cache)
end

# On non-affine meshes, we use the average of the geometric matrices at nodes i and j
# for provably entropy-stable de-aliasing of the geometric terms.
@inline function get_low_order_geometric_matrix(i, j, element,
                                                mesh::DGMultiMesh,
                                                cache)
    return get_avg_contravariant_matrix(i, j, element, mesh, cache)
end

# Calculates the volume integral corresponding to an algebraic low order method.
# This is used, for example, in shock capturing.
function volume_integral_kernel!(du, u, element, mesh::DGMultiMesh,
                                 have_nonconservative_terms::False, equations,
                                 volume_integral::VolumeIntegralPureLGLFiniteVolume,
                                 dg::DGMultiFluxDiff{<:GaussSBP}, cache,
                                 alpha = true)
    (; volume_flux_fv) = volume_integral

    # accumulates output from flux differencing
    rhs_local = cache.rhs_local_threaded[Threads.threadid()]
    fill!(rhs_local, zero(eltype(rhs_local)))

    u_local = view(cache.entropy_projected_u_values, :, element)

    (; sparsity_pattern) = cache
    A_base, row_ids, rows, _ = sparse_operator_data(sparsity_pattern)
    for i in row_ids
        u_i = u_local[i]
        du_i = zero(u_i)
        for id in nzrange(A_base, i)
            # nonzero column indices for row i of the sparse operator. 
            # note that because Julia uses SparseMatrixCSC, rows[id] 
            # are efficient to access. We assume here that `sparsity_pattern`
            # is symmetric (which is true since A_base is skew-symmetric), 
            # so nonzero row indices are the same as nonzero column indices.
            j = rows[id]
            u_j = u_local[j]

            # compute (Q_1[i,j], Q_2[i,j], ...) where Q_i = ∑_j dxidxhatj * Q̂_j
            geometric_matrix = get_low_order_geometric_matrix(i, j, element,
                                                              mesh, cache)
            reference_operator_entries = get_sparse_operator_entries(i, j, mesh, cache)
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
        rhs_local[i] = du_i
    end

    # TODO: factor this out to avoid calling it twice during calc_volume_integral!
    return project_rhs_to_gauss_nodes!(du, rhs_local, element, mesh, dg, cache, alpha)
end

# Calculates the volume integral corresponding to an algebraic low order method for
# DGMultiFluxDiffSBP (traditional SBP operators with LGL-type nodes).
# Unlike GaussSBP, the solution lives at nodes that include the face nodes (at positions
# `rd.Fmask`), so no entropy projection is needed. We build the extended [interior; face]
# vector in-kernel and project back by scattering face contributions to Fmask positions.
function volume_integral_kernel!(du, u, element, mesh::DGMultiMesh,
                                 have_nonconservative_terms::False, equations,
                                 volume_integral::VolumeIntegralPureLGLFiniteVolume,
                                 dg::DGMultiFluxDiffSBP, cache, alpha = true)
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
            #                        = 0 (since \sum_j D_ij = 0) + (D * f(u))_i
            du_i = du_i + 2 * f_ij
        end
        du[i, element] = du[i, element] + alpha * du_i * inv_wq[i]
    end

    return nothing
end
end # @muladd
