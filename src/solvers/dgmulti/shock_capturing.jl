# by default, return an empty tuple for volume integral caches
function create_cache(mesh::DGMultiMesh{NDIMS}, equations,
                      volume_integral::VolumeIntegralShockCapturingHG,
                      dg::DGMultiFluxDiff{<:GaussSBP}, RealT, uEltype) where {NDIMS}
    element_ids_dg = Int[]
    element_ids_dgfv = Int[]

    # build element to element (element_to_element_connectivity) connectivity for smoothing of
    # shock capturing parameters.
    face_to_face_connectivity = mesh.md.FToF # num_faces x num_elements matrix
    element_to_element_connectivity = similar(face_to_face_connectivity)
    for e in axes(face_to_face_connectivity, 2)
        for f in axes(face_to_face_connectivity, 1)
            neighbor_face_index = face_to_face_connectivity[f, e]

            # reverse-engineer element index from face. Assumes all elements
            # have the same number of faces.
            neighbor_element_index = ((neighbor_face_index - 1) ÷ dg.basis.num_faces) + 1
            element_to_element_connectivity[f, e] = neighbor_element_index
        end
    end

    # create sparse hybridized operators for low order scheme
    Qrst, E = StartUpDG.sparse_low_order_SBP_operators(dg.basis)
    Brst = map(n -> Diagonal(n .* dg.basis.wf), dg.basis.nrstJ)
    sparse_hybridized_SBP_operators = map((Q, B) -> 0.5 * [Q-Q' E'*B; -B*E zeros(size(B))],
                                          Qrst, Brst)

    # Find the joint sparsity pattern of the entire matrix. We store the sparsity pattern as
    # an adjoint for faster iteration through the rows.
    sparsity_pattern = sum(map(A -> abs.(A)', sparse_hybridized_SBP_operators)) .>
                       100 * eps()

    return (; element_ids_dg, element_ids_dgfv,
            sparse_hybridized_SBP_operators, sparsity_pattern,
            element_to_element_connectivity)
end

# this method is used when the indicator is constructed as for shock-capturing volume integrals
function create_cache(::Type{IndicatorHennemannGassner}, equations::AbstractEquations,
                      basis::RefElemData{NDIMS}) where {NDIMS}
    alpha = Vector{real(basis)}()
    alpha_tmp = similar(alpha)

    A = Vector{real(basis)}
    indicator_threaded = [A(undef, nnodes(basis)) for _ in 1:Threads.nthreads()]
    modal_threaded = [A(undef, nnodes(basis)) for _ in 1:Threads.nthreads()]

    # initialize inverse Vandermonde matrices at Gauss-Legendre nodes
    (; N) = basis
    lobatto_node_coordinates_1D, _ = StartUpDG.gauss_lobatto_quad(0, 0, N)
    VDM_1D = StartUpDG.vandermonde(Line(), N, lobatto_node_coordinates_1D)
    inverse_vandermonde = SimpleKronecker(NDIMS, inv(VDM_1D))

    return (; alpha, alpha_tmp, indicator_threaded, modal_threaded, inverse_vandermonde)
end

function (indicator_hg::IndicatorHennemannGassner)(u, mesh::DGMultiMesh,
                                                   equations, dg::DGMulti{NDIMS}, cache;
                                                   kwargs...) where {NDIMS}
    (; alpha_max, alpha_min, alpha_smooth, variable) = indicator_hg
    (; alpha, alpha_tmp, indicator_threaded, modal_threaded, inverse_vandermonde) = indicator_hg.cache

    resize!(alpha, nelements(mesh, dg))
    if alpha_smooth
        resize!(alpha_tmp, nelements(mesh, dg))
    end

    # magic parameters
    threshold = 0.5 * 10^(-1.8 * (dg.basis.N + 1)^0.25)
    parameter_s = log((1 - 0.0001) / 0.0001)

    @threaded for element in eachelement(mesh, dg)
        indicator = indicator_threaded[Threads.threadid()]
        modal_ = modal_threaded[Threads.threadid()]

        # Calculate indicator variable at interpolation (Lobatto) nodes.
        # TODO: calculate indicator variables at Gauss nodes or using `cache.entropy_projected_u_values`
        for i in eachnode(dg)
            indicator[i] = indicator_hg.variable(u[i, element], equations)
        end

        # multiply by invVDM::SimpleKronecker
        LinearAlgebra.mul!(modal_, inverse_vandermonde, indicator)

        # As of Julia 1.9, Base.ReshapedArray does not produce allocations when setting values.
        # Thus, Base.ReshapedArray should be used if you are setting values in the array.
        # `reshape` is fine if you are only accessing values.
        # Here, we reshape modal coefficients to expose the tensor product structure.
        modal = Base.ReshapedArray(modal_, ntuple(_ -> dg.basis.N + 1, NDIMS), ())

        # Calculate total energies for all modes, all modes minus the highest mode, and
        # all modes without the two highest modes
        total_energy = sum(x -> x^2, modal)
        clip_1_ranges = ntuple(_ -> Base.OneTo(dg.basis.N), NDIMS)
        clip_2_ranges = ntuple(_ -> Base.OneTo(dg.basis.N - 1), NDIMS)
        # These splattings do not seem to allocate as of Julia 1.9.0?
        total_energy_clip1 = sum(x -> x^2, view(modal, clip_1_ranges...))
        total_energy_clip2 = sum(x -> x^2, view(modal, clip_2_ranges...))

        # Calculate energy in higher modes
        if !(iszero(total_energy))
            energy_frac_1 = (total_energy - total_energy_clip1) / total_energy
        else
            energy_frac_1 = zero(total_energy)
        end
        if !(iszero(total_energy_clip1))
            energy_frac_2 = (total_energy_clip1 - total_energy_clip2) / total_energy_clip1
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

    # Copy alpha values such that smoothing is indpedenent of the element access order
    alpha_tmp .= alpha

    # smooth alpha with its neighboring value
    for element in eachelement(mesh, dg)
        for face in Base.OneTo(StartUpDG.num_faces(dg.basis.element_type))
            neighboring_element = cache.element_to_element_connectivity[face, element]
            alpha_neighbor = alpha_tmp[neighboring_element]
            alpha[element] = max(alpha[element], 0.5 * alpha_neighbor)
        end
    end
end

#     pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, dg, cache)
#
# Given blending factors `alpha` and the solver `dg`, fill
# `element_ids_dg` with the IDs of elements using a pure DG scheme and
# `element_ids_dgfv` with the IDs of elements using a blended DG-FV scheme.
function pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha,
                                       mesh::DGMultiMesh, dg::DGMulti)
    empty!(element_ids_dg)
    empty!(element_ids_dgfv)
    # For `Float64`, this gives 1.8189894035458565e-12
    # For `Float32`, this gives 1.1920929f-5
    RealT = eltype(alpha)
    atol = max(100 * eps(RealT), eps(RealT)^convert(RealT, 0.75f0))

    for element in eachelement(mesh, dg)
        # Clip blending factor for values close to zero (-> pure DG)
        dg_only = isapprox(alpha[element], 0, atol = atol)
        if dg_only
            push!(element_ids_dg, element)
        else
            push!(element_ids_dgfv, element)
        end
    end

    return nothing
end

function calc_volume_integral!(du, u,
                               mesh::DGMultiMesh,
                               have_nonconservative_terms, equations,
                               volume_integral::VolumeIntegralShockCapturingHG,
                               dg::DGMultiFluxDiff, cache)
    (; element_ids_dg, element_ids_dgfv) = cache
    (; volume_flux_dg, volume_flux_fv, indicator) = volume_integral

    # Calculate blending factors α: u = u_DG * (1 - α) + u_FV * α
    alpha = @trixi_timeit timer() "blending factors" indicator(u, mesh, equations, dg,
                                                               cache)

    # Determine element ids for DG-only and blended DG-FV volume integral
    pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, mesh, dg)

    # Loop over pure DG elements
    @trixi_timeit timer() "pure DG" @threaded for idx_element in eachindex(element_ids_dg)
        element = element_ids_dg[idx_element]
        flux_differencing_kernel!(du, u, element, mesh, have_nonconservative_terms,
                                  equations, volume_flux_dg, dg, cache)
    end

    # Loop over blended DG-FV elements, blend the high and low order RHS contributions
    # via `rhs_high * (1 - alpha) + rhs_low * (alpha)`.
    @trixi_timeit timer() "blended DG-FV" @threaded for idx_element in eachindex(element_ids_dgfv)
        element = element_ids_dgfv[idx_element]
        alpha_element = alpha[element]

        # Calculate DG volume integral contribution
        flux_differencing_kernel!(du, u, element, mesh,
                                  have_nonconservative_terms, equations,
                                  volume_flux_dg, dg, cache, 1 - alpha_element)

        # Calculate "FV" low order volume integral contribution
        low_order_flux_differencing_kernel!(du, u, element, mesh,
                                            have_nonconservative_terms, equations,
                                            volume_flux_fv, dg, cache, alpha_element)
    end

    return nothing
end

function get_sparse_operator_entries(i, j, mesh::DGMultiMesh{1}, cache)
    SVector(cache.sparse_hybridized_SBP_operators[1][i, j])
end

function get_sparse_operator_entries(i, j, mesh::DGMultiMesh{2}, cache)
    Qr, Qs = cache.sparse_hybridized_SBP_operators
    return SVector(Qr[i, j], Qs[i, j])
end

function get_sparse_operator_entries(i, j, mesh::DGMultiMesh{3}, cache)
    Qr, Qs, Qt = cache.sparse_hybridized_SBP_operators
    return SVector(Qr[i, j], Qs[i, j], Qt[i, j])
end

function get_contravariant_matrix(element, mesh::DGMultiMesh{1}, cache)
    SMatrix{1, 1}(cache.dxidxhatj[1, 1][1, element])
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
    0.5 * (get_contravariant_matrix(i, element, mesh, cache) +
     get_contravariant_matrix(j, element, mesh, cache))
end

# computes an algebraic low order method with internal dissipation.
# This method is for affine/Cartesian meshes
function low_order_flux_differencing_kernel!(du, u, element, mesh::DGMultiMesh,
                                             have_nonconservative_terms::False, equations,
                                             volume_flux_fv,
                                             dg::DGMultiFluxDiff{<:GaussSBP},
                                             cache, alpha = true)

    # accumulates output from flux differencing
    rhs_local = cache.rhs_local_threaded[Threads.threadid()]
    fill!(rhs_local, zero(eltype(rhs_local)))

    u_local = view(cache.entropy_projected_u_values, :, element)

    # constant over each element
    geometric_matrix = get_contravariant_matrix(element, mesh, cache)

    (; sparsity_pattern) = cache
    A_base = parent(sparsity_pattern) # the adjoint of a SparseMatrixCSC is basically a SparseMatrixCSR
    row_ids, rows = axes(sparsity_pattern, 2), rowvals(A_base)
    for i in row_ids
        u_i = u_local[i]
        du_i = zero(u_i)
        for id in nzrange(A_base, i)
            j = rows[id]
            u_j = u_local[j]

            # compute (Q_1[i,j], Q_2[i,j], ...) where Q_i = ∑_j dxidxhatj * Q̂_j
            reference_operator_entries = get_sparse_operator_entries(i, j, mesh, cache)
            normal_direction_ij = geometric_matrix * reference_operator_entries

            # note that we do not need to normalize `normal_direction_ij` since
            # it is typically normalized within the flux computation.
            f_ij = volume_flux_fv(u_i, u_j, normal_direction_ij, equations)
            du_i = du_i + 2 * f_ij
        end
        rhs_local[i] = du_i
    end

    # TODO: factor this out to avoid calling it twice during calc_volume_integral!
    project_rhs_to_gauss_nodes!(du, rhs_local, element, mesh, dg, cache, alpha)
end

function low_order_flux_differencing_kernel!(du, u, element,
                                             mesh::DGMultiMesh{NDIMS, <:NonAffine},
                                             have_nonconservative_terms::False, equations,
                                             volume_flux_fv,
                                             dg::DGMultiFluxDiff{<:GaussSBP},
                                             cache, alpha = true) where {NDIMS}

    # accumulates output from flux differencing
    rhs_local = cache.rhs_local_threaded[Threads.threadid()]
    fill!(rhs_local, zero(eltype(rhs_local)))

    u_local = view(cache.entropy_projected_u_values, :, element)

    (; sparsity_pattern) = cache
    A_base = parent(sparsity_pattern) # the adjoint of a SparseMatrixCSC is basically a SparseMatrixCSR
    row_ids, rows = axes(sparsity_pattern, 2), rowvals(A_base)
    for i in row_ids
        u_i = u_local[i]
        du_i = zero(u_i)
        for id in nzrange(A_base, i)
            j = rows[id]
            u_j = u_local[j]

            # compute (Q_1[i,j], Q_2[i,j], ...) where Q_i = ∑_j dxidxhatj * Q̂_j
            geometric_matrix = get_avg_contravariant_matrix(i, j, element, mesh, cache)
            reference_operator_entries = get_sparse_operator_entries(i, j, mesh, cache)
            normal_direction_ij = geometric_matrix * reference_operator_entries

            # note that we do not need to normalize `normal_direction_ij` since
            # it is typically normalized within the flux computation.
            f_ij = volume_flux_fv(u_i, u_j, normal_direction_ij, equations)
            du_i = du_i + 2 * f_ij
        end
        rhs_local[i] = du_i
    end

    # TODO: factor this out to avoid calling it twice during calc_volume_integral!
    project_rhs_to_gauss_nodes!(du, rhs_local, element, mesh, dg, cache, alpha)
end
