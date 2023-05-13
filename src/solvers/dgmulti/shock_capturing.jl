# by default, return an empty tuple for volume integral caches
function create_cache(mesh::DGMultiMesh{NDIMS}, equations,
                      volume_integral::VolumeIntegralShockCapturingHG,
                      dg::DGMultiFluxDiff{<:GaussSBP}, RealT, uEltype) where {NDIMS}
  element_ids_dg   = Int[]
  element_ids_dgfv = Int[]

  # build element to element (EToE) connectivity for smoothing of
  # shock capturing parameters.
  FToF = mesh.md.FToF # num_faces x num_elements matrix
  EToE = similar(FToF)
  for e in axes(FToF, 2)
    for f in axes(FToF, 1)
      neighbor_face_index = FToF[f, e]

      # reverse-engineer element index from face. Assumes all elements
      # have the same number of faces.
      neighbor_element_index = ((neighbor_face_index - 1) ÷ dg.basis.num_faces) + 1
      EToE[f, e] = neighbor_element_index
    end
  end

  # create sparse hybridized operators for low order scheme
  Qrst, E = StartUpDG.sparse_low_order_SBP_operators(dg.basis)
  Brst = map(n -> Diagonal(n .* dg.basis.wf), dg.basis.nrstJ)
  sparse_hybridized_SBP_operators = map((Q, B) -> [Q-Q' E'*B; -B*E zeros(size(B))], Qrst, Brst)

  return (; element_ids_dg, element_ids_dgfv, sparse_hybridized_SBP_operators, EToE)
end


# this method is used when the indicator is constructed as for shock-capturing volume integrals
function create_cache(::Type{IndicatorHennemannGassner}, equations::AbstractEquations,
                      basis::RefElemData{NDIMS}) where NDIMS

  alpha = Vector{real(basis)}()
  alpha_tmp = similar(alpha)

  A = Vector{real(basis)}
  indicator_threaded  = [A(undef, nnodes(basis)) for _ in 1:Threads.nthreads()]
  modal_threaded      = [A(undef, nnodes(basis)) for _ in 1:Threads.nthreads()]

  # initialize inverse Vandermonde matrices at Gauss-Legendre nodes
  (; N) = basis
  gauss_node_coordinates_1D, _ = StartUpDG.gauss_quad(0, 0, N)
  VDM_1D = StartUpDG.vandermonde(Line(), N, gauss_node_coordinates_1D)
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

    # Calculate indicator variables at *Gauss* nodes.
    for i in eachnode(dg)
      indicator[i] = indicator_hg.variable(u[i, element], equations)
    end

    # multiply by invVDM::SimpleKronecker
    LinearAlgebra.mul!(modal_, inverse_vandermonde, indicator)

    # reshape into a matrix over each element
    modal = reshape(modal_, ntuple(_ -> dg.basis.N + 1, NDIMS))

    # Calculate total energies for all modes, without highest, without two highest
    total_energy = sum(x -> x^2, modal)

    # TODO: check if this allocates
    clip_1_ranges = ntuple(_ -> Base.OneTo(dg.basis.N), NDIMS)
    clip_2_ranges = ntuple(_ -> Base.OneTo(dg.basis.N - 1), NDIMS)
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
      neighboring_element = cache.EToE[face, element]
      alpha_neighbor = alpha_tmp[neighboring_element]
      alpha[element]  = max(alpha[element], 0.5 * alpha_neighbor)
    end
  end

end

#     pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, dg, cache)
#
# Given blending factors `alpha` and the solver `dg`, fill
# `element_ids_dg` with the IDs of elements using a pure DG scheme and
# `element_ids_dgfv` with the IDs of elements using a blended DG-FV scheme.
function pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, mesh::DGMultiMesh, dg::DGMulti)
  empty!(element_ids_dg)
  empty!(element_ids_dgfv)

  for element in eachelement(mesh, dg)
    # Clip blending factor for values close to zero (-> pure DG)
    dg_only = isapprox(alpha[element], 0, atol=1e-12)
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
                               dg::DGMultiFluxDiff{<:GaussSBP}, cache)

  @unpack element_ids_dg, element_ids_dgfv = cache
  @unpack volume_flux_dg, volume_flux_fv, indicator = volume_integral

  # Calculate blending factors α: u = u_DG * (1 - α) + u_FV * α
  alpha = @trixi_timeit timer() "blending factors" indicator(u, mesh, equations, dg, cache)

  # Determine element ids for DG-only and blended DG-FV volume integral
  pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, mesh, dg)

  # Loop over pure DG elements
  @trixi_timeit timer() "pure DG" @threaded for idx_element in eachindex(element_ids_dg)
    e = element_ids_dg[idx_element]

    fluxdiff_local = cache.fluxdiff_local_threaded[Threads.threadid()]
    fill!(fluxdiff_local, zero(eltype(fluxdiff_local)))
    u_local = view(cache.entropy_projected_u_values, :, e)

    local_flux_differencing!(fluxdiff_local, u_local, e,
                             have_nonconservative_terms, volume_integral,
                             has_sparse_operators(dg),
                             mesh, equations, dg, cache)

    # convert `fluxdiff_local::Vector{<:SVector}` to `rhs_local::StructArray{<:SVector}`
    # for faster performance when using `apply_to_each_field`.
    rhs_local = cache.rhs_local_threaded[Threads.threadid()]
    for i in Base.OneTo(length(fluxdiff_local))
      rhs_local[i] = fluxdiff_local[i]
    end

    # stores rhs contributions only at Gauss volume nodes
    rhs_volume_local = cache.rhs_volume_local_threaded[Threads.threadid()]

    # Here, we exploit that under a Gauss nodal basis the structure of the projection
    # matrix `Ph = [diagm(1 ./ wq), projection_matrix_gauss_to_face]` such that `Ph * [u; uf] = (u ./ wq) + projection_matrix_gauss_to_face * uf`.
    volume_indices = Base.OneTo(dg.basis.Nq)
    face_indices = (dg.basis.Nq + 1):(dg.basis.Nq + dg.basis.Nfq)
    local_volume_flux = view(rhs_local, volume_indices)
    local_face_flux = view(rhs_local, face_indices)

    # initialize rhs_volume_local = projection_matrix_gauss_to_face * local_face_flux
    apply_to_each_field(mul_by!(cache.projection_matrix_gauss_to_face), rhs_volume_local, local_face_flux)

    # accumulate volume contributions at Gauss nodes
    for i in eachindex(rhs_volume_local)
      du[i, e] = rhs_volume_local[i] + local_volume_flux[i] * cache.inv_gauss_weights[i]
    end
  end

  # Loop over blended DG-FV elements
  @trixi_timeit timer() "blended DG-FV" @threaded for idx_element in eachindex(element_ids_dgfv)
    element = element_ids_dgfv[idx_element]
    alpha_element = alpha[element]


    # # Calculate DG volume integral contribution
    # flux_differencing_kernel!(du, u, element, mesh,
    #                           nonconservative_terms, equations,
    #                           volume_flux_dg, dg, cache, 1 - alpha_element)

    # # Calculate FV volume integral contribution
    # fv_kernel!(du, u, mesh, nonconservative_terms, equations, volume_flux_fv,
    #            dg, cache, element, alpha_element)

    # blend them together via r_high * (1 - alpha) + r_low * (alpha)
  end

  return nothing
end
