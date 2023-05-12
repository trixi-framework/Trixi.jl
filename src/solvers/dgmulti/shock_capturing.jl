# by default, return an empty tuple for volume integral caches
function create_cache(mesh::DGMultiMesh{NDIMS}, equations,
                      volume_integral::VolumeIntegralShockCapturingHG,
                      dg::DGMultiFluxDiff{<:GaussSBP}, uEltype) where {NDIMS}
  element_ids_dg   = Int[]
  element_ids_dgfv = Int[]

  # create sparse hybridized operators for shock capturing
  Qrst, E = StartUpDG.sparse_low_order_SBP_operators(dg.basis)
  Brst = map(n -> diagm(n .* dg.basis.wf), dg.basis.nrstJ)
  sparse_hybridized_SBP_operators = map((Q, B) -> [Q-Q' E'*B; -B*E B], Qrst, Brst)

  return (; cache..., element_ids_dg, element_ids_dgfv, sparse_hybridized_SBP_operators)
end


# this method is used when the indicator is constructed as for shock-capturing volume integrals
function create_cache(::Type{IndicatorHennemannGassner}, equations::AbstractEquations,
                      basis::RefElemData{NDIMS}) where NDIMS

  alpha = Vector{real(basis)}(undef, md.num_elements)
  alpha_tmp = similar(alpha)

  A = Array{real(basis), ndims(equations)}
  indicator_threaded  = [A(undef, nnodes(basis)) for _ in 1:Threads.nthreads()]
  modal_threaded      = [A(undef, nnodes(basis)) for _ in 1:Threads.nthreads()]

  # initialize inverse Vandermonde matrices at Gauss-Legendre nodes
  (; N) = dg.basis
  gauss_node_coordinates_1D, _ = StartUpDG.gauss_quad(0, 0, N)
  VDM_1D = StartUpDG.vandermonde(Line(), N, gauss_node_coordinates_1D)
  inverse_vandermonde = SimpleKronecker(NDIMS, inv(VDM_1D))

  return (; alpha, alpha_tmp, indicator_threaded, modal_threaded, inverse_vandermonde)
end


function (indicator_hg::IndicatorHennemannGassner)(u, mesh::DGMultiMesh,
                                                   equations, dg::DGMulti, cache;
                                                   kwargs...)
  (; alpha_max, alpha_min, alpha_smooth, variable) = indicator_hg
  (; alpha, alpha_tmp, indicator_threaded, modal_threaded, inverse_vandermonde) = indicator_hg.cache

  nnodes_1D = dg.basis.N + 1

  # magic parameters
  threshold = 0.5 * 10^(-1.8 * (nnodes_1D)^0.25)
  parameter_s = log((1 - 0.0001)/0.0001)

  @threaded for element in eachelement(dg, cache)
    indicator = indicator_threaded[Threads.threadid()]
    modal_vec = modal_threaded[Threads.threadid()]

    # Calculate indicator variables at Gauss-Lobatto nodes
    for i in eachnode(dg)
      indicator[i] = indicator_hg.variable(u[i, element], equations)
    end

    # multiply by invVDM::SimpleKronecker
    LinearAlgebra.mul!(modal_vec, inverse_vandermonde, x_in)

    # reshape into a matrix over each element
    modal = reshape(modal_vec, ntuple(_ -> nnodes_1D, NDIMS))

    # Calculate total energies for all modes, without highest, without two highest
    total_energy = sum(x -> x^2, modal)

    # TODO: check if this allocates
    clip_1_ranges = ntuple(_ -> Base.OneTo(nnodes_1D-1), NDIMS)
    clip_2_ranges = ntuple(_ -> Base.OneTo(nnodes_1D-2), NDIMS)
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

  if alpha_smooth
    apply_smoothing!(mesh, alpha, alpha_tmp, dg, cache)
  end

  return alpha
end