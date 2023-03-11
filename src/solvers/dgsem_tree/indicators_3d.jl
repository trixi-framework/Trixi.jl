# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


# this method is used when the indicator is constructed as for shock-capturing volume integrals
function create_cache(::Type{IndicatorHennemannGassner}, equations::AbstractEquations{3}, basis::LobattoLegendreBasis)

  alpha = Vector{real(basis)}()
  alpha_tmp = similar(alpha)

  A = Array{real(basis), ndims(equations)}
  indicator_threaded  = [A(undef, nnodes(basis), nnodes(basis), nnodes(basis)) for _ in 1:Threads.nthreads()]
  modal_threaded      = [A(undef, nnodes(basis), nnodes(basis), nnodes(basis)) for _ in 1:Threads.nthreads()]
  modal_tmp1_threaded = [A(undef, nnodes(basis), nnodes(basis), nnodes(basis)) for _ in 1:Threads.nthreads()]
  modal_tmp2_threaded = [A(undef, nnodes(basis), nnodes(basis), nnodes(basis)) for _ in 1:Threads.nthreads()]

  return (; alpha, alpha_tmp, indicator_threaded, modal_threaded, modal_tmp1_threaded, modal_tmp2_threaded)
end

# this method is used when the indicator is constructed as for AMR
function create_cache(typ::Type{IndicatorHennemannGassner}, mesh, equations::AbstractEquations{3}, dg::DGSEM, cache)
  create_cache(typ, equations, dg.basis)
end


function (indicator_hg::IndicatorHennemannGassner)(u::AbstractArray{<:Any,5},
                                                   mesh, equations, dg::DGSEM, cache;
                                                   kwargs...)
  @unpack alpha_max, alpha_min, alpha_smooth, variable = indicator_hg
  @unpack alpha, alpha_tmp, indicator_threaded, modal_threaded,
          modal_tmp1_threaded, modal_tmp2_threaded = indicator_hg.cache
  # TODO: Taal refactor, when to `resize!` stuff changed possibly by AMR?
  #       Shall we implement `resize!(semi::AbstractSemidiscretization, new_size)`
  #       or just `resize!` whenever we call the relevant methods as we do now?
  resize!(alpha, nelements(dg, cache))
  if alpha_smooth
    resize!(alpha_tmp, nelements(dg, cache))
  end

  # magic parameters
  threshold = 0.5 * 10^(-1.8 * (nnodes(dg))^0.25)
  parameter_s = log((1 - 0.0001)/0.0001)

  @threaded for element in eachelement(dg, cache)
    indicator  = indicator_threaded[Threads.threadid()]
    modal      = modal_threaded[Threads.threadid()]
    modal_tmp1 = modal_tmp1_threaded[Threads.threadid()]
    modal_tmp2 = modal_tmp2_threaded[Threads.threadid()]

    # Calculate indicator variables at Gauss-Lobatto nodes
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, j, k, element)
      indicator[i, j, k] = indicator_hg.variable(u_local, equations)
    end

    # Convert to modal representation
    multiply_scalar_dimensionwise!(modal, dg.basis.inverse_vandermonde_legendre, indicator, modal_tmp1, modal_tmp2)

    # Calculate total energies for all modes, without highest, without two highest
    total_energy = zero(eltype(modal))
    for k in 1:nnodes(dg), j in 1:nnodes(dg), i in 1:nnodes(dg)
      total_energy += modal[i, j, k]^2
    end
    total_energy_clip1 = zero(eltype(modal))
    for k in 1:(nnodes(dg)-1), j in 1:(nnodes(dg)-1), i in 1:(nnodes(dg)-1)
      total_energy_clip1 += modal[i, j, k]^2
    end
    total_energy_clip2 = zero(eltype(modal))
    for k in 1:(nnodes(dg)-2), j in 1:(nnodes(dg)-2), i in 1:(nnodes(dg)-2)
      total_energy_clip2 += modal[i, j, k]^2
    end

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


function apply_smoothing!(mesh::Union{TreeMesh{3}, P4estMesh{3}}, alpha, alpha_tmp, dg, cache)

  # Diffuse alpha values by setting each alpha to at least 50% of neighboring elements' alpha
  # Copy alpha values such that smoothing is indpedenent of the element access order
  alpha_tmp .= alpha

  # Loop over interfaces
  for interface in eachinterface(dg, cache)
    # Get neighboring element ids
    left  = cache.interfaces.neighbor_ids[1, interface]
    right = cache.interfaces.neighbor_ids[2, interface]

    # Apply smoothing
    alpha[left]  = max(alpha_tmp[left],  0.5 * alpha_tmp[right], alpha[left])
    alpha[right] = max(alpha_tmp[right], 0.5 * alpha_tmp[left],  alpha[right])
  end

  # Loop over L2 mortars
  for mortar in eachmortar(dg, cache)
    # Get neighboring element ids
    lower_left  = cache.mortars.neighbor_ids[1, mortar]
    lower_right = cache.mortars.neighbor_ids[2, mortar]
    upper_left  = cache.mortars.neighbor_ids[3, mortar]
    upper_right = cache.mortars.neighbor_ids[4, mortar]
    large       = cache.mortars.neighbor_ids[5, mortar]

    # Apply smoothing
    alpha[lower_left]  = max(alpha_tmp[lower_left],  0.5 * alpha_tmp[large], alpha[lower_left])
    alpha[lower_right] = max(alpha_tmp[lower_right], 0.5 * alpha_tmp[large], alpha[lower_right])
    alpha[upper_left]  = max(alpha_tmp[upper_left],  0.5 * alpha_tmp[large], alpha[upper_left])
    alpha[upper_right] = max(alpha_tmp[upper_right], 0.5 * alpha_tmp[large], alpha[upper_right])

    alpha[large] = max(alpha_tmp[large], 0.5 * alpha_tmp[lower_left],  alpha[large])
    alpha[large] = max(alpha_tmp[large], 0.5 * alpha_tmp[lower_right], alpha[large])
    alpha[large] = max(alpha_tmp[large], 0.5 * alpha_tmp[upper_left],  alpha[large])
    alpha[large] = max(alpha_tmp[large], 0.5 * alpha_tmp[upper_right], alpha[large])
  end

end


# this method is used when the indicator is constructed as for shock-capturing volume integrals
function create_cache(::Type{IndicatorLöhner}, equations::AbstractEquations{3}, basis::LobattoLegendreBasis)

  alpha = Vector{real(basis)}()

  A = Array{real(basis), ndims(equations)}
  indicator_threaded = [A(undef, nnodes(basis), nnodes(basis), nnodes(basis)) for _ in 1:Threads.nthreads()]

  return (; alpha, indicator_threaded)
end

# this method is used when the indicator is constructed as for AMR
function create_cache(typ::Type{IndicatorLöhner}, mesh, equations::AbstractEquations{3}, dg::DGSEM, cache)
  create_cache(typ, equations, dg.basis)
end


function (löhner::IndicatorLöhner)(u::AbstractArray{<:Any,5},
                                   mesh, equations, dg::DGSEM, cache;
                                   kwargs...)
  @assert nnodes(dg) >= 3 "IndicatorLöhner only works for nnodes >= 3 (polydeg > 1)"
  @unpack alpha, indicator_threaded = löhner.cache
  resize!(alpha, nelements(dg, cache))

  @threaded for element in eachelement(dg, cache)
    indicator = indicator_threaded[Threads.threadid()]

    # Calculate indicator variables at Gauss-Lobatto nodes
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, j, k, element)
      indicator[i, j, k] = löhner.variable(u_local, equations)
    end

    estimate = zero(real(dg))
    for k in eachnode(dg), j in eachnode(dg), i in 2:nnodes(dg)-1
      # x direction
      u0 = indicator[i,   j, k]
      up = indicator[i+1, j, k]
      um = indicator[i-1, j, k]
      estimate = max(estimate, local_löhner_estimate(um, u0, up, löhner))
    end

    for k in eachnode(dg), j in 2:nnodes(dg)-1, i in eachnode(dg)
      # y direction
      u0 = indicator[i, j,   k]
      up = indicator[i, j+1, k]
      um = indicator[i, j-1, k]
      estimate = max(estimate, local_löhner_estimate(um, u0, up, löhner))
    end

    for k in 2:nnodes(dg)-1, j in eachnode(dg), i in eachnode(dg)
      # y direction
      u0 = indicator[i, j, k  ]
      up = indicator[i, j, k+1]
      um = indicator[i, j, k-1]
      estimate = max(estimate, local_löhner_estimate(um, u0, up, löhner))
    end

    # use the maximum as DG element indicator
    alpha[element] = estimate
  end

  return alpha
end


# this method is used when the indicator is constructed as for shock-capturing volume integrals
function create_cache(::Type{IndicatorMax}, equations::AbstractEquations{3}, basis::LobattoLegendreBasis)

  alpha = Vector{real(basis)}()

  A = Array{real(basis), ndims(equations)}
  indicator_threaded = [A(undef, nnodes(basis), nnodes(basis), nnodes(basis)) for _ in 1:Threads.nthreads()]

  return (; alpha, indicator_threaded)
end

# this method is used when the indicator is constructed as for AMR
function create_cache(typ::Type{IndicatorMax}, mesh, equations::AbstractEquations{3}, dg::DGSEM, cache)
  cache = create_cache(typ, equations, dg.basis)
end


function (indicator_max::IndicatorMax)(u::AbstractArray{<:Any,5},
                                       mesh, equations, dg::DGSEM, cache;
                                       kwargs...)
  @unpack alpha, indicator_threaded = indicator_max.cache
  resize!(alpha, nelements(dg, cache))

  @threaded for element in eachelement(dg, cache)
    indicator = indicator_threaded[Threads.threadid()]

    # Calculate indicator variables at Gauss-Lobatto nodes
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, j, k, element)
      indicator[i, j, k] = indicator_max.variable(u_local, equations)
    end

    alpha[element] = maximum(indicator)
  end

  return alpha
end


end # @muladd
