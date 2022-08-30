# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


# this method is used when the indicator is constructed as for shock-capturing volume integrals
function create_cache(::Type{IndicatorHennemannGassner}, equations::AbstractEquations{2}, basis::LobattoLegendreBasis)

  alpha = Vector{real(basis)}()
  alpha_tmp = similar(alpha)

  A = Array{real(basis), ndims(equations)}
  indicator_threaded  = [A(undef, nnodes(basis), nnodes(basis)) for _ in 1:Threads.nthreads()]
  modal_threaded      = [A(undef, nnodes(basis), nnodes(basis)) for _ in 1:Threads.nthreads()]
  modal_tmp1_threaded = [A(undef, nnodes(basis), nnodes(basis)) for _ in 1:Threads.nthreads()]

  return (; alpha, alpha_tmp, indicator_threaded, modal_threaded, modal_tmp1_threaded)
end

# this method is used when the indicator is constructed as for AMR
function create_cache(typ::Type{IndicatorHennemannGassner}, mesh, equations::AbstractEquations{2}, dg::DGSEM, cache)
  create_cache(typ, equations, dg.basis)
end


function (indicator_hg::IndicatorHennemannGassner)(u::AbstractArray{<:Any,4},
                                                   mesh, equations, dg::DGSEM, cache;
                                                   kwargs...)
  @unpack alpha_max, alpha_min, alpha_smooth, variable = indicator_hg
  @unpack alpha, alpha_tmp, indicator_threaded, modal_threaded, modal_tmp1_threaded = indicator_hg.cache
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

    # Calculate indicator variables at Gauss-Lobatto nodes
    for j in eachnode(dg), i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, j, element)
      indicator[i, j] = indicator_hg.variable(u_local, equations)
    end

    # Convert to modal representation
    multiply_scalar_dimensionwise!(modal, dg.basis.inverse_vandermonde_legendre, indicator, modal_tmp1)

    # Calculate total energies for all modes, without highest, without two highest
    total_energy = zero(eltype(modal))
    for j in 1:nnodes(dg), i in 1:nnodes(dg)
      total_energy += modal[i, j]^2
    end
    total_energy_clip1 = zero(eltype(modal))
    for j in 1:(nnodes(dg)-1), i in 1:(nnodes(dg)-1)
      total_energy_clip1 += modal[i, j]^2
    end
    total_energy_clip2 = zero(eltype(modal))
    for j in 1:(nnodes(dg)-2), i in 1:(nnodes(dg)-2)
      total_energy_clip2 += modal[i, j]^2
    end

    # Calculate energy in higher modes
    energy = max((total_energy - total_energy_clip1) / total_energy,
                 (total_energy_clip1 - total_energy_clip2) / total_energy_clip1)

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


# Diffuse alpha values by setting each alpha to at least 50% of neighboring elements' alpha
function apply_smoothing!(mesh::Union{TreeMesh{2}, P4estMesh{2}}, alpha, alpha_tmp, dg, cache)
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
    lower = cache.mortars.neighbor_ids[1, mortar]
    upper = cache.mortars.neighbor_ids[2, mortar]
    large = cache.mortars.neighbor_ids[3, mortar]

    # Apply smoothing
    alpha[lower] = max(alpha_tmp[lower], 0.5 * alpha_tmp[large], alpha[lower])
    alpha[upper] = max(alpha_tmp[upper], 0.5 * alpha_tmp[large], alpha[upper])
    alpha[large] = max(alpha_tmp[large], 0.5 * alpha_tmp[lower], alpha[large])
    alpha[large] = max(alpha_tmp[large], 0.5 * alpha_tmp[upper], alpha[large])
  end

  return alpha
end


# this method is used when the indicator is constructed as for shock-capturing volume integrals
function create_cache(::Type{IndicatorLöhner}, equations::AbstractEquations{2}, basis::LobattoLegendreBasis)

  alpha = Vector{real(basis)}()

  A = Array{real(basis), ndims(equations)}
  indicator_threaded = [A(undef, nnodes(basis), nnodes(basis)) for _ in 1:Threads.nthreads()]

  return (; alpha, indicator_threaded)
end

# this method is used when the indicator is constructed as for AMR
function create_cache(typ::Type{IndicatorLöhner}, mesh, equations::AbstractEquations{2}, dg::DGSEM, cache)
  create_cache(typ, equations, dg.basis)
end


function (löhner::IndicatorLöhner)(u::AbstractArray{<:Any,4},
                                   mesh, equations, dg::DGSEM, cache;
                                   kwargs...)
  @assert nnodes(dg) >= 3 "IndicatorLöhner only works for nnodes >= 3 (polydeg > 1)"
  @unpack alpha, indicator_threaded = löhner.cache
  resize!(alpha, nelements(dg, cache))

  @threaded for element in eachelement(dg, cache)
    indicator = indicator_threaded[Threads.threadid()]

    # Calculate indicator variables at Gauss-Lobatto nodes
    for j in eachnode(dg), i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, j, element)
      indicator[i, j] = löhner.variable(u_local, equations)
    end

    estimate = zero(real(dg))
    for j in eachnode(dg), i in 2:nnodes(dg)-1
      # x direction
      u0 = indicator[i,   j]
      up = indicator[i+1, j]
      um = indicator[i-1, j]
      estimate = max(estimate, local_löhner_estimate(um, u0, up, löhner))
    end

    for j in 2:nnodes(dg)-1, i in eachnode(dg)
      # y direction
      u0 = indicator[i, j  ]
      up = indicator[i, j+1]
      um = indicator[i, j-1]
      estimate = max(estimate, local_löhner_estimate(um, u0, up, löhner))
    end

    # use the maximum as DG element indicator
    alpha[element] = estimate
  end

  return alpha
end


# this method is used when the indicator is constructed as for shock-capturing volume integrals
function create_cache(::Type{IndicatorIDP}, equations::AbstractEquations{2}, basis::LobattoLegendreBasis, length)
  ContainerShockCapturingIndicator = Trixi.ContainerShockCapturingIndicatorIDP{real(basis)}(0, nnodes(basis), length)

  alpha_max_per_timestep  = zeros(real(basis), 200)
  alpha_mean_per_timestep = zeros(real(basis), 200)

  idp_bounds_delta_threaded = [zeros(real(basis), length) for _ in 1:Threads.nthreads()]

  return (; ContainerShockCapturingIndicator,
            alpha_max_per_timestep, alpha_mean_per_timestep,
            idp_bounds_delta_threaded)
end

function (indicator_IDP::IndicatorIDP)(u_safe::AbstractArray{<:Any,4}, u_old::AbstractArray{<:Any,4},
                                       mesh, equations, dg::DGSEM,
                                       dt, cache;
                                       kwargs...)
  @unpack alpha = indicator_IDP.cache.ContainerShockCapturingIndicator
  alpha .= 0.0 # TODO: Correct that we save only the alpha's of the last RK stage.

  indicator_IDP.IDPDensityTVD  &&
    @trixi_timeit timer() "IDPDensityTVD"  IDP_densityTVD!( alpha, indicator_IDP, u_safe,         dt, mesh, equations, dg, cache)
  indicator_IDP.IDPPressureTVD &&
    @trixi_timeit timer() "IDPPressureTVD" IDP_pressureTVD!(alpha, indicator_IDP, u_safe,         dt, mesh, equations, dg, cache)
  indicator_IDP.IDPPositivity  &&
    @trixi_timeit timer() "IDPPositivity"  IDP_positivity!( alpha, indicator_IDP, u_safe,         dt, mesh, equations, dg, cache)
  indicator_IDP.IDPSpecEntropy &&
    @trixi_timeit timer() "IDPSpecEntropy" IDP_specEntropy!(alpha, indicator_IDP, u_safe, u_safe, dt, mesh, equations, dg, cache)
  indicator_IDP.IDPMathEntropy &&
    @trixi_timeit timer() "IDPMathEntropy" IDP_mathEntropy!(alpha, indicator_IDP, u_safe, u_safe, dt, mesh, equations, dg, cache)

  # Clip the maximum amount of FV allowed (default: alpha_maxIDP = 1.0)
  @unpack alpha_maxIDP = indicator_IDP
  if alpha_maxIDP != 1.0
    @threaded for element in eachelement(dg, cache)
      for j in eachnode(dg), i in eachnode(dg)
        alpha[i, j, element] = min(alpha_maxIDP, alpha[i, j, element])
      end
    end
  end

  # Calculate alpha1 and alpha2
  @unpack alpha1, alpha2 = indicator_IDP.cache.ContainerShockCapturingIndicator
  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in 2:nnodes(dg)
      alpha1[i, j, element] = max(alpha[i-1, j, element], alpha[i, j, element])
    end
    for j in 2:nnodes(dg), i in eachnode(dg)
      alpha2[i, j, element] = max(alpha[i, j-1, element], alpha[i, j, element])
    end
    alpha1[1,            :, element] .= zero(eltype(alpha1))
    alpha1[nnodes(dg)+1, :, element] .= zero(eltype(alpha1))
    alpha2[:,            1, element] .= zero(eltype(alpha2))
    alpha2[:, nnodes(dg)+1, element] .= zero(eltype(alpha2))
  end

  return nothing
end

@inline function calc_bounds_2sided!(var_min, var_max, variable, u, mesh, equations, dg, cache)
  # Values inside each element
  @threaded for element in eachelement(dg, cache)
    var_min[:, :, element] .= typemax(eltype(var_min))
    var_max[:, :, element] .= typemin(eltype(var_max))
    # Calculate indicator variables at Gauss-Lobatto nodes
    for j in eachnode(dg), i in eachnode(dg)
      var = variable(get_node_vars(u, equations, dg, i, j, element), equations)
      var_min[i, j, element] = min(var_min[i, j, element], var)
      var_max[i, j, element] = max(var_max[i, j, element], var)

      if i > 1
        var_min[i-1, j, element] = min(var_min[i-1, j, element], var)
        var_max[i-1, j, element] = max(var_max[i-1, j, element], var)
      end
      if i < nnodes(dg)
        var_min[i+1, j, element] = min(var_min[i+1, j, element], var)
        var_max[i+1, j, element] = max(var_max[i+1, j, element], var)
      end
      if j > 1
        var_min[i, j-1, element] = min(var_min[i, j-1, element], var)
        var_max[i, j-1, element] = max(var_max[i, j-1, element], var)
      end
      if j < nnodes(dg)
        var_min[i, j+1, element] = min(var_min[i, j+1, element], var)
        var_max[i, j+1, element] = max(var_max[i, j+1, element], var)
      end
    end
  end

  # Values at element boundary
  calc_bounds_2sided_interface!(var_min, var_max, variable, u, mesh, equations, dg, cache)
end

@inline function calc_bounds_2sided_interface!(var_min, var_max, variable, u, mesh::TreeMesh2D, equations, dg, cache)
  for interface in eachinterface(dg, cache)
    # Get neighboring element ids
    left  = cache.interfaces.neighbor_ids[1, interface]
    right = cache.interfaces.neighbor_ids[2, interface]

    orientation = cache.interfaces.orientations[interface]

    for i in eachnode(dg)
      if orientation == 1
        index_left  = (nnodes(dg), i, left)
        index_right = (1,          i, right)
      else
        index_left  = (i, nnodes(dg), left)
        index_right = (i,          1, right)
      end
      var_left  = variable(get_node_vars(u, equations, dg, index_left...), equations)
      var_right = variable(get_node_vars(u, equations, dg, index_right...), equations)

      var_min[index_right...] = min(var_min[index_right...], var_left)
      var_max[index_right...] = max(var_max[index_right...], var_left)

      var_min[index_left...] = min(var_min[index_left...], var_right)
      var_max[index_left...] = max(var_max[index_left...], var_right)
    end
  end
end

@inline function calc_bounds_1sided!(var_minmax, minmax, typeminmax, variable, u, mesh, equations, dg, cache)
  # Values inside each element
  @threaded for element in eachelement(dg, cache)
    var_minmax[:, :, element] .= typeminmax(eltype(var_minmax))

    # Calculate indicator variables at Gauss-Lobatto nodes
    for j in eachnode(dg), i in eachnode(dg)
      var = variable(get_node_vars(u, equations, dg, i, j, element), equations)
      var_minmax[i, j, element] = minmax(var_minmax[i, j, element], var)

      if i > 1
        var_minmax[i-1, j, element] = minmax(var_minmax[i-1, j, element], var)
      end
      if i < nnodes(dg)
        var_minmax[i+1, j, element] = minmax(var_minmax[i+1, j, element], var)
      end
      if j > 1
        var_minmax[i, j-1, element] = minmax(var_minmax[i, j-1, element], var)
      end
      if j < nnodes(dg)
        var_minmax[i, j+1, element] = minmax(var_minmax[i, j+1, element], var)
      end
    end
  end

  # Values at element boundary
  calc_bounds_1sided_interface!(var_minmax, minmax, variable, u, mesh, equations, dg, cache)
end

@inline function calc_bounds_1sided_interface!(var_minmax, minmax, variable, u, mesh::TreeMesh2D, equations, dg, cache)
  for interface in eachinterface(dg, cache)
    # Get neighboring element ids
    left  = cache.interfaces.neighbor_ids[1, interface]
    right = cache.interfaces.neighbor_ids[2, interface]

    orientation = cache.interfaces.orientations[interface]

    for i in eachnode(dg)
      if orientation == 1
        index_left  = (nnodes(dg), i, left)
        index_right = (1,          i, right)
      else
        index_left  = (i, nnodes(dg), left)
        index_right = (i,          1, right)
      end
      var_left  = variable(get_node_vars(u, equations, dg, index_left...), equations)
      var_right = variable(get_node_vars(u, equations, dg, index_right...), equations)

      var_minmax[index_right...] = minmax(var_minmax[index_right...], var_left)
      var_minmax[index_left...]  = minmax(var_minmax[index_left...],  var_right)
    end
  end
end

@inline function IDP_densityTVD!(alpha, indicator_IDP, u_safe, dt, mesh, equations, dg, cache)
  @unpack var_bounds = indicator_IDP.cache.ContainerShockCapturingIndicator

  rho_min = var_bounds[1]
  rho_max = var_bounds[2]
  calc_bounds_2sided!(rho_min, rho_max, density, u_safe, mesh, equations, dg, cache)

  @unpack antidiffusive_flux1, antidiffusive_flux2 = cache.ContainerAntidiffusiveFlux2D
  @unpack inverse_weights = dg.basis
  @threaded for element in eachelement(dg, cache)
    inverse_jacobian = cache.elements.inverse_jacobian[element]

    for j in eachnode(dg), i in eachnode(dg)
      rho = u_safe[1, i, j, element]
      # Real Zalesak type limiter
      #   * Zalesak (1979). "Fully multidimensional flux-corrected transport algorithms for fluids"
      #   * Kuzmin et al. (2010). "Failsafe flux limiting and constrained data projections for equations of gas dynamics"
      #   Note: The Zalesak limiter has to be computed, even if the state is valid, because the correction is
      #         for each interface, not each node

      # Calculate P_plus and P_minus
      # Note: Boundaries of antidiffusive_flux1/2 are constant 0, so they make no difference here.
      val_flux1_local     =  inverse_weights[i] * antidiffusive_flux1[1,   i,   j, element]
      val_flux1_local_ip1 = -inverse_weights[i] * antidiffusive_flux1[1, i+1,   j, element]
      val_flux2_local     =  inverse_weights[j] * antidiffusive_flux2[1,   i,   j, element]
      val_flux2_local_jp1 = -inverse_weights[j] * antidiffusive_flux2[1,   i, j+1, element]

      P_plus  = max(0.0, val_flux1_local) + max(0.0, val_flux1_local_ip1) +
                max(0.0, val_flux2_local) + max(0.0, val_flux2_local_jp1)
      P_minus = min(0.0, val_flux1_local) + min(0.0, val_flux1_local_ip1) +
                min(0.0, val_flux2_local) + min(0.0, val_flux2_local_jp1)
      P_plus  = dt * inverse_jacobian * P_plus
      P_minus = dt * inverse_jacobian * P_minus

      # Calculate alpha_plus and alpha_minus
      if P_plus == 0.0
        frac_plus = 1.0
      else
        frac_plus = max(0.0, rho_max[i, j, element] - rho) / P_plus
      end
      if P_minus == 0.0
        frac_minus = 1.0
      else
        frac_minus = min(0.0, rho_min[i, j, element] - rho) / P_minus
      end

      # Calculate alpha at nodes
      alpha[i, j, element]  = 1 - min(1.0, frac_plus, frac_minus)
    end
  end

  return nothing
end

@inline function IDP_pressureTVD!(alpha, indicator_IDP, u_safe, dt, mesh, equations, dg, cache)
  # IDP limiter for pressure based on
  # - Kuzmin et al. (2020). "Failsafe flux limiting and constrained data projections for equations of gas dynamics"
  @unpack var_bounds = indicator_IDP.cache.ContainerShockCapturingIndicator

  offset = 2 * indicator_IDP.IDPDensityTVD
  p_min = var_bounds[1 + offset]
  p_max = var_bounds[2 + offset]
  calc_bounds_2sided!(p_min, p_max, pressure, u_safe, mesh, equations, dg, cache)

  @unpack antidiffusive_flux1, antidiffusive_flux2 = cache.ContainerAntidiffusiveFlux2D
  @unpack inverse_weights = dg.basis
  @threaded for element in eachelement(dg, cache)
    inverse_jacobian = cache.elements.inverse_jacobian[element]

    for j in eachnode(dg), i in eachnode(dg)
      p = pressure(get_node_vars(u_safe, equations, dg, i, j, element), equations)
      # Real Zalesak type limiter
      #   * Zalesak (1979). "Fully multidimensional flux-corrected transport algorithms for fluids"
      #   * Kuzmin et al. (2010). "Failsafe flux limiting and constrained data projections for equations of gas dynamics"
      #   Note: The Zalesak limiter has to be computed, even if the state is valid, because the correction is
      #         for each interface, not each node

      # Calculate P_plus and P_minus
      # Note: Boundaries of antidiffusive_flux1/2 are constant 0, so they make no difference here.
      v1 = u_safe[2, i, j, element] / u_safe[1, i, j, element]
      v2 = u_safe[3, i, j, element] / u_safe[1, i, j, element]
      v2s2 = 0.5 * (v1^2 + v2^2)
      gamma_m1 = equations.gamma - 1.0

      val_flux1_local     = gamma_m1 * (antidiffusive_flux1[4, i, j, element] + v2s2 * antidiffusive_flux1[1, i, j, element] -
                                          v1 * antidiffusive_flux1[2, i, j, element] - v2 * antidiffusive_flux1[3, i, j, element])
      val_flux1_local_ip1 = gamma_m1 * (antidiffusive_flux1[4, i+1, j, element] + v2s2 * antidiffusive_flux1[1, i+1, j, element] -
                                          v1 * antidiffusive_flux1[2, i+1, j, element] - v2 * antidiffusive_flux1[3, i+1, j, element])
      val_flux2_local     = gamma_m1 * (antidiffusive_flux2[4, i, j, element] + v2s2 * antidiffusive_flux2[1, i, j, element] -
                                          v1 * antidiffusive_flux2[2, i, j, element] - v2 * antidiffusive_flux2[3, i, j, element])
      val_flux2_local_jp1 = gamma_m1 * (antidiffusive_flux2[4, i, j+1, element] + v2s2 * antidiffusive_flux2[1, i, j+1, element] -
                                          v1 * antidiffusive_flux2[2, i, j+1, element] - v2 * antidiffusive_flux2[3, i, j+1, element])

      P_plus  = max(0.0, inverse_weights[i] * val_flux1_local) + max(0.0, -inverse_weights[i] * val_flux1_local_ip1) +
                max(0.0, inverse_weights[j] * val_flux2_local) + max(0.0, -inverse_weights[j] * val_flux2_local_jp1)
      P_minus = min(0.0, inverse_weights[i] * val_flux1_local) + min(0.0, -inverse_weights[i] * val_flux1_local_ip1) +
                min(0.0, inverse_weights[j] * val_flux2_local) + min(0.0, -inverse_weights[j] * val_flux2_local_jp1)
      P_plus  = dt * inverse_jacobian * P_plus
      P_minus = dt * inverse_jacobian * P_minus

      # Calculate alpha_plus and alpha_minus
      if P_plus == 0.0
        frac_plus = 1.0
      else
        frac_plus = max(0.0, p_max[i, j, element] - p) / P_plus
      end
      if P_minus == 0.0
        frac_minus = 1.0
      else
        frac_minus = min(0.0, p_min[i, j, element] - p) / P_minus
      end

      # Calculate alpha at nodes
      alpha[i, j, element]  = max(alpha[i, j, element], 1 - min(1.0, frac_plus, frac_minus))
    end
  end

  return nothing
end

@inline function IDP_specEntropy!(alpha, indicator_IDP, u_safe, u_old, dt, mesh, equations, dg, cache)
  @unpack IDPDensityTVD, IDPPressureTVD, IDPPositivity = indicator_IDP
  @unpack var_bounds = indicator_IDP.cache.ContainerShockCapturingIndicator

  offset = 2 * (IDPDensityTVD + IDPPressureTVD) + min(IDPPositivity, !IDPDensityTVD) + min(IDPPositivity, !IDPPressureTVD)
  s_min = var_bounds[offset + 1]
  calc_bounds_1sided!(s_min, min, typemax, entropy_spec, u_old, mesh, equations, dg, cache)

  # Perform Newton's bisection method to find new alpha
  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      u_local = get_node_vars(u_safe, equations, dg, i, j, element)
      newton_loops_alpha!(alpha, s_min[i, j, element], u_local, i, j, element,
                          specEntropy_goal, specEntropy_dGoal_dbeta, specEntropy_initialCheck, standard_finalCheck,
                          dt, mesh, equations, dg, cache, indicator_IDP)
    end
  end

  return nothing
end

specEntropy_goal(bound, u, equations) = bound - entropy_spec(u, equations)
specEntropy_dGoal_dbeta(u, dt, antidiffusive_flux, equations) = -dot(cons2entropy_spec(u, equations), dt * antidiffusive_flux)
specEntropy_initialCheck(bound, goal, newton_abstol) = goal <= max(newton_abstol, abs(bound) * newton_abstol)

@inline function IDP_mathEntropy!(alpha, indicator_IDP, u_safe, u_old, dt, mesh, equations, dg, cache)
  @unpack IDPDensityTVD, IDPPressureTVD, IDPPositivity, IDPSpecEntropy = indicator_IDP
  @unpack var_bounds = indicator_IDP.cache.ContainerShockCapturingIndicator

  offset = 2 * (IDPDensityTVD + IDPPressureTVD) + IDPSpecEntropy +
           min(IDPPositivity, !IDPDensityTVD)+ min(IDPPositivity, !IDPPressureTVD)
  s_max = var_bounds[offset + 1]
  calc_bounds_1sided!(s_max, max, typemin, entropy_math, u_old, mesh, equations, dg, cache)

  # Perform Newton's bisection method to find new alpha
  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      u_local = get_node_vars(u_safe, equations, dg, i, j, element)
      newton_loops_alpha!(alpha, s_max[i, j, element], u_local, i, j, element,
                          mathEntropy_goal, mathEntropy_dGoal_dbeta, mathEntropy_initialCheck, standard_finalCheck,
                          dt, mesh, equations, dg, cache, indicator_IDP)
    end
  end

  return nothing
end

mathEntropy_goal(bound, u, equations) = bound - entropy_math(u, equations)
mathEntropy_dGoal_dbeta(u, dt, antidiffusive_flux, equations) = -dot(cons2entropy(u, equations), dt * antidiffusive_flux)
mathEntropy_initialCheck(bound, goal, newton_abstol) = goal >= -max(newton_abstol, abs(bound) * newton_abstol)

@inline function IDP_positivity!(alpha, indicator_IDP, u_safe, dt, mesh, equations, dg, cache)
  @unpack antidiffusive_flux1, antidiffusive_flux2 = cache.ContainerAntidiffusiveFlux2D
  @unpack inverse_weights = dg.basis
  @unpack positCorrFactor = indicator_IDP

  @unpack var_bounds = indicator_IDP.cache.ContainerShockCapturingIndicator

  if indicator_IDP.IDPDensityTVD
    rho_min = var_bounds[1]
    p_min   = var_bounds[3]
  else
    if indicator_IDP.IDPPressureTVD
      rho_min = var_bounds[3]
      p_min   = var_bounds[1]
    else
      rho_min = var_bounds[1]
      p_min   = var_bounds[2]
    end
  end

  @threaded for element in eachelement(dg, cache)
    inverse_jacobian = cache.elements.inverse_jacobian[element]
    for j in eachnode(dg), i in eachnode(dg)

      #######################
      # Correct density
      #######################
      if u_safe[1, i, j, element] < 0.0
        println("Error: safe density is not safe. element=$element, node: $i $j, density=$(u_safe[1, i, j, element])")
      end

      # Compute bound
      if indicator_IDP.IDPDensityTVD
        rho_min[i, j, element] = max(rho_min[i, j, element], positCorrFactor * u_safe[1, i, j, element])
      else
        rho_min[i, j, element] = positCorrFactor * u_safe[1, i, j, element]
      end

      # Real one-sided Zalesak-type limiter
      # * Zalesak (1979). "Fully multidimensional flux-corrected transport algorithms for fluids"
      # * Kuzmin et al. (2010). "Failsafe flux limiting and constrained data projections for equations of gas dynamics"
      # Note: The Zalesak limiter has to be computed, even if the state is valid, because the correction is
      #       for each interface, not each node
      frac_minus = min(0.0, rho_min[i, j, element] - u_safe[1, i, j, element])

      # Calculate P_minus
      # Note: Boundaries of antidiffusive_flux1/2 are constant 0, so they make no difference here.
      val_flux1_local     =  inverse_weights[i] * antidiffusive_flux1[1,   i,   j, element]
      val_flux1_local_ip1 = -inverse_weights[i] * antidiffusive_flux1[1, i+1,   j, element]
      val_flux2_local     =  inverse_weights[j] * antidiffusive_flux2[1,   i,   j, element]
      val_flux2_local_jp1 = -inverse_weights[j] * antidiffusive_flux2[1,   i, j+1, element]

      P_minus = min(0.0, val_flux1_local) + min(0.0, val_flux1_local_ip1) +
                min(0.0, val_flux2_local) + min(0.0, val_flux2_local_jp1)
      P_minus = dt * inverse_jacobian * P_minus

      if P_minus < 0.0
        frac_minus = min(1.0, frac_minus / P_minus)
      else
        frac_minus = 1.0
      end

      # Calculate alpha
      alpha[i, j, element]  = max(alpha[i, j, element], 1 - frac_minus)

      #######################
      # Correct pressure
      #######################

      # Compute bound
      u_local = get_node_vars(u_safe, equations, dg, i, j, element)
      p_safe = pressure(u_local, equations)
      if p_safe < 0.0
        println("Error: safe pressure is not safe. element=$element, node: $i $j, pressure=$p_safe")
      end
      if indicator_IDP.IDPPressureTVD
        p_min[i, j, element] = max(p_min[i, j, element], positCorrFactor * p_safe)
      else
        p_min[i, j, element] = positCorrFactor * p_safe
      end

      # Perform Newton's bisection method to find new alpha
      newton_loops_alpha!(alpha, p_min[i, j, element], u_local, i, j, element,
                          pressure_goal, pressure_dgoal_dbeta, pressure_initialCheck, pressure_finalCheck,
                          dt, mesh, equations, dg, cache, indicator_IDP)
    end
  end

  return nothing
end

pressure_goal(bound, u, equations) = bound - pressure(u, equations)
pressure_dgoal_dbeta(u, dt, antidiffusive_flux, equations) = -dot(dpdu(u, equations), dt * antidiffusive_flux)
pressure_initialCheck(bound, goal, newton_abstol) = goal <= 0.0
pressure_finalCheck(bound, goal, newton_abstol) = (goal <= eps()) && (goal > -max(newton_abstol, abs(bound) * newton_abstol))

@inline function newton_loops_alpha!(alpha, bound, u_safe, i, j, element,
                                     goal_fct, dgoal_fct, initialCheck, finalCheck,
                                     dt, mesh, equations, dg, cache, indicator_IDP)
  @unpack inverse_weights = dg.basis
  @unpack antidiffusive_flux1, antidiffusive_flux2 = cache.ContainerAntidiffusiveFlux2D
  inverse_jacobian = cache.elements.inverse_jacobian[element]

  @unpack IDPgamma = indicator_IDP

  # negative xi direction
  antidiffusive_flux = IDPgamma * inverse_jacobian * inverse_weights[i] * get_node_vars(antidiffusive_flux1, equations, dg, i, j, element)
  newton_loop!(alpha, bound, u_safe, i, j, element, goal_fct, dgoal_fct, initialCheck, finalCheck, equations, dt, indicator_IDP, antidiffusive_flux)

  # positive xi direction
  antidiffusive_flux = -IDPgamma * inverse_jacobian * inverse_weights[i] * get_node_vars(antidiffusive_flux1, equations, dg, i+1, j, element)
  newton_loop!(alpha, bound, u_safe, i, j, element, goal_fct, dgoal_fct, initialCheck, finalCheck, equations, dt, indicator_IDP, antidiffusive_flux)

  # negative eta direction
  antidiffusive_flux = IDPgamma * inverse_jacobian * inverse_weights[j] * get_node_vars(antidiffusive_flux2, equations, dg, i, j, element)
  newton_loop!(alpha, bound, u_safe, i, j, element, goal_fct, dgoal_fct, initialCheck, finalCheck, equations, dt, indicator_IDP, antidiffusive_flux)

  # positive eta direction
  antidiffusive_flux = -IDPgamma * inverse_jacobian * inverse_weights[j] * get_node_vars(antidiffusive_flux2, equations, dg, i, j+1, element)
  newton_loop!(alpha, bound, u_safe, i, j, element, goal_fct, dgoal_fct, initialCheck, finalCheck, equations, dt, indicator_IDP, antidiffusive_flux)

  return nothing
end

@inline function newton_loop!(alpha, bound, u_safe, i, j, element,
                              goal_fct, dgoal_fct, initialCheck, finalCheck,
                              equations, dt, indicator_IDP, antidiffusive_flux)
  newton_reltol, newton_abstol = indicator_IDP.newton_tol

  beta = 1.0 - alpha[i, j, element]

  beta_L = 0.0  # alpha = 1
  beta_R = beta # No higher beta (lower alpha) than the current one

  u_curr = u_safe + beta * dt * antidiffusive_flux

  # If state is valid, perform initial check and return if correction is not needed
  if isValidState(u_curr, equations)
    as = goal_fct(bound, u_curr, equations)

    initialCheck(bound, as, newton_abstol) && return nothing
  end

  # Newton iterations
  for iter in 1:indicator_IDP.IDPMaxIter
    beta_old = beta

    # If the state is valid, evaluate d(goal)/d(beta)
    if isValidState(u_curr, equations)
      dSdbeta = dgoal_fct(u_curr, dt, antidiffusive_flux, equations)
    else # Otherwise, perform a bisection step
      dSdbeta = 0.0
    end

    if dSdbeta != 0.0
      # Update beta with Newton's method
      beta = beta - as / dSdbeta
    end

    # Check bounds
    if (beta < beta_L) || (beta > beta_R) || (dSdbeta == 0.0) || isnan(beta)
      # Out of bounds, do a bisection step
      beta = 0.5 * (beta_L + beta_R)
      # Get new u
      u_curr = u_safe + beta * dt * antidiffusive_flux

      # If the state is invalid, finish bisection step without checking tolerance and iterate further
      if !isValidState(u_curr, equations)
        beta_R = beta
        continue
      end

      # Check new beta for condition and update bounds
      as = goal_fct(bound, u_curr, equations)
      if initialCheck(bound, as, newton_abstol)
        beta_L = beta
      else
        beta_R = beta
      end
    else
      # Get new u
      u_curr = u_safe + beta * dt * antidiffusive_flux

      # If the state is invalid, redefine right bound without checking tolerance and iterate further
      if !isValidState(u_curr, equations)
        beta_R = beta
        continue
      end

      # Evaluate goal function
      as = goal_fct(bound, u_curr, equations)
    end

    # Check relative tolerance
    if abs(beta_old - beta) <= newton_reltol
      break
    end

    # Check absolute tolerance
    if finalCheck(bound, as, newton_abstol)
      break
    end
  end

  new_alpha = 1.0 - beta
  if alpha[i, j, element] > new_alpha + newton_abstol
    error("Alpha is getting smaller. old: $(alpha[i, j, element]), new: $new_alpha")
  else
    alpha[i, j, element] = new_alpha
  end

  return nothing
end

standard_finalCheck(bound, goal, newton_abstol) = abs(goal) < max(newton_abstol, abs(bound) * newton_abstol)


# this method is used when the indicator is constructed as for shock-capturing volume integrals
function create_cache(::Type{IndicatorMax}, equations::AbstractEquations{2}, basis::LobattoLegendreBasis)

  alpha = Vector{real(basis)}()

  A = Array{real(basis), ndims(equations)}
  indicator_threaded = [A(undef, nnodes(basis), nnodes(basis)) for _ in 1:Threads.nthreads()]

  return (; alpha, indicator_threaded)
end

# this method is used when the indicator is constructed as for AMR
function create_cache(typ::Type{IndicatorMax}, mesh, equations::AbstractEquations{2}, dg::DGSEM, cache)
  cache = create_cache(typ, equations, dg.basis)
end


function (indicator_max::IndicatorMax)(u::AbstractArray{<:Any,4},
                                       mesh, equations, dg::DGSEM, cache;
                                       kwargs...)
  @unpack alpha, indicator_threaded = indicator_max.cache
  resize!(alpha, nelements(dg, cache))

  @threaded for element in eachelement(dg, cache)
    indicator = indicator_threaded[Threads.threadid()]

    # Calculate indicator variables at Gauss-Lobatto nodes
    for j in eachnode(dg), i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, j, element)
      indicator[i, j] = indicator_max.variable(u_local, equations)
    end

    alpha[element] = maximum(indicator)
  end

  return alpha
end

# this method is used when the indicator is constructed as for shock-capturing volume integrals
# empty cache is default
function create_cache(::Type{IndicatorNeuralNetwork},
                      equations::AbstractEquations{2}, basis::LobattoLegendreBasis)
  return NamedTuple()
end

# cache for NeuralNetworkPerssonPeraire-type indicator
function create_cache(::Type{IndicatorNeuralNetwork{NeuralNetworkPerssonPeraire}},
                      equations::AbstractEquations{2}, basis::LobattoLegendreBasis)

  alpha = Vector{real(basis)}()
  alpha_tmp = similar(alpha)
  A = Array{real(basis), ndims(equations)}

  @assert nnodes(basis) >= 4 "Indicator only works for nnodes >= 4 (polydeg > 2)"

  prototype = A(undef, nnodes(basis), nnodes(basis))
  indicator_threaded  = [similar(prototype) for _ in 1:Threads.nthreads()]
  modal_threaded      = [similar(prototype) for _ in 1:Threads.nthreads()]
  modal_tmp1_threaded = [similar(prototype) for _ in 1:Threads.nthreads()]

  return (; alpha, alpha_tmp, indicator_threaded, modal_threaded, modal_tmp1_threaded)
end

# cache for NeuralNetworkRayHesthaven-type indicator
function create_cache(::Type{IndicatorNeuralNetwork{NeuralNetworkRayHesthaven}},
                      equations::AbstractEquations{2}, basis::LobattoLegendreBasis)

  alpha = Vector{real(basis)}()
  alpha_tmp = similar(alpha)
  A = Array{real(basis), ndims(equations)}

  prototype = A(undef, nnodes(basis), nnodes(basis))
  indicator_threaded  = [similar(prototype) for _ in 1:Threads.nthreads()]
  modal_threaded      = [similar(prototype) for _ in 1:Threads.nthreads()]
  modal_tmp1_threaded = [similar(prototype) for _ in 1:Threads.nthreads()]

  network_input = Vector{Float64}(undef, 15)
  neighbor_ids= Array{Int64}(undef, 8)
  neighbor_mean = Array{Float64}(undef, 4, 3)

  return (; alpha, alpha_tmp, indicator_threaded, modal_threaded, modal_tmp1_threaded,
            network_input, neighbor_ids, neighbor_mean)
end

# cache for NeuralNetworkCNN-type indicator
function create_cache(::Type{IndicatorNeuralNetwork{NeuralNetworkCNN}},
                      equations::AbstractEquations{2}, basis::LobattoLegendreBasis)

  alpha = Vector{real(basis)}()
  alpha_tmp = similar(alpha)
  A = Array{real(basis), ndims(equations)}

  prototype = A(undef, nnodes(basis), nnodes(basis))
  indicator_threaded  = [similar(prototype) for _ in 1:Threads.nthreads()]
  n_cnn = 4
  nodes,_ = gauss_lobatto_nodes_weights(nnodes(basis))
  cnn_nodes,_= gauss_lobatto_nodes_weights(n_cnn)
  vandermonde = polynomial_interpolation_matrix(nodes, cnn_nodes)
  network_input = Array{Float32}(undef, n_cnn, n_cnn, 1, 1)

  return (; alpha, alpha_tmp, indicator_threaded, nodes, cnn_nodes, vandermonde, network_input)
end

# this method is used when the indicator is constructed as for AMR
function create_cache(typ::Type{<:IndicatorNeuralNetwork},
                      mesh, equations::AbstractEquations{2}, dg::DGSEM, cache)
  create_cache(typ, equations, dg.basis)
end


function (indicator_ann::IndicatorNeuralNetwork{NeuralNetworkPerssonPeraire})(
    u, mesh::TreeMesh{2}, equations, dg::DGSEM, cache; kwargs...)

  @unpack indicator_type, alpha_max, alpha_min, alpha_smooth, alpha_continuous, alpha_amr, variable, network = indicator_ann

  @unpack alpha, alpha_tmp, indicator_threaded, modal_threaded, modal_tmp1_threaded = indicator_ann.cache
  # TODO: Taal refactor, when to `resize!` stuff changed possibly by AMR?
  #       Shall we implement `resize!(semi::AbstractSemidiscretization, new_size)`
  #       or just `resize!` whenever we call the relevant methods as we do now?
  resize!(alpha, nelements(dg, cache))
  if alpha_smooth
    resize!(alpha_tmp, nelements(dg, cache))
  end

  @threaded for element in eachelement(dg, cache)
    indicator  = indicator_threaded[Threads.threadid()]
    modal      = modal_threaded[Threads.threadid()]
    modal_tmp1 = modal_tmp1_threaded[Threads.threadid()]

    # Calculate indicator variables at Gauss-Lobatto nodes
    for j in eachnode(dg), i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, j, element)
      indicator[i, j] = indicator_ann.variable(u_local, equations)
    end

    # Convert to modal representation
    multiply_scalar_dimensionwise!(modal, dg.basis.inverse_vandermonde_legendre, indicator, modal_tmp1)

    # Calculate total energies for all modes, without highest, without two highest
    total_energy = zero(eltype(modal))
    for j in 1:nnodes(dg), i in 1:nnodes(dg)
      total_energy += modal[i, j]^2
    end
    total_energy_clip1 = zero(eltype(modal))
    for j in 1:(nnodes(dg)-1), i in 1:(nnodes(dg)-1)
      total_energy_clip1 += modal[i, j]^2
    end
    total_energy_clip2 = zero(eltype(modal))
    for j in 1:(nnodes(dg)-2), i in 1:(nnodes(dg)-2)
      total_energy_clip2 += modal[i, j]^2
    end
    total_energy_clip3 = zero(eltype(modal))
    for j in 1:(nnodes(dg)-3), i in 1:(nnodes(dg)-3)
      total_energy_clip3 += modal[i, j]^2
    end

    # Calculate energy in higher modes and polynomial degree for the network input
    X1 = (total_energy - total_energy_clip1)/total_energy
    X2 = (total_energy_clip1 - total_energy_clip2)/total_energy_clip1
    X3 = (total_energy_clip2 - total_energy_clip3)/total_energy_clip2
    X4 = nnodes(dg)
    network_input = SVector(X1, X2, X3, X4)

    # Scale input data
    network_input = network_input / max(maximum(abs, network_input), one(eltype(network_input)))
    probability_troubled_cell = network(network_input)[1]

    # Compute indicator value
    alpha[element] = probability_to_indicator(probability_troubled_cell, alpha_continuous,
                                              alpha_amr, alpha_min, alpha_max)
  end

  if alpha_smooth
    apply_smoothing!(mesh, alpha, alpha_tmp, dg, cache)
  end

  return alpha
end


function (indicator_ann::IndicatorNeuralNetwork{NeuralNetworkRayHesthaven})(
    u, mesh::TreeMesh{2}, equations, dg::DGSEM, cache; kwargs...)

  @unpack indicator_type, alpha_max, alpha_min, alpha_smooth, alpha_continuous, alpha_amr, variable, network = indicator_ann

  @unpack alpha, alpha_tmp, indicator_threaded, modal_threaded, modal_tmp1_threaded, network_input, neighbor_ids, neighbor_mean = indicator_ann.cache #X, network_input
  # TODO: Taal refactor, when to `resize!` stuff changed possibly by AMR?
  #       Shall we implement `resize!(semi::AbstractSemidiscretization, new_size)`
  #       or just `resize!` whenever we call the relevant methods as we do now?
  resize!(alpha, nelements(dg, cache))
  if alpha_smooth
    resize!(alpha_tmp, nelements(dg, cache))
  end

  c2e = zeros(Int, length(mesh.tree))
  for element in eachelement(dg, cache)
    c2e[cache.elements.cell_ids[element]] = element
  end

  X = Array{Float64}(undef, 3, nelements(dg, cache))

  @threaded for element in eachelement(dg, cache)
    indicator  = indicator_threaded[Threads.threadid()]
    modal      = modal_threaded[Threads.threadid()]
    modal_tmp1 = modal_tmp1_threaded[Threads.threadid()]

    # Calculate indicator variables at Gauss-Lobatto nodes
    for j in eachnode(dg), i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, j, element)
      indicator[i, j] = indicator_ann.variable(u_local, equations)
    end

    # Convert to modal representation
    multiply_scalar_dimensionwise!(modal, dg.basis.inverse_vandermonde_legendre, indicator, modal_tmp1)
    # Save linear modal coefficients for the network input
    X[1,element] = modal[1,1]
    X[2,element] = modal[1,2]
    X[3,element] = modal[2,1]
  end

  @threaded for element in eachelement(dg, cache)
    cell_id = cache.elements.cell_ids[element]

    network_input[1] = X[1,element]
    network_input[2] = X[2,element]
    network_input[3] = X[3,element]

    for direction in eachdirection(mesh.tree)
      if direction == 1 # -x
          dir = 4
      elseif direction == 2 # +x
          dir = 1
      elseif direction == 3 # -y
          dir = 3
      elseif direction == 4 # +y
          dir = 2
      end

      # Of no neighbor exists and current cell is not small
      if !has_any_neighbor(mesh.tree, cell_id, direction)
        network_input[3*dir+1] = X[1, element]
        network_input[3*dir+2] = X[2, element]
        network_input[3*dir+3] = X[3, element]
        continue
      end

      # Get Input data from neighbors
      if has_neighbor(mesh.tree, cell_id, direction)
        neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
        if has_children(mesh.tree, neighbor_cell_id) # Cell has small neighbor
          # Mean over 4 neighbor cells
          neighbor_ids[1] = mesh.tree.child_ids[1, neighbor_cell_id]
          neighbor_ids[2] = mesh.tree.child_ids[2, neighbor_cell_id]
          neighbor_ids[3] = mesh.tree.child_ids[3, neighbor_cell_id]
          neighbor_ids[4] = mesh.tree.child_ids[4, neighbor_cell_id]

          for i in 1:4
            if has_children(mesh.tree, neighbor_ids[i])
              neighbor_ids5 = c2e[mesh.tree.child_ids[1, neighbor_ids[i]]]
              neighbor_ids6 = c2e[mesh.tree.child_ids[2, neighbor_ids[i]]]
              neighbor_ids7 = c2e[mesh.tree.child_ids[3, neighbor_ids[i]]]
              neighbor_ids8 = c2e[mesh.tree.child_ids[4, neighbor_ids[i]]]

              neighbor_mean[i,1] = (X[1,neighbor_ids5] + X[1,neighbor_ids6] + X[1,neighbor_ids7] + X[1,neighbor_ids8])/4
              neighbor_mean[i,2] = (X[2,neighbor_ids5] + X[2,neighbor_ids6] + X[2,neighbor_ids7] + X[2,neighbor_ids8])/4
              neighbor_mean[i,3] = (X[3,neighbor_ids5] + X[3,neighbor_ids6] + X[3,neighbor_ids7] + X[3,neighbor_ids8])/4
            else
              neighbor_id = c2e[neighbor_ids[i]]
              neighbor_mean[i,1] = X[1,neighbor_id]
              neighbor_mean[i,2] = X[2,neighbor_id]
              neighbor_mean[i,3] = X[3,neighbor_id]
            end
          end
          network_input[3*dir+1] = (neighbor_mean[1,1] + neighbor_mean[2,1] + neighbor_mean[3,1] + neighbor_mean[4,1])/4
          network_input[3*dir+2] = (neighbor_mean[1,2] + neighbor_mean[2,2] + neighbor_mean[3,2] + neighbor_mean[4,2])/4
          network_input[3*dir+3] = (neighbor_mean[1,3] + neighbor_mean[2,3] + neighbor_mean[3,3] + neighbor_mean[4,3])/4

        else # Cell has same refinement level neighbor
          neighbor_id = c2e[neighbor_cell_id]
          network_input[3*dir+1] = X[1,neighbor_id]
          network_input[3*dir+2] = X[2,neighbor_id]
          network_input[3*dir+3] = X[3,neighbor_id]
        end
      else # Cell is small and has large neighbor
        parent_id = mesh.tree.parent_ids[cell_id]
        neighbor_id = c2e[mesh.tree.neighbor_ids[direction, parent_id]]

        network_input[3*dir+1] = X[1,neighbor_id]
        network_input[3*dir+2] = X[2,neighbor_id]
        network_input[3*dir+3] = X[3,neighbor_id]
      end
    end

    # Scale input data
    network_input = network_input / max(maximum(abs, network_input), one(eltype(network_input)))
    probability_troubled_cell = network(network_input)[1]

    # Compute indicator value
    alpha[element] = probability_to_indicator(probability_troubled_cell, alpha_continuous,
                                              alpha_amr, alpha_min, alpha_max)
  end

  if alpha_smooth
    apply_smoothing!(mesh, alpha, alpha_tmp, dg, cache)
  end

  return alpha
end


function (indicator_ann::IndicatorNeuralNetwork{NeuralNetworkCNN})(
    u, mesh::TreeMesh{2}, equations, dg::DGSEM, cache; kwargs...)
  @unpack indicator_type, alpha_max, alpha_min, alpha_smooth, alpha_continuous, alpha_amr, variable, network = indicator_ann

  @unpack alpha, alpha_tmp, indicator_threaded, nodes, cnn_nodes, vandermonde, network_input = indicator_ann.cache
  # TODO: Taal refactor, when to `resize!` stuff changed possibly by AMR?
  #       Shall we implement `resize!(semi::AbstractSemidiscretization, new_size)`
  #       or just `resize!` whenever we call the relevant methods as we do now?
  resize!(alpha, nelements(dg, cache))
  if alpha_smooth
    resize!(alpha_tmp, nelements(dg, cache))
  end

  @threaded for element in eachelement(dg, cache)
    indicator  = indicator_threaded[Threads.threadid()]

    # Calculate indicator variables at Gauss-Lobatto nodes
    for j in eachnode(dg), i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, j, element)
      indicator[i, j] = indicator_ann.variable(u_local, equations)
    end

    # Interpolate nodal data to 4x4 LGL nodes
    for j in 1:4, i in 1:4
      acc = zero(eltype(indicator))
      for jj in eachnode(dg), ii in eachnode(dg)
        acc += vandermonde[i,ii] * indicator[ii,jj] * vandermonde[j,jj]
      end
      network_input[i,j,1,1] = acc
    end

    # Scale input data
    network_input = network_input / max(maximum(abs, network_input), one(eltype(network_input)))
    probability_troubled_cell = network(network_input)[1]

    # Compute indicator value
    alpha[element] = probability_to_indicator(probability_troubled_cell, alpha_continuous,
                                              alpha_amr, alpha_min, alpha_max)
  end

  if alpha_smooth
    apply_smoothing!(mesh, alpha, alpha_tmp, dg, cache)
  end

  return alpha
end

end # @muladd
