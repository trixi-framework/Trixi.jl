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


# Use this function barrier and unpack inside to avoid passing closures to Polyester.jl
# with @batch (@threaded).
# Otherwise, @threaded does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function calc_indicator_hennemann_gassner!(indicator_hg, threshold, parameter_s, u,
                                                   element, mesh::AbstractMesh{2},
                                                   equations, dg, cache)
  @unpack alpha_max, alpha_min, alpha_smooth, variable = indicator_hg
  @unpack alpha, alpha_tmp, indicator_threaded, modal_threaded,
          modal_tmp1_threaded = indicator_hg.cache

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
function create_cache(indicator::Type{IndicatorIDP}, equations::AbstractEquations{2}, basis::LobattoLegendreBasis, number_bounds, bar_states)
  container_shock_capturing = Trixi.ContainerShockCapturingIndicatorIDP2D{real(basis)}(0, nnodes(basis), number_bounds)

  cache = (; )
  if bar_states
    container_bar_states = Trixi.ContainerBarStates{real(basis)}(0, nvariables(equations), nnodes(basis))
    cache = (; cache..., container_bar_states)
  end

  idp_bounds_delta = zeros(real(basis), number_bounds)

  return (; cache..., container_shock_capturing, idp_bounds_delta)
end

function (indicator::IndicatorIDP)(u::AbstractArray{<:Any,4}, semi, dg::DGSEM, t, dt; kwargs...)
  @unpack alpha = indicator.cache.container_shock_capturing
  alpha .= zero(eltype(alpha))
  if indicator.smoothness_indicator
    elements = semi.cache.element_ids_dgfv
  else
    elements = eachelement(dg, semi.cache)
  end

  if indicator.density_tvd
    @trixi_timeit timer() "density_tvd"  idp_density_tvd!( alpha, indicator, u, t, dt, semi, elements)
  end
  if indicator.positivity
    @trixi_timeit timer() "positivity"   idp_positivity!( alpha, indicator, u,    dt, semi, elements)
  end
  if indicator.spec_entropy
    @trixi_timeit timer() "spec_entropy" idp_spec_entropy!(alpha, indicator, u, t, dt, semi, elements)
  end
  if indicator.math_entropy
    @trixi_timeit timer() "math_entropy" idp_math_entropy!(alpha, indicator, u, t, dt, semi, elements)
  end

  # Calculate alpha1 and alpha2
  @unpack alpha1, alpha2 = indicator.cache.container_shock_capturing
  @threaded for element in elements
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

@inline function calc_bounds_2sided!(var_min, var_max, variable, u, t, semi)
  mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
  # Calc bounds inside elements
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
  calc_bounds_2sided_interface!(var_min, var_max, variable, u, t, semi, mesh)
end

@inline function calc_bounds_2sided_interface!(var_min, var_max, variable, u, t, semi, mesh::TreeMesh2D)
  _, equations, dg, cache = mesh_equations_solver_cache(semi)
  @unpack boundary_conditions = semi
  # Calc bounds at interfaces and periodic boundaries
  for interface in eachinterface(dg, cache)
    # Get neighboring element ids
    left  = cache.interfaces.neighbor_ids[1, interface]
    right = cache.interfaces.neighbor_ids[2, interface]

    orientation = cache.interfaces.orientations[interface]

    for i in eachnode(dg)
      index_left  = (nnodes(dg), i)
      index_right = (1, i)
      if orientation == 2
        index_left = reverse(index_left)
        index_right = reverse(index_right)
      end
      var_left  = variable(get_node_vars(u, equations, dg, index_left...,  left),  equations)
      var_right = variable(get_node_vars(u, equations, dg, index_right..., right), equations)

      var_min[index_right..., right] = min(var_min[index_right..., right], var_left)
      var_max[index_right..., right] = max(var_max[index_right..., right], var_left)

      var_min[index_left..., left] = min(var_min[index_left..., left], var_right)
      var_max[index_left..., left] = max(var_max[index_left..., left], var_right)
    end
  end

  # Calc bounds at physical boundaries
  for boundary in eachboundary(dg, cache)
    element = cache.boundaries.neighbor_ids[boundary]
    orientation = cache.boundaries.orientations[boundary]
    neighbor_side = cache.boundaries.neighbor_sides[boundary]

    for i in eachnode(dg)
      if neighbor_side == 2 # Element is on the right, boundary on the left
        index = (1, i)
        boundary_index = 1
      else # Element is on the left, boundary on the right
        index = (nnodes(dg), i)
        boundary_index = 2
      end
      if orientation == 2
        index = reverse(index)
        boundary_index += 2
      end
      u_inner = get_node_vars(u, equations, dg, index..., element)
      u_outer = get_boundary_outer_state(u_inner, cache, t, boundary_conditions[boundary_index], orientation, boundary_index,
                                         equations, dg, index..., element)
      var_outer = variable(u_outer, equations)

      var_min[index..., element] = min(var_min[index..., element], var_outer)
      var_max[index..., element] = max(var_max[index..., element], var_outer)
    end
  end

  return nothing
end


@inline function calc_bounds_1sided!(var_minmax, minmax, typeminmax, variable, u, t, semi)
  mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
  # Calc bounds inside elements
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
  calc_bounds_1sided_interface!(var_minmax, minmax, variable, u, t, semi, mesh)
end

@inline function calc_bounds_1sided_interface!(var_minmax, minmax, variable, u, t, semi, mesh::TreeMesh2D)
  _, equations, dg, cache = mesh_equations_solver_cache(semi)
  @unpack boundary_conditions = semi
  # Calc bounds at interfaces and periodic boundaries
  for interface in eachinterface(dg, cache)
    # Get neighboring element ids
    left  = cache.interfaces.neighbor_ids[1, interface]
    right = cache.interfaces.neighbor_ids[2, interface]

    orientation = cache.interfaces.orientations[interface]

    if orientation == 1
      for j in eachnode(dg)
        var_left  = variable(get_node_vars(u, equations, dg, nnodes(dg), j, left),  equations)
        var_right = variable(get_node_vars(u, equations, dg, 1,          j, right), equations)

        var_minmax[1,          j, right] = minmax(var_minmax[1,          j, right], var_left)
        var_minmax[nnodes(dg), j, left]  = minmax(var_minmax[nnodes(dg), j, left],  var_right)
      end
    else # orientation == 2
      for i in eachnode(dg)
        var_left  = variable(get_node_vars(u, equations, dg, i, nnodes(dg), left),  equations)
        var_right = variable(get_node_vars(u, equations, dg, i,          1, right), equations)

        var_minmax[i,          1, right] = minmax(var_minmax[i,          1, right], var_left)
        var_minmax[i, nnodes(dg), left]  = minmax(var_minmax[i, nnodes(dg), left],  var_right)
      end
    end
  end

  # Calc bounds at physical boundaries
  for boundary in eachboundary(dg, cache)
    element = cache.boundaries.neighbor_ids[boundary]
    orientation = cache.boundaries.orientations[boundary]
    neighbor_side = cache.boundaries.neighbor_sides[boundary]

    if orientation == 1
      if neighbor_side == 2 # Element is on the right, boundary on the left
        for j in eachnode(dg)
          u_inner = get_node_vars(u, equations, dg, 1, j, element)
          u_outer = get_boundary_outer_state(u_inner, cache, t, boundary_conditions[1], orientation, 1,
                                             equations, dg, 1, j, element)
          var_outer = variable(u_outer, equations)

          var_minmax[1, j, element] = minmax(var_minmax[1, j, element], var_outer)
        end
      else # Element is on the left, boundary on the right
        for j in eachnode(dg)
          u_inner = get_node_vars(u, equations, dg, nnodes(dg), j, element)
          u_outer = get_boundary_outer_state(u_inner, cache, t, boundary_conditions[2], orientation, 2,
                                             equations, dg, nnodes(dg), j, element)
          var_outer = variable(u_outer, equations)

          var_minmax[nnodes(dg), j, element] = minmax(var_minmax[nnodes(dg), j, element], var_outer)
        end
      end
    else # orientation == 2
      if neighbor_side == 2 # Element is on the right, boundary on the left
        for i in eachnode(dg)
          u_inner = get_node_vars(u, equations, dg, i, 1, element)
          u_outer = get_boundary_outer_state(u_inner, cache, t, boundary_conditions[3], orientation, 3,
                                             equations, dg, i, 1, element)
          var_outer = variable(u_outer, equations)

          var_minmax[i, 1, element] = minmax(var_minmax[i, 1, element], var_outer)
        end
      else # Element is on the left, boundary on the right
        for i in eachnode(dg)
          u_inner = get_node_vars(u, equations, dg, i, nnodes(dg), element)
          u_outer = get_boundary_outer_state(u_inner, cache, t, boundary_conditions[4], orientation, 4,
                                             equations, dg, i, nnodes(dg), element)
          var_outer = variable(u_outer, equations)

          var_minmax[i, nnodes(dg), element] = minmax(var_minmax[i, nnodes(dg), element], var_outer)
        end
      end
    end
  end

  return nothing
end

@inline function idp_density_tvd!(alpha, indicator, u, t, dt, semi, elements)
  mesh, _, dg, cache = mesh_equations_solver_cache(semi)
  @unpack boundary_conditions = semi
  @unpack variable_bounds = indicator.cache.container_shock_capturing

  rho_min = variable_bounds[1]
  rho_max = variable_bounds[2]
  if !indicator.bar_states
    calc_bounds_2sided!(rho_min, rho_max, density, u, t, semi)
  end

  @unpack antidiffusive_flux1, antidiffusive_flux2 = cache.container_antidiffusive_flux
  @unpack inverse_weights = dg.basis

  @threaded for element in elements
    if mesh isa TreeMesh
      inverse_jacobian = cache.elements.inverse_jacobian[element]
    end
    for j in eachnode(dg), i in eachnode(dg)
      if mesh isa StructuredMesh
        inverse_jacobian = cache.elements.inverse_jacobian[i, j, element]
      end
      rho = u[1, i, j, element]
      # Real Zalesak type limiter
      #   * Zalesak (1979). "Fully multidimensional flux-corrected transport algorithms for fluids"
      #   * Kuzmin et al. (2010). "Failsafe flux limiting and constrained data projections for equations of gas dynamics"
      #   Note: The Zalesak limiter has to be computed, even if the state is valid, because the correction is
      #         for each interface, not each node

      Qp = max(0, (rho_max[i, j, element] - rho) / dt)
      Qm = min(0, (rho_min[i, j, element] - rho) / dt)

      # Calculate Pp and Pm
      # Note: Boundaries of antidiffusive_flux1/2 are constant 0, so they make no difference here.
      val_flux1_local     =  inverse_weights[i] * antidiffusive_flux1[1,   i,   j, element]
      val_flux1_local_ip1 = -inverse_weights[i] * antidiffusive_flux1[1, i+1,   j, element]
      val_flux2_local     =  inverse_weights[j] * antidiffusive_flux2[1,   i,   j, element]
      val_flux2_local_jp1 = -inverse_weights[j] * antidiffusive_flux2[1,   i, j+1, element]

      Pp = max(0, val_flux1_local) + max(0, val_flux1_local_ip1) +
           max(0, val_flux2_local) + max(0, val_flux2_local_jp1)
      Pm = min(0, val_flux1_local) + min(0, val_flux1_local_ip1) +
           min(0, val_flux2_local) + min(0, val_flux2_local_jp1)
      Pp = inverse_jacobian * Pp
      Pm = inverse_jacobian * Pm

      # Compute blending coefficient avoiding division by zero
      # (as in paper of [Guermond, Nazarov, Popov, Thomas] (4.8))
      Qp = abs(Qp) / (abs(Pp) + eps(typeof(Qp)) * 100 * abs(rho_max[i, j, element]))
      Qm = abs(Qm) / (abs(Pm) + eps(typeof(Qm)) * 100 * abs(rho_max[i, j, element]))

      # Calculate alpha at nodes
      alpha[i, j, element] = 1 - min(1, Qp, Qm)
    end
  end

  return nothing
end

@inline function idp_spec_entropy!(alpha, indicator, u, t, dt, semi, elements)
  mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
  @unpack boundary_conditions = semi
  @unpack density_tvd, positivity = indicator
  @unpack variable_bounds = indicator.cache.container_shock_capturing

  s_min = variable_bounds[2 * density_tvd + 1]
  if !indicator.bar_states
    calc_bounds_1sided!(s_min, min, typemax, entropy_spec, u, t, semi)
  end

  # Perform Newton's bisection method to find new alpha
  @threaded for element in elements
    for j in eachnode(dg), i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, j, element)
      newton_loops_alpha!(alpha, s_min[i, j, element], u_local, i, j, element,
                          specEntropy_goal, specEntropy_dGoal_dbeta, specEntropy_initialCheck, standard_finalCheck,
                          dt, mesh, equations, dg, cache, indicator)
    end
  end

  return nothing
end

specEntropy_goal(bound, u, equations) = bound - entropy_spec(u, equations)
specEntropy_dGoal_dbeta(u, dt, antidiffusive_flux, equations) = -dot(cons2entropy_spec(u, equations), dt * antidiffusive_flux)
specEntropy_initialCheck(bound, goal, newton_abstol) = goal <= max(newton_abstol, abs(bound) * newton_abstol)

@inline function idp_math_entropy!(alpha, indicator, u, t, dt, semi, elements)
  mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
  @unpack boundary_conditions = semi
  @unpack density_tvd, positivity, spec_entropy = indicator
  @unpack variable_bounds = indicator.cache.container_shock_capturing

  s_max = variable_bounds[2 * density_tvd + spec_entropy + 1]
  if !indicator.bar_states
    calc_bounds_1sided!(s_max, max, typemin, entropy_math, u, t, semi)
  end

  # Perform Newton's bisection method to find new alpha
  @threaded for element in elements
    for j in eachnode(dg), i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, j, element)
      newton_loops_alpha!(alpha, s_max[i, j, element], u_local, i, j, element,
                          mathEntropy_goal, mathEntropy_dGoal_dbeta, mathEntropy_initialCheck, standard_finalCheck,
                          dt, mesh, equations, dg, cache, indicator)
    end
  end

  return nothing
end

mathEntropy_goal(bound, u, equations) = bound - entropy_math(u, equations)
mathEntropy_dGoal_dbeta(u, dt, antidiffusive_flux, equations) = -dot(cons2entropy(u, equations), dt * antidiffusive_flux)
mathEntropy_initialCheck(bound, goal, newton_abstol) = goal >= -max(newton_abstol, abs(bound) * newton_abstol)

@inline function idp_positivity!(alpha, indicator, u, dt, semi, elements)

  # Conservative variables
  for (index, variable) in enumerate(indicator.variables_cons)
    idp_positivity!(alpha, indicator, u, dt, semi, elements, variable, index)
  end

  # Nonlinear variables
  for (index, variable) in enumerate(indicator.variables_nonlinear)
    idp_positivity_newton!(alpha, indicator, u, dt, semi, elements, variable, index)
  end

  return nothing
end

@inline function idp_positivity!(alpha, indicator, u, dt, semi, elements, variable, index)
  mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
  @unpack antidiffusive_flux1, antidiffusive_flux2 = cache.container_antidiffusive_flux
  @unpack inverse_weights = dg.basis
  @unpack density_tvd, spec_entropy, math_entropy, positivity_correction_factor, variables_cons = indicator

  @unpack variable_bounds = indicator.cache.container_shock_capturing

  if 1 in variables_cons && density_tvd
    if variables_cons[variable] == 1
      var_min = variable_bounds[1]
    else
      var_min = variable_bounds[2 * density_tvd + spec_entropy + math_entropy + index - 1]
    end
  else
    var_min = variable_bounds[2 * density_tvd + spec_entropy + math_entropy + index]
  end

  @threaded for element in elements
    if mesh isa TreeMesh
      inverse_jacobian = cache.elements.inverse_jacobian[element]
    end
    for j in eachnode(dg), i in eachnode(dg)
      if mesh isa StructuredMesh
        inverse_jacobian = cache.elements.inverse_jacobian[i, j, element]
      end

      var = get_node_vars(u, equations, dg, i, j, element)[variable]
      if var < 0
        error("Safe $variable is not safe. element=$element, node: $i $j, value=$var")
      end

      # Compute bound
      if indicator.density_tvd
        var_min[i, j, element] = max(var_min[i, j, element], positivity_correction_factor * var)
      else
        var_min[i, j, element] = positivity_correction_factor * var
      end

      # Real one-sided Zalesak-type limiter
      # * Zalesak (1979). "Fully multidimensional flux-corrected transport algorithms for fluids"
      # * Kuzmin et al. (2010). "Failsafe flux limiting and constrained data projections for equations of gas dynamics"
      # Note: The Zalesak limiter has to be computed, even if the state is valid, because the correction is
      #       for each interface, not each node
      Qm = min(0, (var_min[i, j, element] - var) / dt)

      # Calculate Pm
      # Note: Boundaries of antidiffusive_flux1/2 are constant 0, so they make no difference here.
      val_flux1_local     =  inverse_weights[i] * get_node_vars(antidiffusive_flux1, equations, dg,   i,   j, element)[variable]
      val_flux1_local_ip1 = -inverse_weights[i] * get_node_vars(antidiffusive_flux1, equations, dg, i+1,   j, element)[variable]
      val_flux2_local     =  inverse_weights[j] * get_node_vars(antidiffusive_flux2, equations, dg,   i,   j, element)[variable]
      val_flux2_local_jp1 = -inverse_weights[j] * get_node_vars(antidiffusive_flux2, equations, dg,   i, j+1, element)[variable]

      Pm = min(0, val_flux1_local) + min(0, val_flux1_local_ip1) +
           min(0, val_flux2_local) + min(0, val_flux2_local_jp1)
      Pm = inverse_jacobian * Pm

      # Compute blending coefficient avoiding division by zero
      # (as in paper of [Guermond, Nazarov, Popov, Thomas] (4.8))
      Qm = abs(Qm) / (abs(Pm) + eps(typeof(Qm)) * 100)

      # Calculate alpha
      alpha[i, j, element]  = max(alpha[i, j, element], 1 - Qm)
    end
  end

  return nothing
end


@inline function idp_positivity_newton!(alpha, indicator, u, dt, semi, elements, variable, index)
  mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
  @unpack density_tvd, spec_entropy, math_entropy, positivity_correction_factor, variables_cons, variables_nonlinear = indicator

  @unpack variable_bounds = indicator.cache.container_shock_capturing

  var_min = variable_bounds[2 * density_tvd + spec_entropy + math_entropy +
                            length(variables_cons) - min(density_tvd, 1 in variables_cons) +
                            index]

  @threaded for element in elements
    for j in eachnode(dg), i in eachnode(dg)
      # Compute bound
      u_local = get_node_vars(u, equations, dg, i, j, element)
      var = variable(u_local, equations)
      if var < 0
        error("Safe $variable is not safe. element=$element, node: $i $j, value=$var")
      end
      var_min[i, j, element] = positivity_correction_factor * var

      # Perform Newton's bisection method to find new alpha
      newton_loops_alpha!(alpha, var_min[i, j, element], u_local, i, j, element,
                          pressure_goal, pressure_dgoal_dbeta, pressure_initialCheck, pressure_finalCheck,
                          dt, mesh, equations, dg, cache, indicator)
    end
  end

  return nothing
end

pressure_goal(bound, u, equations) = bound - pressure(u, equations)
pressure_dgoal_dbeta(u, dt, antidiffusive_flux, equations) = -dot(dpdu(u, equations), dt * antidiffusive_flux)
pressure_initialCheck(bound, goal, newton_abstol) = goal <= 0
pressure_finalCheck(bound, goal, newton_abstol) = (goal <= eps()) && (goal > -max(newton_abstol, abs(bound) * newton_abstol))

@inline function newton_loops_alpha!(alpha, bound, u, i, j, element,
                                     goal_fct, dgoal_fct, initialCheck, finalCheck,
                                     dt, mesh, equations, dg, cache, indicator)
  @unpack inverse_weights = dg.basis
  @unpack antidiffusive_flux1, antidiffusive_flux2 = cache.container_antidiffusive_flux
  if mesh isa TreeMesh
    inverse_jacobian = cache.elements.inverse_jacobian[element]
  else # mesh isa StructuredMesh
    inverse_jacobian = cache.elements.inverse_jacobian[i, j, element]
  end

  @unpack gamma_constant_newton = indicator

  # negative xi direction
  antidiffusive_flux = gamma_constant_newton * inverse_jacobian * inverse_weights[i] * get_node_vars(antidiffusive_flux1, equations, dg, i, j, element)
  newton_loop!(alpha, bound, u, i, j, element, goal_fct, dgoal_fct, initialCheck, finalCheck, equations, dt, indicator, antidiffusive_flux)

  # positive xi direction
  antidiffusive_flux = -gamma_constant_newton * inverse_jacobian * inverse_weights[i] * get_node_vars(antidiffusive_flux1, equations, dg, i+1, j, element)
  newton_loop!(alpha, bound, u, i, j, element, goal_fct, dgoal_fct, initialCheck, finalCheck, equations, dt, indicator, antidiffusive_flux)

  # negative eta direction
  antidiffusive_flux = gamma_constant_newton * inverse_jacobian * inverse_weights[j] * get_node_vars(antidiffusive_flux2, equations, dg, i, j, element)
  newton_loop!(alpha, bound, u, i, j, element, goal_fct, dgoal_fct, initialCheck, finalCheck, equations, dt, indicator, antidiffusive_flux)

  # positive eta direction
  antidiffusive_flux = -gamma_constant_newton * inverse_jacobian * inverse_weights[j] * get_node_vars(antidiffusive_flux2, equations, dg, i, j+1, element)
  newton_loop!(alpha, bound, u, i, j, element, goal_fct, dgoal_fct, initialCheck, finalCheck, equations, dt, indicator, antidiffusive_flux)

  return nothing
end

@inline function newton_loop!(alpha, bound, u, i, j, element,
                              goal_fct, dgoal_fct, initialCheck, finalCheck,
                              equations, dt, indicator, antidiffusive_flux)
  newton_reltol, newton_abstol = indicator.newton_tolerances

  beta = 1 - alpha[i, j, element]

  beta_L = 0 # alpha = 1
  beta_R = beta # No higher beta (lower alpha) than the current one

  u_curr = u + beta * dt * antidiffusive_flux

  # If state is valid, perform initial check and return if correction is not needed
  if isValidState(u_curr, equations)
    as = goal_fct(bound, u_curr, equations)

    initialCheck(bound, as, newton_abstol) && return nothing
  end

  # Newton iterations
  for iter in 1:indicator.max_iterations_newton
    beta_old = beta

    # If the state is valid, evaluate d(goal)/d(beta)
    if isValidState(u_curr, equations)
      dSdbeta = dgoal_fct(u_curr, dt, antidiffusive_flux, equations)
    else # Otherwise, perform a bisection step
      dSdbeta = 0
    end

    if dSdbeta != 0
      # Update beta with Newton's method
      beta = beta - as / dSdbeta
    end

    # Check bounds
    if (beta < beta_L) || (beta > beta_R) || (dSdbeta == 0) || isnan(beta)
      # Out of bounds, do a bisection step
      beta = 0.5 * (beta_L + beta_R)
      # Get new u
      u_curr = u + beta * dt * antidiffusive_flux

      # If the state is invalid, finish bisection step without checking tolerance and iterate further
      if !isValidState(u_curr, equations)
        beta_R = beta
        continue
      end

      # Check new beta for condition and update bounds
      as = goal_fct(bound, u_curr, equations)
      if initialCheck(bound, as, newton_abstol)
        # New beta fulfills condition
        beta_L = beta
      else
        # New beta does not fulfill condition
        beta_R = beta
      end
    else
      # Get new u
      u_curr = u + beta * dt * antidiffusive_flux

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

  new_alpha = 1 - beta
  if alpha[i, j, element] > new_alpha + newton_abstol
    error("Alpha is getting smaller. old: $(alpha[i, j, element]), new: $new_alpha")
  else
    alpha[i, j, element] = new_alpha
  end

  return nothing
end

standard_finalCheck(bound, goal, newton_abstol) = abs(goal) < max(newton_abstol, abs(bound) * newton_abstol)


# this method is used when the indicator is constructed as for shock-capturing volume integrals
function create_cache(indicator::Type{IndicatorMCL}, equations::AbstractEquations{2},
                      basis::LobattoLegendreBasis, PressurePositivityLimiterKuzmin)
  container_shock_capturing = Trixi.ContainerShockCapturingIndicatorMCL2D{real(basis)}(0, nvariables(equations), nnodes(basis))
  container_bar_states = Trixi.ContainerBarStates{real(basis)}(0, nvariables(equations), nnodes(basis))

  idp_bounds_delta = zeros(real(basis), 2, nvariables(equations) + PressurePositivityLimiterKuzmin)

  return (; container_shock_capturing, container_bar_states, idp_bounds_delta)
end


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
