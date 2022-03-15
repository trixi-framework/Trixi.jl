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
function create_cache(::Type{IndicatorIDP}, equations::AbstractEquations{2}, basis::LobattoLegendreBasis)

  A = Array{real(basis), ndims(equations)}
  indicator_threaded = [A(undef, nnodes(basis), nnodes(basis)) for _ in 1:Threads.nthreads()]

  ContainerShockCapturingIndicator = Trixi.ContainerShockCapturingIndicator{real(basis)}(0, nnodes(basis))

  # TODO: Nicer way to set a length?
  alpha_max_per_timestep  = zeros(real(basis), 200)
  alpha_mean_per_timestep = zeros(real(basis), 200)

  return (; indicator_threaded,
            ContainerShockCapturingIndicator,
            alpha_max_per_timestep, alpha_mean_per_timestep)
end

function (indicator_IDP::IndicatorIDP)(u::AbstractArray{<:Any,4}, u_old::AbstractArray{<:Any,4},
                                       mesh, equations, dg::DGSEM,
                                       dt, cache;
                                       kwargs...)
  @unpack indicator_threaded = indicator_IDP.cache
  @unpack antidiffusive_flux1, antidiffusive_flux2 = cache.ContainerFCT2D

  @unpack alpha, alpha1, alpha2, var_max, var_min, alpha_max_per_element, alpha_mean_per_element = indicator_IDP.cache.ContainerShockCapturingIndicator

  @unpack inverse_weights = dg.basis

  @unpack alpha_maxIDP = indicator_IDP

  @threaded for element in eachelement(dg, cache)

    # Calculate indicator variables at Gauss-Lobatto nodes
    indicator = indicator_threaded[Threads.threadid()]
    for j in eachnode(dg), i in eachnode(dg)
      u_local = get_node_vars(u_old, equations, dg, i, j, element)
      indicator[i, j] = indicator_IDP.variable(u_local, equations)
    end

    for j in eachnode(dg), i in eachnode(dg)
      # Calculate max and min of variable at Gauss-Lobatto nodes
      var_min[i, j, element] = indicator[i, j]
      var_max[i, j, element] = indicator[i, j]
      if i > 1
        var_min[i, j, element] = min(var_min[i, j, element], indicator[i-1, j])
        var_max[i, j, element] = max(var_max[i, j, element], indicator[i-1, j])
      end
      if i < nnodes(dg)
        var_min[i, j, element] = min(var_min[i, j, element], indicator[i+1, j])
        var_max[i, j, element] = max(var_max[i, j, element], indicator[i+1, j])
      end
      if j > 1
        var_min[i, j, element] = min(var_min[i, j, element], indicator[i, j-1])
        var_max[i, j, element] = max(var_max[i, j, element], indicator[i, j-1])
      end
      if j < nnodes(dg)
        var_min[i, j, element] = min(var_min[i, j, element], indicator[i, j+1])
        var_max[i, j, element] = max(var_max[i, j, element], indicator[i, j+1])
      end
    end
  end

  # Loop over interfaces
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
      u_local_left  = get_node_vars(u_old, equations, dg, index_left...)
      u_local_right = get_node_vars(u_old, equations, dg, index_right...)
      var_neighbor_left  = indicator_IDP.variable(u_local_left,  equations)
      var_neighbor_right = indicator_IDP.variable(u_local_right, equations)

      var_min[index_right...] = min(var_min[index_right...], var_neighbor_left)
      var_max[index_right...] = max(var_max[index_right...], var_neighbor_left)

      var_min[index_left...] = min(var_min[index_left...], var_neighbor_right)
      var_max[index_left...] = max(var_max[index_left...], var_neighbor_right)
    end
  end

  # Loop over L2 mortars
  for mortar in eachmortar(dg, cache)
    # Get neighboring element ids
    lower = cache.mortars.neighbor_ids[1, mortar]
    upper = cache.mortars.neighbor_ids[2, mortar]
    large = cache.mortars.neighbor_ids[3, mortar]

    # Interpolate element face data to adjacent interface locations and use for var_max/min
    if cache.mortars.large_sides[mortar] == 1 # -> large element on left side
      if cache.mortars.orientations[mortar] == 1
        # L2 mortars in x-direction
        index_large = i -> (nnodes(dg), i)
        index_small = i -> (1, i)
        element_solutions_to_mortars!(u_old, indicator_IDP, dg, equations,
                                      large, upper, lower, index_large, index_small)
      else
        # L2 mortars in y-direction
        index_large = i -> (i, nnodes(dg))
        index_small = i -> (i, 1)
        element_solutions_to_mortars!(u_old, indicator_IDP, dg, equations,
                                      large, upper, lower, index_large, index_small)
      end
    else # large_sides[mortar] == 2 -> large element on right side
      if cache.mortars.orientations[mortar] == 1
        # L2 mortars in x-direction
        index_large = i -> (1, i)
        index_small = i -> (nnodes(dg), i)
        element_solutions_to_mortars!(u_old, indicator_IDP, dg, equations,
                                      large, upper, lower, index_large, index_small)
      else
        # L2 mortars in y-direction
        index_large = i -> (i, 1)
        index_small = i -> (i, nnodes(dg))
        element_solutions_to_mortars!(u_old, indicator_IDP, dg, equations,
                                      large, upper, lower, index_large, index_small)
      end
    end
  end

  @threaded for element in eachelement(dg, cache)
    inverse_jacobian = cache.elements.inverse_jacobian[element]

    for j in eachnode(dg), i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, j, element)
      var = indicator_IDP.variable(u_local, equations)
      if abs(var_max[i, j, element] - var) < sqrt(eps()) || abs(var_min[i, j, element] - var) < sqrt(eps())
        alpha[i, j, element] = 0.0
      else
        # Calculate P_plus and P_minus
        # Note: Boundaries of antidiffusive_flux1/2 are constant 0, so they make no difference here.
        val_flux1_local     = indicator_IDP.variable( dt * inverse_jacobian * inverse_weights[i] * get_node_vars(antidiffusive_flux1, equations, dg, i,   j,   element), equations)
        val_flux1_local_ip1 = indicator_IDP.variable(-dt * inverse_jacobian * inverse_weights[i] * get_node_vars(antidiffusive_flux1, equations, dg, i+1, j,   element), equations)
        val_flux2_local     = indicator_IDP.variable( dt * inverse_jacobian * inverse_weights[j] * get_node_vars(antidiffusive_flux2, equations, dg, i,   j,   element), equations)
        val_flux2_local_jp1 = indicator_IDP.variable(-dt * inverse_jacobian * inverse_weights[j] * get_node_vars(antidiffusive_flux2, equations, dg, i,   j+1, element), equations)

        P_plus  = max(0.0, val_flux1_local) + max(0.0, val_flux1_local_ip1) +
                  max(0.0, val_flux2_local) + max(0.0, val_flux2_local_jp1)
        P_minus = min(0.0, val_flux1_local) + min(0.0, val_flux1_local_ip1) +
                  min(0.0, val_flux2_local) + min(0.0, val_flux2_local_jp1)

        # Calculate alpha_plus and alpha_minus
        frac_plus  = (var_max[i, j, element] - var) / P_plus
        frac_minus = (var_min[i, j, element] - var) / P_minus

        alpha_plus  = 1 - min(1.0, max(0.0, frac_plus))
        alpha_minus = 1 - min(1.0, max(0.0, frac_minus))

        # Calculate alpha at nodes
        alpha[i, j, element] = max(alpha_plus, alpha_minus)

        # Clip the maximum amount of FV allowed
        alpha[i, j, element] = min(alpha_maxIDP, alpha[i, j, element])
      end

      # Calculate maximum and mean alpha per element
      alpha_max_per_element[element] = max(alpha_max_per_element[element], alpha[i, j, element])
      alpha_mean_per_element[element] += 1/3 * 1/(nnodes(dg)^2) * alpha[i, j, element]
    end

    # Calculate alpha1 and alpha2
    for j in eachnode(dg), i in 2:nnodes(dg)
      alpha1[i, j, element] = max(alpha[i-1, j, element], alpha[i, j, element])
    end
    for j in 2:nnodes(dg), i in eachnode(dg)
      alpha2[i, j, element] = max(alpha[i, j-1, element], alpha[i, j, element])
    end
    alpha1[1, :, element] .= zero(eltype(alpha1))
    alpha1[nnodes(dg)+1, :, element] .= zero(eltype(alpha1))
    alpha2[:, 1, element] .= zero(eltype(alpha2))
    alpha2[:, nnodes(dg)+1, element] .= zero(eltype(alpha2))
  end

  return nothing
end


@inline function element_solutions_to_mortars!(u_old::AbstractArray{<:Any,4}, indicator_IDP, dg, equations,
                                               large, upper, lower,
                                               index_large, index_small)

  @unpack var_max, var_min = indicator_IDP.cache.ContainerShockCapturingIndicator

  u_tmp_upper  = similar(view(u_old, :, 1, :, large))
  u_tmp_lower  = similar(u_tmp_upper)
  u_tmp_large1 = similar(u_tmp_upper)
  u_tmp_large2 = similar(u_tmp_upper)

  u_large = view(u_old, :, index_large(:)..., large)
  u_upper = view(u_old, :, index_small(:)..., upper)
  u_lower = view(u_old, :, index_small(:)..., lower)

  multiply_dimensionwise!(u_tmp_upper, dg.mortar.forward_upper, u_large)
  multiply_dimensionwise!(u_tmp_lower, dg.mortar.forward_lower, u_large)

  multiply_dimensionwise!(u_tmp_large1, dg.mortar.reverse_upper, u_upper)
  multiply_dimensionwise!(u_tmp_large2, dg.mortar.reverse_lower, u_lower)

  for i in eachnode(dg)
    # large to small
    var_min[index_small(i)..., upper] = min(var_min[index_small(i)..., upper], indicator_IDP.variable(view(u_tmp_upper, :, i), equations))
    var_max[index_small(i)..., upper] = max(var_max[index_small(i)..., upper], indicator_IDP.variable(view(u_tmp_upper, :, i), equations))

    var_min[index_small(i)..., lower] = min(var_min[index_small(i)..., lower], indicator_IDP.variable(view(u_tmp_lower, :, i), equations))
    var_max[index_small(i)..., lower] = max(var_max[index_small(i)..., lower], indicator_IDP.variable(view(u_tmp_lower, :, i), equations))

    # small to large
    if i <= nnodes(dg)/2
      var_min[index_large(i)..., large] = min(var_min[index_large(i)..., large], indicator_IDP.variable(view(u_tmp_large2, :, i), equations))
      var_max[index_large(i)..., large] = max(var_max[index_large(i)..., large], indicator_IDP.variable(view(u_tmp_large2, :, i), equations))
    else
      var_min[index_large(i)..., large] = min(var_min[index_large(i)..., large], indicator_IDP.variable(view(u_tmp_large1, :, i), equations))
      var_max[index_large(i)..., large] = max(var_max[index_large(i)..., large], indicator_IDP.variable(view(u_tmp_large1, :, i), equations))
    end
  end

  return nothing
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
