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

# Overload indicator when ShallowWaterEquations2D is used to apply full FV method on cells
# containing some "dry" LGL nodes. That is, if an element is partially "wet" then it becomes a
# full FV element.
function (indicator_hg::IndicatorHennemannGassner)(u::AbstractArray{<:Any,4},
                                                   mesh, equations::ShallowWaterEquations2D, 
                                                   dg::DGSEM, cache; kwargs...)
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

  # If h at one LGL node is lower than threshold_wet, set alpha[element]=1 via indicator_wet
  # in order to apply a full FV method in the partially wet cell and guarantee the well-balanced property.
  #
  # Determination of hard coded threshold: Showed good results in many numerical experiments. Idea is to
  # gain more stability when computing the velocity on (nearly) dry cells which
  # could be counteracted by division of conservative variables, e.g., v2 / v1 = hv / h.
  # Here, the impact of the threshold on the number of cells being updated with FV is not that
  # significant. However, its impact on the stability is very significant.
  # The value can be seen as a trade-off between accuracy and stability.
  # Well-balancedness of the scheme on partially wet cells with hydrostatic reconstruction 
  # can only be proven for the FV method (see Chen and Noelle).
  # Therefore we set alpha to one regardless of its given maximum value.
  #    
  threshold_wet = 1e-4

  @threaded for element in eachelement(dg, cache)
    indicator  = indicator_threaded[Threads.threadid()]
    modal      = modal_threaded[Threads.threadid()]
    modal_tmp1 = modal_tmp1_threaded[Threads.threadid()]

    # (Re-)set dummy variable for alpha_dry
    indicator_wet = 1.0

    # Calculate indicator variables at Gauss-Lobatto nodes
    for j in eachnode(dg), i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, j, element)
      h, _, _, _ = u_local
      indicator_wet = min(indicator_wet, Int32(h > threshold_wet))
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

    # Clip the maximum amount of FV allowed or set to 1 depending on indicator_wet
    alpha[element] = indicator_wet * min(alpha_max, alpha_element) - (indicator_wet-1)
  end

  if alpha_smooth
    apply_smoothing!(mesh, alpha, alpha_tmp, dg, cache)
  end

  return alpha
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
