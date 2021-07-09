# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


# this method is used when the indicator is constructed as for shock-capturing volume integrals
function create_cache(::Type{IndicatorHennemannGassner}, equations::AbstractEquations{1}, basis::LobattoLegendreBasis)

  alpha = Vector{real(basis)}()
  alpha_tmp = similar(alpha)

  A = Array{real(basis), ndims(equations)}
  indicator_threaded  = [A(undef, nnodes(basis)) for _ in 1:Threads.nthreads()]
  modal_threaded      = [A(undef, nnodes(basis)) for _ in 1:Threads.nthreads()]

  return (; alpha, alpha_tmp, indicator_threaded, modal_threaded)
end

# this method is used when the indicator is constructed as for AMR
function create_cache(typ::Type{IndicatorHennemannGassner}, mesh, equations::AbstractEquations{1}, dg::DGSEM, cache)
  create_cache(typ, equations, dg.basis)
end


function (indicator_hg::IndicatorHennemannGassner)(u::AbstractArray{<:Any,3},
                                                   equations, dg::DGSEM, cache;
                                                   kwargs...)
  @unpack alpha_max, alpha_min, alpha_smooth, variable = indicator_hg
  @unpack alpha, alpha_tmp, indicator_threaded, modal_threaded = indicator_hg.cache
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

    # Calculate indicator variables at Gauss-Lobatto nodes
    for i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, element)
      indicator[i] = indicator_hg.variable(u_local, equations)
    end

    # Convert to modal representation
    multiply_scalar_dimensionwise!(modal, dg.basis.inverse_vandermonde_legendre, indicator)

    # Calculate total energies for all modes, without highest, without two highest
    total_energy = zero(eltype(modal))
    for i in 1:nnodes(dg)
      total_energy += modal[i]^2
    end
    total_energy_clip1 = zero(eltype(modal))
    for i in 1:(nnodes(dg)-1)
      total_energy_clip1 += modal[i]^2
    end
    total_energy_clip2 = zero(eltype(modal))
    for i in 1:(nnodes(dg)-2)
      total_energy_clip2 += modal[i]^2
    end

    # Calculate energy in lower modes
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

  if (alpha_smooth)
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
  end

  return alpha
end



# this method is used when the indicator is constructed as for shock-capturing volume integrals
function create_cache(::Type{IndicatorLöhner}, equations::AbstractEquations{1}, basis::LobattoLegendreBasis)

  alpha = Vector{real(basis)}()

  A = Array{real(basis), ndims(equations)}
  indicator_threaded = [A(undef, nnodes(basis)) for _ in 1:Threads.nthreads()]

  return (; alpha, indicator_threaded)
end

# this method is used when the indicator is constructed as for AMR
function create_cache(typ::Type{IndicatorLöhner}, mesh, equations::AbstractEquations{1}, dg::DGSEM, cache)
  create_cache(typ, equations, dg.basis)
end


function (löhner::IndicatorLöhner)(u::AbstractArray{<:Any,3},
                                   equations, dg::DGSEM, cache;
                                   kwargs...)
  @assert nnodes(dg) >= 3 "IndicatorLöhner only works for nnodes >= 3 (polydeg > 1)"
  @unpack alpha, indicator_threaded = löhner.cache
  resize!(alpha, nelements(dg, cache))

  @threaded for element in eachelement(dg, cache)
    indicator = indicator_threaded[Threads.threadid()]

    # Calculate indicator variables at Gauss-Lobatto nodes
    for i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, element)
      indicator[i] = löhner.variable(u_local, equations)
    end

    estimate = zero(real(dg))
    for i in 2:nnodes(dg)-1
      # x direction
      u0 = indicator[i  ]
      up = indicator[i+1]
      um = indicator[i-1]
      estimate = max(estimate, local_löhner_estimate(um, u0, up, löhner))
    end

    # use the maximum as DG element indicator
    alpha[element] = estimate
  end

  return alpha
end



# this method is used when the indicator is constructed as for shock-capturing volume integrals
function create_cache(::Type{IndicatorMax}, equations::AbstractEquations{1}, basis::LobattoLegendreBasis)

  alpha = Vector{real(basis)}()

  A = Array{real(basis), ndims(equations)}
  indicator_threaded = [A(undef, nnodes(basis)) for _ in 1:Threads.nthreads()]

  return (; alpha, indicator_threaded)
end

# this method is used when the indicator is constructed as for AMR
function create_cache(typ::Type{IndicatorMax}, mesh, equations::AbstractEquations{1}, dg::DGSEM, cache)
  cache = create_cache(typ, equations, dg.basis)
end


function (indicator_max::IndicatorMax)(u::AbstractArray{<:Any,3},
                                       equations, dg::DGSEM, cache;
                                       kwargs...)
  @unpack alpha, indicator_threaded = indicator_max.cache
  resize!(alpha, nelements(dg, cache))

  @threaded for element in eachelement(dg, cache)
    indicator = indicator_threaded[Threads.threadid()]

    # Calculate indicator variables at Gauss-Lobatto nodes
    for i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, element)
      indicator[i] = indicator_max.variable(u_local, equations)
    end

    alpha[element] = maximum(indicator)
  end

  return alpha
end


# this method is used when the indicator is constructed as for shock-capturing volume integrals
function create_cache(::Type{IndicatorNNPP}, equations::AbstractEquations{1}, basis::LobattoLegendreBasis)

  alpha = Vector{real(basis)}()
  alpha_tmp = similar(alpha)

  A = Array{real(basis), ndims(equations)}
  indicator_threaded  = [A(undef, nnodes(basis)) for _ in 1:Threads.nthreads()]
  modal_threaded      = [A(undef, nnodes(basis)) for _ in 1:Threads.nthreads()]

  return (; alpha, alpha_tmp, indicator_threaded, modal_threaded)
end

# this method is used when the indicator is constructed as for AMR
function create_cache(typ::Type{IndicatorNNPP}, mesh, equations::AbstractEquations{1}, dg::DGSEM, cache)
  create_cache(typ, equations, dg.basis)
end


function (indicator_nnpp::IndicatorNNPP)(u::AbstractArray{<:Any,3},
                                                   equations, dg::DGSEM, cache;
                                                   kwargs...)
  @unpack alpha_max, alpha_min, alpha_smooth, alpha_continuous, alpha_amr, variable, network = indicator_nnpp
  @unpack alpha, alpha_tmp, indicator_threaded, modal_threaded = indicator_nnpp.cache
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

    # Calculate indicator variables at Gauss-Lobatto nodes
    for i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, element)
      indicator[i] = indicator_nnpp.variable(u_local, equations)
    end

    # Convert to modal representation
    multiply_scalar_dimensionwise!(modal, dg.basis.inverse_vandermonde_legendre, indicator)

    # Calculate total energies for all modes, without highest, without two highest
    total_energy = zero(eltype(modal))
    for i in 1:nnodes(dg)
      total_energy += modal[i]^2
    end
    total_energy_clip1 = zero(eltype(modal))
    for i in 1:(nnodes(dg)-1)
      total_energy_clip1 += modal[i]^2
    end
    total_energy_clip2 = zero(eltype(modal))
    for i in 1:(nnodes(dg)-2)
      total_energy_clip2 += modal[i]^2
    end


    if size(Flux.params(network)[1],2) == 2 
      # Calculate energy in lower modes for the network input
      X = zeros(2,1)
      X[1] = (total_energy - total_energy_clip1)/total_energy
      X[2] = (total_energy_clip1 - total_energy_clip2)/total_energy_clip1

    elseif size(Flux.params(network)[1],2) == 3
      # Calculate energy in lower modes and polynomial degree for the network input
    	X = zeros(3,1)
      X[1] = (total_energy - total_energy_clip1)/total_energy
      X[2] = (total_energy_clip1 - total_energy_clip2)/total_energy_clip1
      X[3] = nnodes(dg) 
    end

    # Scale input data
    X = X ./max(maximum(abs.(X)),1)

    if alpha_continuous && !alpha_amr
      # Set good cells to 0 and troubled cells to continuous value of the network prediction
      if network(X)[1] > 0.5
        alpha_element = network(X)[1]
      else
        alpha_element = 0
      end
      
      # Take care of the case close to pure FV
      if alpha_element > 1 - alpha_min
       alpha_element = one(alpha_element)
      end

      # Clip the maximum amount of FV allowed
      alpha[element] = alpha_max * alpha_element
    elseif !alpha_continuous && !alpha_amr
      # Set good cells to 0 and troubled cells to 1
      if network(X)[1] > 0.5
        alpha[element] = 1
      else
        alpha[element] = 0
      end
    elseif alpha_amr
      # The entire continuous output of the neural network is used for AMR
      alpha_element = network(X)[1]
    end
  end

  if (alpha_smooth)
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
  end

  return alpha
end


# this method is used when the indicator is constructed as for shock-capturing volume integrals
function create_cache(::Type{IndicatorNNRH}, equations::AbstractEquations{1}, basis::LobattoLegendreBasis, mesh::TreeMesh{1})

  alpha = Vector{real(basis)}()
  alpha_tmp = similar(alpha)

  A = Array{real(basis), ndims(equations)}
  indicator_threaded  = [A(undef, nnodes(basis)) for _ in 1:Threads.nthreads()]

  return (; alpha, alpha_tmp, indicator_threaded, mesh)
end

# this method is used when the indicator is constructed as for AMR
function create_cache(typ::Type{IndicatorNNRH}, mesh, equations::AbstractEquations{1}, dg::DGSEM, cache)
  create_cache(typ, equations, dg.basis)
end


function (indicator_nnrh::IndicatorNNRH)(u::AbstractArray{<:Any,3},
                                                   equations, dg::DGSEM, cache;
                                                   kwargs...)
  @unpack alpha_max, alpha_min, alpha_smooth, alpha_continuous, alpha_amr, variable, network = indicator_nnrh
  @unpack alpha, alpha_tmp, indicator_threaded, mesh = indicator_nnrh.cache
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
  

  @threaded for element in eachelement(dg, cache)
    indicator  = indicator_threaded[Threads.threadid()]
    cell_id = cache.elements.cell_ids[element]
    neighbor_ids = Array{Int64}(undef, 2)

    for direction in eachdirection(mesh.tree)
      if !has_any_neighbor(mesh.tree, cell_id, direction)
        neighbor_ids[direction] = element_id
      continue
      end
      if has_neighbor(mesh.tree, cell_id, direction)
        neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
        if has_children(mesh.tree, neighbor_cell_id) # Cell has small neighbor
          if direction == 1
            neighbor_ids[direction] = c2e[mesh.tree.child_ids[2, neighbor_cell_id]]
          else
            neighbor_ids[direction] = c2e[mesh.tree.child_ids[1, neighbor_cell_id]]
          end
        else # Cell has same refinement level neighbor
          neighbor_ids[direction] = c2e[neighbor_cell_id]
        end
      else # Cell is small and has large neighbor
        parent_id = mesh.tree.parent_ids[cell_id]
        neighbor_cell_id = mesh.tree.neighbor_ids[direction, parent_id]
        neighbor_ids[direction] = c2e[neighbor_cell_id]
      end
    end

    # Calculate indicator variables at Gauss-Lobatto nodes
    for i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, element)
      indicator[i] = indicator_nnrh.variable(u_local, equations)
    end

    X = Array{Float64}(undef, 5)
    # Cell average and interface values of the cell
    X[2] = sum(indicator)/nnodes(dg)
    X[4] = indicator[1]
    X[5] = indicator[end]
  
    # Calculate indicator variables from left neighboring cell at Gauss-Lobatto nodes 
    for i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, neighbor_ids[1])
      indicator[i] = indicator_nnrh.variable(u_local, equations)
    end
    X[1] = sum(indicator)/nnodes(dg)

    # Calculate indicator variables from right neighboring cell at Gauss-Lobatto nodes 
    for i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, neighbor_ids[2])
      indicator[i] = indicator_nnrh.variable(u_local, equations)
    end
    X[3] = sum(indicator)/nnodes(dg)
 

    # Scale input data
    X = X ./max(maximum(abs.(X)),1)

    if alpha_continuous && !alpha_amr
      # Set good cells to 0 and troubled cells to continuous value of the network prediction
      if network(X)[1] > 0.5
        alpha_element = network(X)[1]
      else
        alpha_element = 0
      end
      
      # Take care of the case close to pure FV
      if alpha_element > 1 - alpha_min
       alpha_element = one(alpha_element)
      end

      # Clip the maximum amount of FV allowed
      alpha[element] = alpha_max * alpha_element
    elseif !alpha_continuous && !alpha_amr
      # Set good cells to 0 and troubled cells to 1
      if network(X)[1] > 0.5
        alpha[element] = 1
      else
        alpha[element] = 0
      end
    elseif alpha_amr
      # The entire continuous output of the neural network is used for AMR
      alpha_element = network(X)[1]
    end
  end

  if (alpha_smooth)
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
  end

  return alpha
end
end # @muladd
