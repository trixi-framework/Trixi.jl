
function create_cache(::Type{AnalysisCallback}, analyzer,
                      equations::AbstractEquations{2}, dg::DG, cache)
  eltype_u = eltype_x = eltype(cache.elements.node_coordinates) # TODO: AD, needs to be adapted

  # pre-allocate buffers
  u_local = zeros(eltype_u,
                  nvariables(equations), nnodes(analyzer), nnodes(analyzer))
  u_tmp1 = similar(u_local,
                   nvariables(equations), nnodes(analyzer), nnodes(dg))
  x_local = zeros(eltype_x,
                  ndims(equations), nnodes(analyzer), nnodes(analyzer))
  x_tmp1 = similar(x_local,
                   ndims(equations), nnodes(analyzer), nnodes(dg))

  return (; u_local, u_tmp1, x_local, x_tmp1)
end


function calc_error_norms(func, u::AbstractArray{<:Any,4}, t, analyzer,
                          mesh::TreeMesh{2}, equations, initial_condition,
                          dg::DGSEM, cache, cache_analysis)
  @unpack vandermonde, weights = analyzer
  @unpack node_coordinates = cache.elements
  @unpack u_local, u_tmp1, x_local, x_tmp1 = cache_analysis

  # Set up data structures
  l2_error   = zero(func(get_node_vars(u, equations, dg, 1, 1, 1), equations))
  linf_error = copy(l2_error)

  # Iterate over all elements for error calculations
  for element in eachelement(dg, cache)
    # Interpolate solution and node locations to analysis nodes
    multiply_dimensionwise!(u_local, vandermonde, view(u,                :, :, :, element), u_tmp1)
    multiply_dimensionwise!(x_local, vandermonde, view(node_coordinates, :, :, :, element), x_tmp1)

    # Calculate errors at each analysis node
    jacobian_volume = inv(cache.elements.inverse_jacobian[element])^ndims(equations)

    for j in eachnode(analyzer), i in eachnode(analyzer)
      u_exact = initial_condition(get_node_coords(x_local, equations, dg, i, j), t, equations)
      diff = func(u_exact, equations) - func(get_node_vars(u_local, equations, dg, i, j), equations)
      l2_error += diff.^2 * (weights[i] * weights[j] * jacobian_volume)
      linf_error = @. max(linf_error, abs(diff))
    end
  end

  # For L2 error, divide by total volume
  total_volume = mesh.tree.length_level_0^ndims(mesh)
  l2_error = @. sqrt(l2_error / total_volume)

  return l2_error, linf_error
end


function integrate_via_indices(func::Func, u::AbstractArray{<:Any,4},
                               mesh::TreeMesh{2}, equations, dg::DGSEM, cache,
                               args...; normalize=true) where {Func}
  @unpack weights = dg.basis

  # Initialize integral with zeros of the right shape
  integral = zero(func(u, 1, 1, 1, equations, dg, args...))

  # Use quadrature to numerically integrate over entire domain
  for element in eachelement(dg, cache)
    jacobian_volume = inv(cache.elements.inverse_jacobian[element])^ndims(equations)
    for j in eachnode(dg), i in eachnode(dg)
      integral += jacobian_volume * weights[i] * weights[j] * func(u, i, j, element, equations, dg, args...)
    end
  end

  # Normalize with total volume
  if normalize
    total_volume = mesh.tree.length_level_0^ndims(mesh)
    integral = integral / total_volume
  end

  return integral
end

function integrate(func::Func, u::AbstractArray{<:Any,4},
                   mesh::TreeMesh{2}, equations, dg::DGSEM, cache; normalize=true) where {Func}
  integrate_via_indices(u, mesh, equations, dg, cache; normalize=normalize) do u, i, j, element, equations, dg
    u_local = get_node_vars(u, equations, dg, i, j, element)
    return func(u_local, equations)
  end
end


function analyze(::typeof(entropy_timederivative), du::AbstractArray{<:Any,4}, u, t,
                 mesh::TreeMesh{2}, equations, dg::DG, cache)
  # Calculate ∫(∂S/∂u ⋅ ∂u/∂t)dΩ
  integrate_via_indices(u, mesh, equations, dg, cache, du) do u, i, j, element, equations, dg, du
    u_node  = get_node_vars(u,  equations, dg, i, j, element)
    du_node = get_node_vars(du, equations, dg, i, j, element)
    dot(cons2entropy(u_node, equations), du_node)
  end
end



function analyze(::Val{:l2_divb}, du::AbstractArray{<:Any,4}, u, t,
                 mesh::TreeMesh{2}, equations::IdealGlmMhdEquations2D,
                 dg::DG, cache)
  integrate_via_indices(u, mesh, equations, dg, cache, cache, dg.basis.derivative_matrix) do u, i, j, element, equations, dg, cache, derivative_matrix
    divb = zero(eltype(u))
    for k in eachnode(dg)
      divb += ( derivative_matrix[i, k] * u[6, k, j, element] +
                derivative_matrix[j, k] * u[7, i, k, element] )
    end
    divb *= cache.elements.inverse_jacobian[element]
    divb^2
  end |> sqrt
end

function analyze(::Val{:linf_divb}, du::AbstractArray{<:Any,4}, u, t,
                 mesh::TreeMesh{2}, equations::IdealGlmMhdEquations2D,
                 dg::DG, cache)
  @unpack derivative_matrix, weights = dg.basis

  # integrate over all elements to get the divergence-free condition errors
  linf_divb = zero(eltype(u))
  for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      divb = zero(eltype(u))
      for k in eachnode(dg)
        divb += ( derivative_matrix[i, k] * u[6, k, j, element] +
                  derivative_matrix[j, k] * u[7, i, k, element] )
      end
      divb *= cache.elements.inverse_jacobian[element]
      linf_divb = max(linf_divb, abs(divb))
    end
  end

  return linf_divb
end
