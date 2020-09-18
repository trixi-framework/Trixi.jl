
function calc_error_norms(func, u::AbstractArray{<:Any,4}, t, analyzer,
                          mesh::TreeMesh{2}, equations, initial_conditions, dg::DGSEM, cache)
  @unpack vandermonde, weights = analyzer
  @unpack node_coordinates = cache.elements

  # pre-allocate buffers
  u_local = zeros(eltype(u),
                  nvariables(equations), nnodes(analyzer), nnodes(analyzer))
  u_tmp1 = similar(u_local,
                   nvariables(equations), nnodes(analyzer), nnodes(dg))
  x_local = zeros(eltype(node_coordinates),
                  ndims(equations), nnodes(analyzer), nnodes(analyzer))
  x_tmp1 = similar(x_local,
                   ndims(equations), nnodes(analyzer), nnodes(dg))

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
      u_exact = initial_conditions(get_node_coords(x_local, equations, dg, i, j), t, equations)
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


function integrate(func, mesh::TreeMesh{2}, equations, dg::DGSEM, cache,
                   u::AbstractArray{<:Any,4}, args...; normalize=true)
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

function integrate(func, u::AbstractArray{<:Any,4},
                   mesh::TreeMesh{2}, equations, dg::DGSEM, cache; normalize=true)
  func_wrapped = function(u, i, j, element, equations, dg)
    u_local = get_node_vars(u, equations, dg, i, j, element)
    return func(u_local, equations)
  end
  return integrate(func_wrapped, mesh, equations, dg, cache, u; normalize=normalize)
end


function calc_entropy_timederivative(du::AbstractArray{<:Any,4}, u,
                                     mesh::TreeMesh{2}, equations, dg::DG, cache)
  # Calculate ∫(∂S/∂u ⋅ ∂u/∂t)dΩ
  dsdu_ut = integrate(mesh, equations, dg, cache, u, du) do u, i, j, element, equations, dg, du
    u_node  = get_node_vars(u,  equations, dg, i, j, element)
    du_node = get_node_vars(du, equations, dg, i, j, element)
    dot(cons2entropy(u_node, equations), du_node)
  end

  return dsdu_ut
end