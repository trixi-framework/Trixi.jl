
function create_cache_analysis(analyzer, mesh::TreeMesh{3},
                               equations::AbstractEquations{3}, dg::DG, cache,
                               RealT, uEltype)

  # pre-allocate buffers
  u_local = zeros(uEltype,
                  nvariables(equations), nnodes(analyzer), nnodes(analyzer), nnodes(analyzer))
  u_tmp1 = similar(u_local,
                   nvariables(equations), nnodes(analyzer), nnodes(dg), nnodes(dg))
  u_tmp2 = similar(u_local,
                   nvariables(equations), nnodes(analyzer), nnodes(analyzer), nnodes(dg))
  x_local = zeros(RealT,
                  ndims(equations), nnodes(analyzer), nnodes(analyzer), nnodes(analyzer))
  x_tmp1 = similar(x_local,
                   ndims(equations), nnodes(analyzer), nnodes(dg), nnodes(dg))
  x_tmp2 = similar(x_local,
                   ndims(equations), nnodes(analyzer), nnodes(analyzer), nnodes(dg))

  return (; u_local, u_tmp1, u_tmp2, x_local, x_tmp1, x_tmp2)
end


function calc_error_norms(func, u::AbstractArray{<:Any,5}, t, analyzer,
                          mesh::Union{TreeMesh{3},CurvedMesh{3}}, equations, initial_condition,
                          dg::DGSEM, cache, cache_analysis)
  @unpack vandermonde, weights = analyzer
  @unpack node_coordinates = cache.elements
  @unpack u_local, u_tmp1, u_tmp2, x_local, x_tmp1, x_tmp2 = cache_analysis

  # Set up data structures
  l2_error   = zero(func(get_node_vars(u, equations, dg, 1, 1, 1, 1), equations))
  linf_error = copy(l2_error)

  # Iterate over all elements for error calculations
  for element in eachelement(dg, cache)
    # Interpolate solution and node locations to analysis nodes
    multiply_dimensionwise!(u_local, vandermonde, view(u,                :, :, :, :, element), u_tmp1, u_tmp2)
    multiply_dimensionwise!(x_local, vandermonde, view(node_coordinates, :, :, :, :, element), x_tmp1, x_tmp2)

    # Calculate errors at each analysis node
    volume_jacobian_ = volume_jacobian(element, mesh, cache)

    for k in eachnode(analyzer), j in eachnode(analyzer), i in eachnode(analyzer)
      u_exact = initial_condition(get_node_coords(x_local, equations, dg, i, j, k), t, equations)
      diff = func(u_exact, equations) - func(get_node_vars(u_local, equations, dg, i, j, k), equations)
      l2_error += diff.^2 * (weights[i] * weights[j] * weights[k] * volume_jacobian_)
      linf_error = @. max(linf_error, abs(diff))
    end
  end

  # For L2 error, divide by total volume
  total_volume_ = total_volume(mesh)
  l2_error = @. sqrt(l2_error / total_volume_)

  return l2_error, linf_error
end


function integrate_via_indices(func::Func, u::AbstractArray{<:Any,5},
                               mesh::Union{TreeMesh{3},CurvedMesh{3}}, equations, dg::DGSEM, cache,
                               args...; normalize=true) where {Func}
  @unpack weights = dg.basis

  # Initialize integral with zeros of the right shape
  integral = zero(func(u, 1, 1, 1, 1, equations, dg, args...))

  # Use quadrature to numerically integrate over entire domain
  for element in eachelement(dg, cache)
    volume_jacobian_ = volume_jacobian(element, mesh, cache)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      integral += volume_jacobian_ * weights[i] * weights[j] * weights[k] * func(u, i, j, k, element, equations, dg, args...)
    end
  end

  # Normalize with total volume
  if normalize
    total_volume_ = total_volume(mesh)
    integral = integral / total_volume_
  end

  return integral
end


function integrate(func::Func, u::AbstractArray{<:Any,5},
                   mesh::Union{TreeMesh{3},CurvedMesh{3}}, equations, dg::DGSEM, cache; normalize=true) where {Func}
  integrate_via_indices(u, mesh, equations, dg, cache; normalize=normalize) do u, i, j, k, element, equations, dg
    u_local = get_node_vars(u, equations, dg, i, j, k, element)
    return func(u_local, equations)
  end
end


function analyze(::typeof(entropy_timederivative), du::AbstractArray{<:Any,5}, u, t,
                 mesh::Union{TreeMesh{3},CurvedMesh{3}}, equations, dg::DG, cache)
  # Calculate ∫(∂S/∂u ⋅ ∂u/∂t)dΩ
  integrate_via_indices(u, mesh, equations, dg, cache, du) do u, i, j, k, element, equations, dg, du
    u_node  = get_node_vars(u,  equations, dg, i, j, k, element)
    du_node = get_node_vars(du, equations, dg, i, j, k, element)
    dot(cons2entropy(u_node, equations), du_node)
  end
end



function analyze(::Val{:l2_divb}, du::AbstractArray{<:Any,5}, u, t,
                 mesh::TreeMesh{3}, equations::IdealGlmMhdEquations3D,
                 dg::DG, cache)
  integrate_via_indices(u, mesh, equations, dg, cache, cache, dg.basis.derivative_matrix) do u, i, j, k, element, equations, dg, cache, derivative_matrix
    divb = zero(eltype(u))
    for l in eachnode(dg)
      divb += ( derivative_matrix[i, l] * u[6, l, j, k, element] +
                derivative_matrix[j, l] * u[7, i, l, k, element] +
                derivative_matrix[k, l] * u[7, i, j, l, element] )
    end
    divb *= cache.elements.inverse_jacobian[element]
    divb^2
  end |> sqrt
end

function analyze(::Val{:linf_divb}, du::AbstractArray{<:Any,5}, u, t,
                 mesh::TreeMesh{3}, equations::IdealGlmMhdEquations3D,
                 dg::DG, cache)
  @unpack derivative_matrix, weights = dg.basis

  # integrate over all elements to get the divergence-free condition errors
  linf_divb = zero(eltype(u))
  for element in eachelement(dg, cache)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      divb = zero(eltype(u))
      for l in eachnode(dg)
        divb += ( derivative_matrix[i, l] * u[6, l, j, k, element] +
                  derivative_matrix[j, l] * u[7, i, l, k, element] +
                  derivative_matrix[k, l] * u[7, i, j, l, element] )
      end
      divb *= cache.elements.inverse_jacobian[element]
      linf_divb = max(linf_divb, abs(divb))
    end
  end

  return linf_divb
end
