# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


function create_cache_analysis(analyzer, mesh::TreeMesh{1},
                               equations, dg::DG, cache,
                               RealT, uEltype)

  # pre-allocate buffers
  # We use `StrideArray`s here since these buffers are used in performance-critical
  # places and the additional information passed to the compiler makes them faster
  # than native `Array`s.
  u_local = StrideArray(undef, uEltype,
                        StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)))
  x_local = StrideArray(undef, RealT,
                        StaticInt(ndims(equations)), StaticInt(nnodes(analyzer)))

  return (; u_local, x_local)
end


function create_cache_analysis(analyzer, mesh::StructuredMesh{1},
                               equations, dg::DG, cache,
                               RealT, uEltype)

  # pre-allocate buffers
  # We use `StrideArray`s here since these buffers are used in performance-critical
  # places and the additional information passed to the compiler makes them faster
  # than native `Array`s.
  u_local = StrideArray(undef, uEltype,
                        StaticInt(nvariables(equations)), StaticInt(nnodes(analyzer)))
  x_local = StrideArray(undef, RealT,
                        StaticInt(ndims(equations)), StaticInt(nnodes(analyzer)))
  jacobian_local = StrideArray(undef, RealT,
                               StaticInt(nnodes(analyzer)))

  return (; u_local, x_local, jacobian_local)
end


function calc_error_norms(func, u, t, analyzer,
                          mesh::StructuredMesh{1}, equations, initial_condition,
                          dg::DGSEM, cache, cache_analysis)
  @unpack vandermonde, weights = analyzer
  @unpack node_coordinates, inverse_jacobian = cache.elements
  @unpack u_local, x_local, jacobian_local = cache_analysis

  # Set up data structures
  l2_error   = zero(func(get_node_vars(u, equations, dg, 1, 1), equations))
  linf_error = copy(l2_error)
  total_volume = zero(real(mesh))

  # Iterate over all elements for error calculations
  for element in eachelement(dg, cache)
    # Interpolate solution and node locations to analysis nodes
    multiply_dimensionwise!(u_local, vandermonde, view(u,                :, :, element))
    multiply_dimensionwise!(x_local, vandermonde, view(node_coordinates, :, :, element))
    multiply_scalar_dimensionwise!(jacobian_local, vandermonde, inv.(view(inverse_jacobian, :, element)))

    # Calculate errors at each analysis node
    @. jacobian_local = abs(jacobian_local)

    for i in eachnode(analyzer)
      u_exact = initial_condition(get_node_coords(x_local, equations, dg, i), t, equations)
      diff = func(u_exact, equations) - func(get_node_vars(u_local, equations, dg, i), equations)
      l2_error += diff.^2 * (weights[i] * jacobian_local[i])
      linf_error = @. max(linf_error, abs(diff))
      total_volume += weights[i] * jacobian_local[i]
    end
  end

  # For L2 error, divide by total volume
  l2_error = @. sqrt(l2_error / total_volume)

  return l2_error, linf_error
end


function calc_error_norms(func, u, t, analyzer,
                          mesh::TreeMesh{1}, equations, initial_condition,
                          dg::DGSEM, cache, cache_analysis)
  @unpack vandermonde, weights = analyzer
  @unpack node_coordinates = cache.elements
  @unpack u_local, x_local = cache_analysis

  # Set up data structures
  l2_error   = zero(func(get_node_vars(u, equations, dg, 1, 1), equations))
  linf_error = copy(l2_error)

  # Iterate over all elements for error calculations
  for element in eachelement(dg, cache)
    # Interpolate solution and node locations to analysis nodes
    multiply_dimensionwise!(u_local, vandermonde, view(u,                :, :, element))
    multiply_dimensionwise!(x_local, vandermonde, view(node_coordinates, :, :, element))

    # Calculate errors at each analysis node
    volume_jacobian_ = volume_jacobian(element, mesh, cache)

    for i in eachnode(analyzer)
      u_exact = initial_condition(get_node_coords(x_local, equations, dg, i), t, equations)
      diff = func(u_exact, equations) - func(get_node_vars(u_local, equations, dg, i), equations)
      l2_error += diff.^2 * (weights[i] * volume_jacobian_)
      linf_error = @. max(linf_error, abs(diff))
    end
  end

  # For L2 error, divide by total volume
  total_volume_ = total_volume(mesh)
  l2_error = @. sqrt(l2_error / total_volume_)

  return l2_error, linf_error
end


function integrate_via_indices(func::Func, u,
                               mesh::StructuredMesh{1}, equations, dg::DGSEM, cache,
                               args...; normalize=true) where {Func}
  @unpack weights = dg.basis

  # Initialize integral with zeros of the right shape
  integral = zero(func(u, 1, 1, equations, dg, args...))
  total_volume = zero(real(mesh))

  # Use quadrature to numerically integrate over entire domain
  for element in eachelement(dg, cache)
    for i in eachnode(dg)
      jacobian_volume = abs(inv(cache.elements.inverse_jacobian[i, element]))
      integral += jacobian_volume * weights[i] * func(u, i, element, equations, dg, args...)
      total_volume += jacobian_volume * weights[i]
    end
  end
  # Normalize with total volume
  if normalize
    integral = integral / total_volume
  end

  return integral
end


function integrate_via_indices(func::Func, u,
                               mesh::TreeMesh{1}, equations, dg::DGSEM, cache,
                               args...; normalize=true) where {Func}
  @unpack weights = dg.basis

  # Initialize integral with zeros of the right shape
  integral = zero(func(u, 1, 1, equations, dg, args...))

  # Use quadrature to numerically integrate over entire domain
  for element in eachelement(dg, cache)
    volume_jacobian_ = volume_jacobian(element, mesh, cache)
    for i in eachnode(dg)
      integral += volume_jacobian_ * weights[i] * func(u, i, element, equations, dg, args...)
    end
  end

  # Normalize with total volume
  if normalize
    integral = integral / total_volume(mesh)
  end

  return integral
end


function integrate(func::Func, u,
                   mesh::Union{TreeMesh{1},StructuredMesh{1}},
                   equations, dg::DG, cache; normalize=true) where {Func}
  integrate_via_indices(u, mesh, equations, dg, cache; normalize=normalize) do u, i, element, equations, dg
    u_local = get_node_vars(u, equations, dg, i, element)
    return func(u_local, equations)
  end
end


function analyze(::typeof(entropy_timederivative), du, u, t,
                 mesh::Union{TreeMesh{1},StructuredMesh{1}}, equations, dg::DG, cache)
  # Calculate ∫(∂S/∂u ⋅ ∂u/∂t)dΩ
  integrate_via_indices(u, mesh, equations, dg, cache, du) do u, i, element, equations, dg, du
    u_node  = get_node_vars(u,  equations, dg, i, element)
    du_node = get_node_vars(du, equations, dg, i, element)
    dot(cons2entropy(u_node, equations), du_node)
  end
end

function analyze(::Val{:l2_divb}, du, u, t,
                 mesh::TreeMesh{1}, equations::IdealGlmMhdEquations1D,
                 dg::DG, cache)
  integrate_via_indices(u, mesh, equations, dg, cache, dg.basis.derivative_matrix) do u, i, element, equations, dg, derivative_matrix
    divb = zero(eltype(u))
    for k in eachnode(dg)
      divb += derivative_matrix[i, k] * u[6, k, element]
    end
    divb *= cache.elements.inverse_jacobian[element]
    divb^2
  end |> sqrt
end

function analyze(::Val{:linf_divb}, du, u, t,
                 mesh::TreeMesh{1}, equations::IdealGlmMhdEquations1D,
                 dg::DG, cache)
  @unpack derivative_matrix, weights = dg.basis

  # integrate over all elements to get the divergence-free condition errors
  linf_divb = zero(eltype(u))
  for element in eachelement(dg, cache)
    for i in eachnode(dg)
      divb = zero(eltype(u))
      for k in eachnode(dg)
        divb += derivative_matrix[i, k] * u[6, k, element]
      end
      divb *= cache.elements.inverse_jacobian[element]
      linf_divb = max(linf_divb, abs(divb))
    end
  end

  return linf_divb
end


end # @muladd
