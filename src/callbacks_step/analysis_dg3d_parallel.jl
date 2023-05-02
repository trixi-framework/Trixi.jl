# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


function calc_error_norms(func, u, t, analyzer,
                          mesh::ParallelP4estMesh{3}, equations,
                          initial_condition, dg::DGSEM, cache, cache_analysis)
  @unpack vandermonde, weights = analyzer
  @unpack node_coordinates, inverse_jacobian = cache.elements
  @unpack u_local, u_tmp1, u_tmp2, x_local, x_tmp1, x_tmp2, jacobian_local, jacobian_tmp1, jacobian_tmp2 = cache_analysis

  # Set up data structures
  l2_error   = zero(func(get_node_vars(u, equations, dg, 1, 1, 1, 1), equations))
  linf_error = copy(l2_error)
  volume = zero(real(mesh))

  # Iterate over all elements for error calculations
  for element in eachelement(dg, cache)
    # Interpolate solution and node locations to analysis nodes
    multiply_dimensionwise!(u_local, vandermonde, view(u,                :, :, :, :, element), u_tmp1, u_tmp2)
    multiply_dimensionwise!(x_local, vandermonde, view(node_coordinates, :, :, :, :, element), x_tmp1, x_tmp2)
    multiply_scalar_dimensionwise!(jacobian_local, vandermonde, inv.(view(inverse_jacobian, :, :, :, element)), jacobian_tmp1, jacobian_tmp2)

    # Calculate errors at each analysis node
    @. jacobian_local = abs(jacobian_local)

    for k in eachnode(analyzer), j in eachnode(analyzer), i in eachnode(analyzer)
      u_exact = initial_condition(get_node_coords(x_local, equations, dg, i, j, k), t, equations)
      diff = func(u_exact, equations) - func(get_node_vars(u_local, equations, dg, i, j, k), equations)
      l2_error += diff.^2 * (weights[i] * weights[j] * weights[k] * jacobian_local[i, j, k])
      linf_error = @. max(linf_error, abs(diff))
      volume += weights[i] * weights[j] * weights[k] * jacobian_local[i, j, k]
    end
  end

  # Accumulate local results on root process
  global_l2_error = Vector(l2_error)
  global_linf_error = Vector(linf_error)
  MPI.Reduce!(global_l2_error, +, mpi_root(), mpi_comm())
  MPI.Reduce!(global_linf_error, max, mpi_root(), mpi_comm())
  total_volume = MPI.Reduce(volume, +, mpi_root(), mpi_comm())
  if mpi_isroot()
    l2_error   = convert(typeof(l2_error),   global_l2_error)
    linf_error = convert(typeof(linf_error), global_linf_error)
    # For L2 error, divide by total volume
    l2_error = @. sqrt(l2_error / total_volume)
  else
    l2_error   = convert(typeof(l2_error),   NaN * global_l2_error)
    linf_error = convert(typeof(linf_error), NaN * global_linf_error)
  end

  return l2_error, linf_error
end


function integrate_via_indices(func::Func, u,
                               mesh::ParallelP4estMesh{3}, equations,
                               dg::DGSEM, cache, args...; normalize=true) where {Func}
  @unpack weights = dg.basis

  # Initialize integral with zeros of the right shape
  # Pass `zero(SVector{nvariables(equations), eltype(u))}` to `func` since `u` might be empty, if the
  # current rank has no elements, see also https://github.com/trixi-framework/Trixi.jl/issues/1096.
  integral = zero(func(zero(SVector{nvariables(equations), eltype(u)}), 1, 1, 1, 1, equations, dg, args...))
  volume = zero(real(mesh))

  # Use quadrature to numerically integrate over entire domain
  for element in eachelement(dg, cache)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      volume_jacobian = abs(inv(cache.elements.inverse_jacobian[i, j, k, element]))
      integral += volume_jacobian * weights[i] * weights[j] * weights[k] * func(u, i, j, k, element, equations, dg, args...)
      volume += volume_jacobian * weights[i] * weights[j] * weights[k]
    end
  end

  global_integral = MPI.Reduce!(Ref(integral), +, mpi_root(), mpi_comm())
  total_volume = MPI.Reduce(volume, +, mpi_root(), mpi_comm())
  if mpi_isroot()
    integral = convert(typeof(integral), global_integral[])
  else
    integral = convert(typeof(integral), NaN * integral)
    total_volume = volume # non-root processes receive nothing from reduce -> overwrite
  end

  # Normalize with total volume
  if normalize
    integral = integral / total_volume
  end

  return integral
end


end # @muladd
