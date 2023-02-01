# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


function calc_error_norms(func, u, t, analyzer,
                          mesh::ParallelTreeMesh{2}, equations, initial_condition,
                          dg::DGSEM, cache, cache_analysis)
  l2_errors, linf_errors = calc_error_norms_per_element(func, u, t, analyzer,
                                                        mesh, equations, initial_condition,
                                                        dg, cache, cache_analysis)

  # Collect local error norms for each element on root process. That way, when aggregating the L2
  # errors, the order of summation is the same as in the serial case to ensure exact equality.
  # This facilitates easier parallel development and debugging (see
  # https://github.com/trixi-framework/Trixi.jl/pull/850#pullrequestreview-757463943 for details).
  # Note that this approach does not scale.
  if mpi_isroot()
    global_l2_errors = zeros(eltype(l2_errors), cache.mpi_cache.n_elements_global)
    global_linf_errors = similar(global_l2_errors)

    n_elements_by_rank = parent(cache.mpi_cache.n_elements_by_rank) # convert OffsetArray to Array
    l2_buf = MPI.VBuffer(global_l2_errors, n_elements_by_rank)
    linf_buf = MPI.VBuffer(global_linf_errors, n_elements_by_rank)
    MPI.Gatherv!(l2_errors, l2_buf, mpi_root(), mpi_comm())
    MPI.Gatherv!(linf_errors, linf_buf, mpi_root(), mpi_comm())
  else
    MPI.Gatherv!(l2_errors, nothing, mpi_root(), mpi_comm())
    MPI.Gatherv!(linf_errors, nothing, mpi_root(), mpi_comm())
  end

  # Aggregate element error norms on root process
  if mpi_isroot()
    # sum(global_l2_errors) does not produce the same result as in the serial case, thus a
    # hand-written loop is used
    l2_error = zero(eltype(global_l2_errors))
    for error in global_l2_errors
      l2_error += error
    end
    linf_error = reduce((x, y) -> max.(x, y), global_linf_errors)

    # For L2 error, divide by total volume
    total_volume_ = total_volume(mesh)
    l2_error = @. sqrt(l2_error / total_volume_)
  else
    l2_error = convert(eltype(l2_errors), NaN * zero(eltype(l2_errors)))
    linf_error = convert(eltype(linf_errors), NaN * zero(eltype(linf_errors)))
  end

  return l2_error, linf_error
end

function calc_error_norms_per_element(func, u, t, analyzer,
                                      mesh::ParallelTreeMesh{2}, equations, initial_condition,
                                      dg::DGSEM, cache, cache_analysis)
  @unpack vandermonde, weights = analyzer
  @unpack node_coordinates = cache.elements
  @unpack u_local, u_tmp1, x_local, x_tmp1 = cache_analysis

  # Set up data structures
  T = typeof(zero(func(get_node_vars(u, equations, dg, 1, 1, 1), equations)))
  l2_errors = zeros(T, nelements(dg, cache))
  linf_errors = copy(l2_errors)

  # Iterate over all elements for error calculations
  for element in eachelement(dg, cache)
    # Interpolate solution and node locations to analysis nodes
    multiply_dimensionwise!(u_local, vandermonde, view(u,                :, :, :, element), u_tmp1)
    multiply_dimensionwise!(x_local, vandermonde, view(node_coordinates, :, :, :, element), x_tmp1)

    # Calculate errors at each analysis node
    volume_jacobian_ = volume_jacobian(element, mesh, cache)

    for j in eachnode(analyzer), i in eachnode(analyzer)
      u_exact = initial_condition(get_node_coords(x_local, equations, dg, i, j), t, equations)
      diff = func(u_exact, equations) - func(get_node_vars(u_local, equations, dg, i, j), equations)
      l2_errors[element] += diff.^2 * (weights[i] * weights[j] * volume_jacobian_)
      linf_errors[element] = @. max(linf_errors[element], abs(diff))
    end
  end

  return l2_errors, linf_errors
end


function calc_error_norms(func, u, t, analyzer,
                          mesh::ParallelP4estMesh{2}, equations,
                          initial_condition, dg::DGSEM, cache, cache_analysis)
  @unpack vandermonde, weights = analyzer
  @unpack node_coordinates, inverse_jacobian = cache.elements
  @unpack u_local, u_tmp1, x_local, x_tmp1, jacobian_local, jacobian_tmp1 = cache_analysis

  # Set up data structures
  l2_error   = zero(func(get_node_vars(u, equations, dg, 1, 1, 1), equations))
  linf_error = copy(l2_error)
  volume = zero(real(mesh))

  # Iterate over all elements for error calculations
  for element in eachelement(dg, cache)
    # Interpolate solution and node locations to analysis nodes
    multiply_dimensionwise!(u_local, vandermonde, view(u,                :, :, :, element), u_tmp1)
    multiply_dimensionwise!(x_local, vandermonde, view(node_coordinates, :, :, :, element), x_tmp1)
    multiply_scalar_dimensionwise!(jacobian_local, vandermonde, inv.(view(inverse_jacobian, :, :, element)), jacobian_tmp1)

    # Calculate errors at each analysis node
    @. jacobian_local = abs(jacobian_local)

    for j in eachnode(analyzer), i in eachnode(analyzer)
      u_exact = initial_condition(get_node_coords(x_local, equations, dg, i, j), t, equations)
      diff = func(u_exact, equations) - func(get_node_vars(u_local, equations, dg, i, j), equations)
      l2_error += diff.^2 * (weights[i] * weights[j] * jacobian_local[i, j])
      linf_error = @. max(linf_error, abs(diff))
      volume += weights[i] * weights[j] * jacobian_local[i, j]
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
                               mesh::ParallelTreeMesh{2}, equations, dg::DGSEM, cache,
                               args...; normalize=true) where {Func}
  # call the method accepting a general `mesh::TreeMesh{2}`
  # TODO: MPI, we should improve this; maybe we should dispatch on `u`
  #       and create some MPI array type, overloading broadcasting and mapreduce etc.
  #       Then, this specific array type should also work well with DiffEq etc.
  local_integral = invoke(integrate_via_indices,
    Tuple{typeof(func), typeof(u), TreeMesh{2}, typeof(equations),
          typeof(dg), typeof(cache), map(typeof, args)...},
    func, u, mesh, equations, dg, cache, args..., normalize=normalize)

  # OBS! Global results are only calculated on MPI root, all other domains receive `nothing`
  global_integral = MPI.Reduce!(Ref(local_integral), +, mpi_root(), mpi_comm())
  if mpi_isroot()
    integral = convert(typeof(local_integral), global_integral[])
  else
    integral = convert(typeof(local_integral), NaN * local_integral)
  end

  return integral
end


function integrate_via_indices(func::Func, u,
                               mesh::ParallelP4estMesh{2}, equations,
                               dg::DGSEM, cache, args...; normalize=true) where {Func}
  @unpack weights = dg.basis

  # Initialize integral with zeros of the right shape
  # Pass `zero(SVector{nvariables(equations), eltype(u))}` to `func` since `u` might be empty, if the
  # current rank has no elements, see also https://github.com/trixi-framework/Trixi.jl/issues/1096.
  integral = zero(func(zero(SVector{nvariables(equations), eltype(u)}), 1, 1, 1, equations, dg, args...))
  volume = zero(real(mesh))


  # Use quadrature to numerically integrate over entire domain
  for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      volume_jacobian = abs(inv(cache.elements.inverse_jacobian[i, j, element]))
      integral += volume_jacobian * weights[i] * weights[j] * func(u, i, j, element, equations, dg, args...)
      volume += volume_jacobian * weights[i] * weights[j]
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
