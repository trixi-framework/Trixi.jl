# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


function calc_error_norms(func, u, t, analyzer,
                          mesh::ParallelTreeMesh{2}, equations, initial_condition,
                          dg::DGSEM, cache, cache_analysis)
  # call the method accepting a general `mesh::TreeMesh{2}`
  # TODO: MPI, we should improve this; maybe we should dispatch on `u`
  #       and create some MPI array type, overloading broadcasting and mapreduce etc.
  #       Then, this specific array type should also work well with DiffEq etc.
  l2_errors, linf_errors = invoke(calc_error_norms_per_element,
    Tuple{typeof(func), typeof(u), typeof(t), typeof(analyzer), TreeMesh{2},
          typeof(equations), typeof(initial_condition), typeof(dg), typeof(cache),
          typeof(cache_analysis)},
    func, u, t, analyzer, mesh, equations, initial_condition, dg, cache, cache_analysis)

  nvars = length(l2_errors[1])
  T = eltype(l2_errors)

  # Convert errors from Vector of SVectors to Matrix for MPI communication
  FT = eltype(l2_errors[1])
  l2_errors = reshape(reinterpret(FT, l2_errors), nvars, :) |> Matrix
  linf_errors = reshape(reinterpret(FT, linf_errors), nvars, :) |> Matrix

  # Collect local error norms of all elements on root process
  if mpi_isroot()
    global_l2_errors = zeros(eltype(l2_errors), nvars, cache.mpi_cache.n_elements_global)
    global_linf_errors = similar(global_l2_errors)

    n_elements_by_rank = parent(cache.mpi_cache.n_elements_by_rank) # convert OffsetArray to Array
    l2_buf = MPI.VBuffer(global_l2_errors, nvars*n_elements_by_rank)
    linf_buf = MPI.VBuffer(global_linf_errors, nvars*n_elements_by_rank)
    MPI.Gatherv!(l2_errors, l2_buf, mpi_root(), mpi_comm())
    MPI.Gatherv!(linf_errors, linf_buf, mpi_root(), mpi_comm())
  else
    MPI.Gatherv!(l2_errors, nothing, mpi_root(), mpi_comm())
    MPI.Gatherv!(linf_errors, nothing, mpi_root(), mpi_comm())
  end

  # Aggregate element error norms on root process
  if mpi_isroot()
    # Convert from Matrix to Vector of SVectors
    global_l2_errors = reinterpret(T, vec(global_l2_errors))
    global_linf_errors = reinterpret(T, vec(global_linf_errors))

    # Aggregate element errors
    # sum(global_l2_errors) does not produce the same result as in the serial case, while
    # sum(global_l2_errors[:]) produces the same results but leads to unnecessary allocations.
    # Therefore we use a hand-written loop here
    l2_error = zero(T)
    for error in global_l2_errors
      l2_error += error
    end
    linf_error = reduce((x, y) -> max.(x, y), global_linf_errors)

    # For L2 error, divide by total volume
    total_volume_ = total_volume(mesh)
    l2_error = @. sqrt(l2_error / total_volume_)
  else
    l2_error = convert(T, NaN * l2_errors[:, 1])
    linf_error = convert(T, NaN * linf_errors[:, 1])
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


end # @muladd
