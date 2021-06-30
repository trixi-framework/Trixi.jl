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
  l2_error, linf_error = invoke(calc_error_norms,
    Tuple{typeof(func), typeof(u), typeof(t), typeof(analyzer), TreeMesh{2},
          typeof(equations), typeof(initial_condition), typeof(dg), typeof(cache),
          typeof(cache_analysis)},
    func, u, t, analyzer, mesh, equations, initial_condition, dg, cache, cache_analysis)

  # Since the local L2 norm is already normalized and square-rooted, we need to undo this first
  total_volume = mesh.tree.length_level_0^ndims(mesh)
  global_l2_error = Vector(l2_error.^2 .* total_volume)
  global_linf_error = Vector(linf_error)
  MPI.Reduce!(global_l2_error, +, mpi_root(), mpi_comm())
  MPI.Reduce!(global_linf_error, max, mpi_root(), mpi_comm())
  if mpi_isroot()
    l2_error   = convert(typeof(l2_error),   global_l2_error)
    linf_error = convert(typeof(linf_error), global_linf_error)
  else
    l2_error   = convert(typeof(l2_error),   NaN * global_l2_error)
    linf_error = convert(typeof(linf_error), NaN * global_linf_error)
  end

  l2_error = @. sqrt(l2_error / total_volume)

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
