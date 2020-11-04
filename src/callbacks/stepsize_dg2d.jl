
function max_dt(u::AbstractArray{<:Any,4}, t, mesh::TreeMesh{2},
                constant_speed::Val{false}, equations, dg::DG, cache)
  max_λ1 = max_λ2 = nextfloat(zero(t))

  for element in eachelement(dg, cache)
    inv_jacobian = cache.elements.inverse_jacobian[element]
    for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element)
      λ1, λ2 = max_abs_speeds(u_node, equations)
      max_λ1 = max(max_λ1, inv_jacobian * λ1)
      max_λ2 = max(max_λ2, inv_jacobian * λ2)
    end
  end

  return 2 / (nnodes(dg) * (max_λ1 + max_λ2))
end


function max_dt(u::AbstractArray{<:Any,4}, t, mesh::TreeMesh{2},
                constant_speed::Val{true}, equations, dg::DG, cache)
  max_λ1 = max_λ2 = nextfloat(zero(t))

  for element in eachelement(dg, cache)
    inv_jacobian = cache.elements.inverse_jacobian[element]
    λ1, λ2 = max_abs_speeds(equations)
    max_λ1 = max(max_λ1, inv_jacobian * λ1)
    max_λ2 = max(max_λ2, inv_jacobian * λ2)
  end

  return 2 / (nnodes(dg) * (max_λ1 + max_λ2))
end


function max_dt(u::AbstractArray{<:Any,4}, t, mesh::ParallelTreeMesh{2},
                constant_speed::Val{false}, equations, dg::DG, cache)
  # call the method accepting a general `mesh::TreeMesh{2}`
  # TODO: MPI, we should improve this; maybe we should dispatch on `u`
  #       and create some MPI array type, overloading broadcasting and mapreduce etc.
  #       Then, this specific array type should also work well with DiffEq etc.
  dt = invoke(max_dt,
    Tuple{typeof(u), typeof(t), TreeMesh{2},
          typeof(constant_speed), typeof(equations), typeof(dg), typeof(cache)},
    u, t, mesh, constant_speed, equations, dg, cache)
  dt = MPI.Allreduce!(Ref(dt), min, mpi_comm())[]

  return dt
end


function max_dt(u::AbstractArray{<:Any,4}, t, mesh::ParallelTreeMesh{2},
                constant_speed::Val{true}, equations, dg::DG, cache)
  # call the method accepting a general `mesh::TreeMesh{2}`
  # TODO: MPI, we should improve this; maybe we should dispatch on `u`
  #       and create some MPI array type, overloading broadcasting and mapreduce etc.
  #       Then, this specific array type should also work well with DiffEq etc.
  dt = invoke(max_dt,
    Tuple{typeof(u), typeof(t), TreeMesh{2},
          typeof(constant_speed), typeof(equations), typeof(dg), typeof(cache)},
    u, t, mesh, constant_speed, equations, dg, cache)
  dt = MPI.Allreduce!(Ref(dt), min, mpi_comm())[]

  return dt
end
