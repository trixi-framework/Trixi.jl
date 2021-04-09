
function max_dt(u::AbstractArray{<:Any,4}, t, mesh::TreeMesh{2},
                constant_speed::Val{false}, equations, dg::DG, cache)
  # to avoid a division by zero if the speed vanishes everywhere,
  # e.g. for steady-state linear advection
  max_scaled_speed = nextfloat(zero(t))

  for element in eachelement(dg, cache)
    max_λ1 = max_λ2 = zero(max_scaled_speed)
    for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element)
      λ1, λ2 = max_abs_speeds(u_node, equations)
      max_λ1 = max(max_λ1, λ1)
      max_λ2 = max(max_λ2, λ2)
    end
    inv_jacobian = cache.elements.inverse_jacobian[element]
    max_scaled_speed = max(max_scaled_speed, inv_jacobian * (max_λ1 + max_λ2))
  end

  return 2 / (nnodes(dg) * max_scaled_speed)
end


function max_dt(u::AbstractArray{<:Any,4}, t, mesh::TreeMesh{2},
                constant_speed::Val{true}, equations, dg::DG, cache)
  # to avoid a division by zero if the speed vanishes everywhere,
  # e.g. for steady-state linear advection
  max_scaled_speed = nextfloat(zero(t))

  for element in eachelement(dg, cache)
    max_λ1, max_λ2 = max_abs_speeds(equations)
    inv_jacobian = cache.elements.inverse_jacobian[element]
    max_scaled_speed = max(max_scaled_speed, inv_jacobian * (max_λ1 + max_λ2))
  end

  return 2 / (nnodes(dg) * max_scaled_speed)
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


function max_dt(u::AbstractArray{<:Any,4}, t, mesh::CurvedMesh{2},
                constant_speed::Val{false}, equations, dg::DG, cache)
  # to avoid a division by zero if the speed vanishes everywhere,
  # e.g. for steady-state linear advection
  max_scaled_speed = nextfloat(zero(t))

  @unpack metric_terms, inverse_jacobian = cache.elements

  for element in eachelement(dg, cache)
    max_λ1 = max_λ2 = zero(max_scaled_speed)
    for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element)
      λ1, λ2 = max_abs_speeds(u_node, equations)

      # Local speeds transformed to the reference element
      λ1_transformed = abs( metric_terms[2, 2, i, j, element] * λ1 - metric_terms[1, 2, i, j, element] * λ2)
      λ2_transformed = abs(-metric_terms[2, 1, i, j, element] * λ1 + metric_terms[1, 1, i, j, element] * λ2)

      inv_jacobian = cache.elements.inverse_jacobian[i, j, element]

      max_λ1 = max(max_λ1, λ1_transformed * inv_jacobian)
      max_λ2 = max(max_λ2, λ2_transformed * inv_jacobian)
    end
    
    max_scaled_speed = max(max_scaled_speed, max_λ1 + max_λ2)
  end

  return 2 / (nnodes(dg) * max_scaled_speed)
end


function max_dt(u::AbstractArray{<:Any,4}, t, mesh::CurvedMesh{2},
                constant_speed::Val{true}, equations, dg::DG, cache)
  @unpack metric_terms, inverse_jacobian = cache.elements

  # to avoid a division by zero if the speed vanishes everywhere,
  # e.g. for steady-state linear advection
  max_scaled_speed = nextfloat(zero(t))

  max_λ1, max_λ2 = max_abs_speeds(equations)

  for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      # Local speeds transformed to the reference element
      λ1_transformed = abs( metric_terms[2, 2, i, j, element] * max_λ1 - metric_terms[1, 2, i, j, element] * max_λ2)
      λ2_transformed = abs(-metric_terms[2, 1, i, j, element] * max_λ1 + metric_terms[1, 1, i, j, element] * max_λ2)

      inv_jacobian = inverse_jacobian[i, j, element]
      max_scaled_speed = max(max_scaled_speed, inv_jacobian * (λ1_transformed + λ2_transformed))
    end
  end

  return 2 / (nnodes(dg) * max_scaled_speed)
end
