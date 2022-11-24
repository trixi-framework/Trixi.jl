# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


function max_dt(u, t, mesh::TreeMesh{2},
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


function max_dt(u, t, mesh::TreeMesh{2},
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

@inline function max_dt(u, t, mesh::Union{TreeMesh,StructuredMesh},
                        constant_speed::Val{false}, equations, semi, dg::DG, cache, indicator::IndicatorMCL)
  @unpack inverse_weights = dg.basis
  @trixi_timeit timer() "calc_lambda!" calc_lambda!(u, t, mesh, equations, dg, cache, indicator, semi.boundary_conditions)
  @unpack lambda1, lambda2 = indicator.cache.ContainerShockCapturingIndicator

  maxdt = typemax(eltype(u))
  if indicator.indicator_smooth
    @unpack element_ids_dg, element_ids_dgfv = cache
    alpha_element = @trixi_timeit timer() "element-wise blending factors" indicator.IndicatorHG(u, mesh, equations, dg, cache)
    pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha_element, dg, cache)
  else
    element_ids_dgfv = eachelement(dg, cache)
  end

  for idx_element in eachindex(element_ids_dgfv)
    element = element_ids_dgfv[idx_element]
    if mesh isa TreeMesh
      J = 1 / cache.elements.inverse_jacobian[element]
    end
    for j in eachnode(dg), i in eachnode(dg)
      if mesh isa StructuredMesh{2}
        J = 1 / cache.elements.inverse_jacobian[i, j, element]
      end
      denom = inverse_weights[i] * (lambda1[i, j, element] + lambda1[i+1, j, element]) +
              inverse_weights[j] * (lambda2[i, j, element] + lambda2[i, j+1, element])
      maxdt = min(maxdt, J / denom)
    end
  end

  if indicator.indicator_smooth && !isempty(element_ids_dg)
    maxdt = min(maxdt,
                max_dt_RK(u, t, mesh, constant_speed, equations, dg, cache, indicator, element_ids_dg))
  end

  return maxdt
end

@inline function max_dt_RK(u, t, mesh::TreeMesh2D, constant_speed, equations, dg::DG, cache, indicator, element_ids_dg)
  max_scaled_speed = nextfloat(zero(t))
  for idx_element in eachindex(element_ids_dg)
    element = element_ids_dg[idx_element]
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

@inline function max_dt_RK(u, t, mesh::StructuredMesh{2}, constant_speed, equations, dg::DG, cache, indicator, element_ids_dg)
  @unpack contravariant_vectors, inverse_jacobian = cache.elements

  max_scaled_speed = nextfloat(zero(t))
  for element in element_ids_dg
    max_λ1 = max_λ2 = zero(max_scaled_speed)
    for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element)
      λ1, λ2 = max_abs_speeds(u_node, equations)

      # Local speeds transformed to the reference element
      Ja11, Ja12     = get_contravariant_vector(1, contravariant_vectors, i, j, element)
      λ1_transformed = abs(Ja11 * λ1 + Ja12 * λ2)
      Ja21, Ja22     = get_contravariant_vector(2, contravariant_vectors, i, j, element)
      λ2_transformed = abs(Ja21 * λ1 + Ja22 * λ2)

      inv_jacobian = abs(inverse_jacobian[i, j, element])

      max_λ1 = max(max_λ1, λ1_transformed * inv_jacobian)
      max_λ2 = max(max_λ2, λ2_transformed * inv_jacobian)
    end
    max_scaled_speed = max(max_scaled_speed, max_λ1 + max_λ2)
  end

  return 2 / (nnodes(dg) * max_scaled_speed)
end


function max_dt(u, t, mesh::ParallelTreeMesh{2},
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


function max_dt(u, t, mesh::ParallelTreeMesh{2},
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


function max_dt(u, t, mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                constant_speed::Val{false}, equations, dg::DG, cache)
  # to avoid a division by zero if the speed vanishes everywhere,
  # e.g. for steady-state linear advection
  max_scaled_speed = nextfloat(zero(t))

  @unpack contravariant_vectors, inverse_jacobian = cache.elements

  for element in eachelement(dg, cache)
    max_λ1 = max_λ2 = zero(max_scaled_speed)
    for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element)
      λ1, λ2 = max_abs_speeds(u_node, equations)

      # Local speeds transformed to the reference element
      Ja11, Ja12     = get_contravariant_vector(1, contravariant_vectors, i, j, element)
      λ1_transformed = abs(Ja11 * λ1 + Ja12 * λ2)
      Ja21, Ja22     = get_contravariant_vector(2, contravariant_vectors, i, j, element)
      λ2_transformed = abs(Ja21 * λ1 + Ja22 * λ2)

      inv_jacobian = abs(inverse_jacobian[i, j, element])

      max_λ1 = max(max_λ1, λ1_transformed * inv_jacobian)
      max_λ2 = max(max_λ2, λ2_transformed * inv_jacobian)
    end

    max_scaled_speed = max(max_scaled_speed, max_λ1 + max_λ2)
  end

  return 2 / (nnodes(dg) * max_scaled_speed)
end


function max_dt(u, t, mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                constant_speed::Val{true}, equations, dg::DG, cache)
  @unpack contravariant_vectors, inverse_jacobian = cache.elements

  # to avoid a division by zero if the speed vanishes everywhere,
  # e.g. for steady-state linear advection
  max_scaled_speed = nextfloat(zero(t))

  max_λ1, max_λ2 = max_abs_speeds(equations)

  for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      # Local speeds transformed to the reference element
      Ja11, Ja12     = get_contravariant_vector(1, contravariant_vectors, i, j, element)
      λ1_transformed = abs(Ja11 * max_λ1 + Ja12 * max_λ2)
      Ja21, Ja22     = get_contravariant_vector(2, contravariant_vectors, i, j, element)
      λ2_transformed = abs(Ja21 * max_λ1 + Ja22 * max_λ2)

      inv_jacobian = abs(inverse_jacobian[i, j, element])
      max_scaled_speed = max(max_scaled_speed, inv_jacobian * (λ1_transformed + λ2_transformed))
    end
  end

  return 2 / (nnodes(dg) * max_scaled_speed)
end


function max_dt(u, t, mesh::ParallelP4estMesh{2},
                constant_speed::Val{false}, equations, dg::DG, cache)
  # call the method accepting a general `mesh::P4estMesh{2}`
  # TODO: MPI, we should improve this; maybe we should dispatch on `u`
  #       and create some MPI array type, overloading broadcasting and mapreduce etc.
  #       Then, this specific array type should also work well with DiffEq etc.
  dt = invoke(max_dt,
    Tuple{typeof(u), typeof(t), P4estMesh{2},
          typeof(constant_speed), typeof(equations), typeof(dg), typeof(cache)},
    u, t, mesh, constant_speed, equations, dg, cache)
  dt = MPI.Allreduce!(Ref(dt), min, mpi_comm())[]

  return dt
end


function max_dt(u, t, mesh::ParallelP4estMesh{2},
                constant_speed::Val{true}, equations, dg::DG, cache)
  # call the method accepting a general `mesh::P4estMesh{2}`
  # TODO: MPI, we should improve this; maybe we should dispatch on `u`
  #       and create some MPI array type, overloading broadcasting and mapreduce etc.
  #       Then, this specific array type should also work well with DiffEq etc.
  dt = invoke(max_dt,
    Tuple{typeof(u), typeof(t), P4estMesh{2},
          typeof(constant_speed), typeof(equations), typeof(dg), typeof(cache)},
    u, t, mesh, constant_speed, equations, dg, cache)
  dt = MPI.Allreduce!(Ref(dt), min, mpi_comm())[]

  return dt
end


end # @muladd
