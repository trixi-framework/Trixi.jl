
function max_dt(u::AbstractArray{<:Any,5}, t, mesh::TreeMesh{3},
                constant_speed::Val{false}, equations, dg::DG, cache)
  # to avoid a division by zero if the speed vanishes everywhere,
  # e.g. for steady-state linear advection
  max_scaled_speed = nextfloat(zero(t))

  for element in eachelement(dg, cache)
    max_λ1 = max_λ2 = max_λ3 = zero(max_scaled_speed)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, k, element)
      λ1, λ2, λ3 = max_abs_speeds(u_node, equations)
      max_λ1 = max(max_λ1, λ1)
      max_λ2 = max(max_λ2, λ2)
      max_λ3 = max(max_λ3, λ3)
    end
    inv_jacobian = cache.elements.inverse_jacobian[element]
    max_scaled_speed = max(max_scaled_speed, inv_jacobian * (max_λ1 + max_λ2 + max_λ3))
  end

  return 2 / (nnodes(dg) * max_scaled_speed)
end


function max_dt(u::AbstractArray{<:Any,5}, t, mesh::TreeMesh{3},
                constant_speed::Val{true}, equations, dg::DG, cache)
  # to avoid a division by zero if the speed vanishes everywhere,
  # e.g. for steady-state linear advection
  max_scaled_speed = nextfloat(zero(t))

  for element in eachelement(dg, cache)
    max_λ1, max_λ2, max_λ3 = max_abs_speeds(equations)
    inv_jacobian = cache.elements.inverse_jacobian[element]
    max_scaled_speed = max(max_scaled_speed, inv_jacobian * (max_λ1 + max_λ2 + max_λ3))
  end

  return 2 / (nnodes(dg) * max_scaled_speed)
end


function max_dt(u::AbstractArray{<:Any,5}, t, mesh::CurvedMesh,
                constant_speed::Val{false}, equations, dg::DG, cache)
  # to avoid a division by zero if the speed vanishes everywhere,
  # e.g. for steady-state linear advection
  max_scaled_speed = nextfloat(zero(t))

  @unpack faces = mesh
  @unpack metric_terms = cache.elements

  for element in eachelement(dg, cache)
    max_λ1 = max_λ2 = max_λ3 = zero(max_scaled_speed)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, k, element)
      λ1, λ2, λ3 = max_abs_speeds(u_node, equations)
      λ1 *= metric_terms[2, 2, i, j, k, element] * metric_terms[3, 3, i, j, k, element]
      λ2 *= metric_terms[1, 1, i, j, k, element] * metric_terms[3, 3, i, j, k, element]
      λ3 *= metric_terms[1, 1, i, j, k, element] * metric_terms[2, 2, i, j, k, element]
      max_λ1 = max(max_λ1, λ1)
      max_λ2 = max(max_λ2, λ2)
      max_λ3 = max(max_λ3, λ3)
    end
    #TODO: Adjust for transformation when curved
    inv_jacobian = cache.elements.inverse_jacobian[1, 1, 1, element]
    max_scaled_speed = max(max_scaled_speed, inv_jacobian * (max_λ1 + max_λ2 + max_λ3))
  end

  return 2 / (nnodes(dg) * max_scaled_speed)
end


function max_dt(u::AbstractArray{<:Any,5}, t, mesh::CurvedMesh,
                constant_speed::Val{true}, equations, dg::DG, cache)
  # to avoid a division by zero if the speed vanishes everywhere,
  # e.g. for steady-state linear advection
  max_scaled_speed = nextfloat(zero(t))

  @unpack faces = mesh
  @unpack metric_terms = cache.elements

  max_λ1, max_λ2, max_λ3 = max_abs_speeds(equations)
  max_λ1 *= metric_terms[2, 2, 1, 1, 1, 1] * metric_terms[3, 3, 1, 1, 1, 1]
  max_λ2 *= metric_terms[1, 1, 1, 1, 1, 1] * metric_terms[3, 3, 1, 1, 1, 1]
  max_λ3 *= metric_terms[1, 1, 1, 1, 1, 1] * metric_terms[2, 2, 1, 1, 1, 1]
  for element in eachelement(dg, cache)
    #TODO: Adjust for transformation when curved
    inv_jacobian = cache.elements.inverse_jacobian[1, 1, 1, element]
    max_scaled_speed = max(max_scaled_speed, inv_jacobian * (max_λ1 + max_λ2 + max_λ3))
  end

  return 2 / (nnodes(dg) * max_scaled_speed)
end

