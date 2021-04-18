
function max_dt(u, t, mesh::TreeMesh{1},
                constant_speed::Val{false}, equations, dg::DG, cache)
  # to avoid a division by zero if the speed vanishes everywhere,
  # e.g. for steady-state linear advection
  max_scaled_speed = nextfloat(zero(t))

  for element in eachelement(dg, cache)
    max_λ1 = zero(max_scaled_speed)
    for i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, element)
      λ1, = max_abs_speeds(u_node, equations)
      max_λ1 = max(max_λ1, λ1)
    end
    inv_jacobian = cache.elements.inverse_jacobian[element]
    max_scaled_speed = max(max_scaled_speed, inv_jacobian * max_λ1)
  end

  return 2 / (nnodes(dg) * max_scaled_speed)
end


function max_dt(u, t, mesh::TreeMesh{1},
                constant_speed::Val{true}, equations, dg::DG, cache)
  # to avoid a division by zero if the speed vanishes everywhere,
  # e.g. for steady-state linear advection
  max_scaled_speed = nextfloat(zero(t))

  for element in eachelement(dg, cache)
    max_λ1, = max_abs_speeds(equations)
    inv_jacobian = cache.elements.inverse_jacobian[element]
    max_scaled_speed = max(max_scaled_speed, inv_jacobian * max_λ1)
  end

  return 2 / (nnodes(dg) * max_scaled_speed)
end


function max_dt(u, t, mesh::CurvedMesh{1},
                constant_speed::Val{false}, equations, dg::DG, cache)
  # to avoid a division by zero if the speed vanishes everywhere,
  # e.g. for steady-state linear advection
  max_scaled_speed = nextfloat(zero(t))

  for element in eachelement(dg, cache)
    max_λ1 = zero(max_scaled_speed)

    for i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, element)
      λ1, = max_abs_speeds(u_node, equations)

      inv_jacobian = cache.elements.inverse_jacobian[i, element]

      max_λ1 = max(max_λ1, inv_jacobian * λ1)
    end

    max_scaled_speed = max(max_scaled_speed, max_λ1)
  end

  return 2 / (nnodes(dg) * max_scaled_speed)
end


function max_dt(u, t, mesh::CurvedMesh{1},
                constant_speed::Val{true}, equations, dg::DG, cache)
  # to avoid a division by zero if the speed vanishes everywhere,
  # e.g. for steady-state linear advection
  max_scaled_speed = nextfloat(zero(t))

  for element in eachelement(dg, cache)
    max_λ1, = max_abs_speeds(equations)

    for i in eachnode(dg)
      inv_jacobian = cache.elements.inverse_jacobian[i, element]
      max_scaled_speed = max(max_scaled_speed, inv_jacobian * max_λ1)
    end
  end

  return 2 / (nnodes(dg) * max_scaled_speed)
end
