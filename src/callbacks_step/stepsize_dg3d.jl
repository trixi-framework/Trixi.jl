
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


function max_dt(u::AbstractArray{<:Any,5}, t, mesh::StructuredMesh,
                constant_speed::Val{false}, equations, dg::DG, cache)
  # to avoid a division by zero if the speed vanishes everywhere,
  # e.g. for steady-state linear advection
  max_scaled_speed = nextfloat(zero(t))

  @unpack coordinates_min, coordinates_max = mesh
  
  dx = (coordinates_max[1] - coordinates_min[1]) / size(mesh, 1)
  dy = (coordinates_max[2] - coordinates_min[2]) / size(mesh, 2)
  dz = (coordinates_max[3] - coordinates_min[3]) / size(mesh, 3)

  for element in eachelement(dg, cache)
    max_λ1 = max_λ2 = max_λ3 = zero(max_scaled_speed)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, k, element)
      λ1, λ2, λ3 = max_abs_speeds(u_node, equations)
      λ1 *= 0.25 * dy * dz
      λ2 *= 0.25 * dx * dz
      λ3 *= 0.25 * dx * dy
      max_λ1 = max(max_λ1, λ1)
      max_λ2 = max(max_λ2, λ2)
      max_λ3 = max(max_λ3, λ3)
    end
    inv_jacobian = cache.elements.inverse_jacobian[element]
    max_scaled_speed = max(max_scaled_speed, inv_jacobian * (max_λ1 + max_λ2 + max_λ3))
  end

  return 2 / (nnodes(dg) * max_scaled_speed)
end


function max_dt(u::AbstractArray{<:Any,5}, t, mesh::StructuredMesh,
                constant_speed::Val{true}, equations, dg::DG, cache)
  # to avoid a division by zero if the speed vanishes everywhere,
  # e.g. for steady-state linear advection
  max_scaled_speed = nextfloat(zero(t))

  @unpack coordinates_min, coordinates_max = mesh
  dx = (coordinates_max[1] - coordinates_min[1]) / size(mesh, 1)
  dy = (coordinates_max[2] - coordinates_min[2]) / size(mesh, 2)
  dz = (coordinates_max[3] - coordinates_min[3]) / size(mesh, 3)

  for element in eachelement(dg, cache)
    max_λ1, max_λ2, max_λ3 = max_abs_speeds(equations)
    max_λ1 *= 0.25 * dy * dz
    max_λ2 *= 0.25 * dx * dz
    max_λ3 *= 0.25 * dx * dy
    inv_jacobian = cache.elements.inverse_jacobian[element]
    max_scaled_speed = max(max_scaled_speed, inv_jacobian * (max_λ1 + max_λ2 + max_λ3))
  end

  return 2 / (nnodes(dg) * max_scaled_speed)
end

