
function max_dt(u::AbstractArray{<:Any,3}, t, mesh::TreeMesh{1},
                constant_speed::Val{false}, equations, dg::DG, cache)
  # Use `nextfloat` to avoid division by zero if lambda is ~zero (e.g., linear scalar advection with
  # vanishing velocity)
  max_λ1 = nextfloat(zero(t))

  for element in eachelement(dg, cache)
    inv_jacobian = cache.elements.inverse_jacobian[element]
    for i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, element)
      λ1, = max_abs_speeds(u_node, equations)
      max_λ1 = max(max_λ1, inv_jacobian * λ1)
    end
  end

  return 2 / (nnodes(dg) * (max_λ1))
end


function max_dt(u::AbstractArray{<:Any,3}, t, mesh::TreeMesh{1},
                constant_speed::Val{true}, equations, dg::DG, cache)
  # Use `nextfloat` to avoid division by zero if lambda is ~zero (e.g., linear scalar advection with
  # vanishing velocity)
  max_λ1 = nextfloat(zero(t))

  for element in eachelement(dg, cache)
    inv_jacobian = cache.elements.inverse_jacobian[element]
    λ1, = max_abs_speeds(equations)
    max_λ1 = max(max_λ1, inv_jacobian * λ1)
  end

  return 2 / (nnodes(dg) * (max_λ1))
end

