
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

