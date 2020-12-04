function apply_collision!(u::AbstractArray{<:Any,4}, dt, collision_op, 
                          mesh, equations, dg::DG, cache)

  Threads.@threads for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element)
      update = collision_op(u_node, dt, equations)
      add_to_node_vars!(u, update, equations, dg, i, j, element)
    end
  end

  return nothing
end
