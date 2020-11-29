function apply_collision!(collision_op, u, dt, mesh, equations, dg::DG, cache)

  Threads.@threads for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element)
      update = collision_op(u_node, dt, equation)
      add_to_node_vars!(u, u_node, equations, dg, i, j, element)
    end
  end

  return nothing
end

