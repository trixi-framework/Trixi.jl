@muladd begin
#! format: noindent

function integrate(func::Func, u,
                   mesh::Union{TreeMesh{2}, StructuredMesh{2}},
                   equations, dg::BlockFV, cache;
                   normalize = true) where {Func}
    integrate_via_indices(u, mesh, equations, dg, cache;
                          normalize = normalize) do u, i, j, element, equations, dg
        u_local = get_node_vars(u, equations, dg, i, j, element)
        return func(u_local, equations)
    end
end

function analyze(::typeof(entropy_timederivative), du, u, t,
                 mesh::Union{TreeMesh{2}, StructuredMesh{2}}, equations,
                 dg::BlockFV, cache)
    integrate_via_indices(u, mesh, equations, dg, cache,
                          du) do u, i, j, element, equations, dg, du
        u_node = get_node_vars(u, equations, dg, i, j, element)
        du_node = get_node_vars(du, equations, dg, i, j, element)
        return dot(cons2entropy(u_node, equations), du_node)
    end
end
end # @muladd
