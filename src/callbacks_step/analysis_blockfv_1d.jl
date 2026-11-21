# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function integrate(func::Func, u,
                   mesh::Union{TreeMesh{1}, StructuredMesh{1}},
                   equations, dg::BlockFV, cache;
                   normalize = true) where {Func}
    integrate_via_indices(u, mesh, equations, dg, cache;
                          normalize = normalize) do u, i, element, equations, dg
        u_local = get_node_vars(u, equations, dg, i, element)
        return func(u_local, equations)
    end
end

function analyze(::typeof(entropy_timederivative), du, u, t,
                 mesh::Union{TreeMesh{1}, StructuredMesh{1}}, equations,
                 dg::BlockFV, cache)
    integrate_via_indices(u, mesh, equations, dg, cache,
                          du) do u, i, element, equations, dg, du
        u_node = get_node_vars(u, equations, dg, i, element)
        du_node = get_node_vars(du, equations, dg, i, element)
        return dot(cons2entropy(u_node, equations), du_node)
    end
end
end # @muladd
