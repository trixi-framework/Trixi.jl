# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function apply_collision!(u, dt, collision_op,
                          mesh::AbstractMesh{3}, equations, dg::DG, cache)
    @threaded for element in eachelement(dg, cache)
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, k, element)
            update = collision_op(u_node, dt, equations)
            add_to_node_vars!(u, update, equations, dg, i, j, k, element)
        end
    end

    return nothing
end
end # @muladd
