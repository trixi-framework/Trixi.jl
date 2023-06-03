# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function (steady_state_callback::SteadyStateCallback)(du, u, mesh::AbstractMesh{2},
                                                      equations, dg::DG, cache)
    @unpack abstol, reltol = steady_state_callback

    terminate = true
    for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            u_local = get_node_vars(u, equations, dg, i, j, element)
            du_local = get_node_vars(du, equations, dg, i, j, element)
            threshold = abstol + reltol * residual_steady_state(u_local, equations)
            terminate = terminate &&
                        residual_steady_state(du_local, equations) <= threshold
        end
    end

    return terminate
end
end # @muladd
