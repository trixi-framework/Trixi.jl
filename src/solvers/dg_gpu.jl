using Atomix: @atomic

# Use this function instead of `add_to_first_axis!` to speed up
# multiply-and-add-to-node-vars operations
# See https://github.com/trixi-framework/Trixi.jl/pull/643
@inline function multiply_add_to_first_axis_atomic!(u, factor, u_node::SVector{N},
                                                    indices...) where {N}
    for v in Base.OneTo(N)
        @atomic u[v, indices...] += factor * u_node[v]
    end
    return nothing
end
