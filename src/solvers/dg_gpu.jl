using Atomix: @atomic

@inline function multiply_add_to_first_axis_atomic!(u, factor, u_node::SVector{N},
                                                    indices...) where {N}
    for v in Base.OneTo(N)
        @atomic u[v, indices...] += factor * u_node[v]
    end
    return nothing
end

@kernel function volume_integral_KAkernel!(du, u, MeshT,
                                           have_nonconservative_terms, equations,
                                           volume_integral, dg::DGSEM, cache)
    element = @index(Global)
    volume_integral_kernel!(du, u, element, MeshT, have_nonconservative_terms,
                            equations, volume_integral, dg, cache)
end
