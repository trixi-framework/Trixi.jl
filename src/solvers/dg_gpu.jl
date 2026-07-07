using Atomix: @atomic

# Atomix.jl requires updating operators such as `+=` to generate atomic operations and avoid race conditions.
# In particular, we cannot use `@muladd` as we do typically in the standard
# CPU code.
# See https://github.com/trixi-framework/Trixi.jl/pull/3015#discussion_r3342476726
@inline function multiply_add_to_first_axis_atomic!(u, factor, u_node::SVector{N},
                                                    indices...) where {N}
    for v in Base.OneTo(N)
        @atomic u[v, indices...] += factor * u_node[v]
    end
    return nothing
end

function calc_volume_integral!(backend::Backend, du, u, mesh,
                               have_nonconservative_terms,
                               have_aux_node_vars, equations,
                               volume_integral, dg::DGSEM, cache)
    nelements(dg, cache) == 0 && return nothing
    kernel! = volume_integral_KAkernel!(backend)
    kernel_cache = kernel_filter_cache(cache)
    kernel!(du, u, typeof(mesh), have_nonconservative_terms, have_aux_node_vars, equations,
            volume_integral, dg, kernel_cache,
            ndrange = nelements(dg, cache))
    return nothing
end

@kernel function volume_integral_KAkernel!(du, u, MeshT,
                                           have_nonconservative_terms,
                                           have_aux_node_vars, equations,
                                           volume_integral, dg::DGSEM, cache)
    element = @index(Global)
    volume_integral_kernel!(du, u, element, MeshT, have_nonconservative_terms,
                            have_aux_node_vars, equations, volume_integral, dg, cache)
end
