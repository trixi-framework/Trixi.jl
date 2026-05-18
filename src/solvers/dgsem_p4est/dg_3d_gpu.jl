# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@kernel function calc_sources_KAkernel!(du, u, t, source_terms,
                                        node_coordinates,
                                        equations::AbstractEquations{3}, dg, cache)
    i, j, k, element = @index(Global, NTuple)
    u_local = get_node_vars(u, equations, dg, i, j, k, element)
    x_local = get_node_coords(node_coordinates, equations, dg, i, j, k, element)

    du_local = source_terms(u_local, x_local, t, equations)

    add_to_node_vars!(du, du_local, equations, dg, i, j, k, element)
end

function calc_sources!(backend::Backend, du, u, t, source_terms,
                       equations::AbstractEquations{3}, dg::DG, cache)
    nelements(dg, cache) == 0 && return nothing
    @unpack node_coordinates = cache.elements
    kernel_cache = kernel_filter_cache(cache)
    kernel! = calc_sources_KAkernel!(backend)
    kernel!(du, u, t, source_terms, node_coordinates, equations, dg, kernel_cache,
            ndrange = (nnodes(dg), nnodes(dg), nnodes(dg), nelements(dg, cache)))

    return nothing
end

function calc_sources!(backend::Backend, du, u, t, source_terms::Nothing,
                       equations::AbstractEquations{3}, dg::DG, cache)
    return nothing
end
end #muladd
