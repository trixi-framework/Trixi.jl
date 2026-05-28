mutable struct ParabolicContainer3D{uEltype <: Real}
    u_transformed::Array{uEltype, 5}
    gradients::NTuple{3, Array{uEltype, 5}}
    flux_parabolic::NTuple{3, Array{uEltype, 5}}
    filtered_gradients::NTuple{3, Array{uEltype, 5}}
    filtered_parabolic::NTuple{3, Array{uEltype, 5}}
    tmp_gradient::NTuple{3, Array{uEltype, 5}}
    tmp_parabolic::NTuple{3, Array{uEltype, 5}}
    # internal `resize!`able storage
    _u_transformed::Vector{uEltype}
    # Use Tuple for outer, fixed-size datastructure
    _gradients::Tuple{Vector{uEltype}, Vector{uEltype}, Vector{uEltype}}
    _flux_parabolic::Tuple{Vector{uEltype}, Vector{uEltype}, Vector{uEltype}}
    _filtered_gradients::Tuple{Vector{uEltype}, Vector{uEltype}, Vector{uEltype}}
    _filtered_parabolic::Tuple{Vector{uEltype}, Vector{uEltype}, Vector{uEltype}}
    _tmp_gradient::Tuple{Vector{uEltype}, Vector{uEltype}, Vector{uEltype}}
    _tmp_parabolic::Tuple{Vector{uEltype}, Vector{uEltype}, Vector{uEltype}}

    function ParabolicContainer3D{uEltype}(n_vars::Integer, n_nodes::Integer,
                                           n_elements::Integer) where {uEltype <: Real}
        return new(Array{uEltype, 5}(undef, n_vars, n_nodes, n_nodes, n_nodes, n_elements), # `u_transformed`
                   # `gradients`
                   (Array{uEltype, 5}(undef, n_vars, n_nodes, n_nodes, n_nodes, n_elements),
                    Array{uEltype, 5}(undef, n_vars, n_nodes, n_nodes, n_nodes, n_elements),
                    Array{uEltype, 5}(undef, n_vars, n_nodes, n_nodes, n_nodes, n_elements)),
                   # `flux_parabolic`
                   (Array{uEltype, 5}(undef, n_vars, n_nodes, n_nodes, n_nodes, n_elements),
                    Array{uEltype, 5}(undef, n_vars, n_nodes, n_nodes, n_nodes, n_elements),
                    Array{uEltype, 5}(undef, n_vars, n_nodes, n_nodes, n_nodes, n_elements)),
                    # 'filtered_gradients'
                    (Array{uEltype, 5}(undef, n_vars, n_nodes, n_nodes, n_nodes, n_elements),
                    Array{uEltype, 5}(undef, n_vars, n_nodes, n_nodes, n_nodes, n_elements),
                    Array{uEltype, 5}(undef, n_vars, n_nodes, n_nodes, n_nodes, n_elements)),
                    # 'filtered_flux_parabolic' 
                    (Array{uEltype, 5}(undef, n_vars, n_nodes, n_nodes, n_nodes, n_elements),
                    Array{uEltype, 5}(undef, n_vars, n_nodes, n_nodes, n_nodes, n_elements),
                    Array{uEltype, 5}(undef, n_vars, n_nodes, n_nodes, n_nodes, n_elements)),
                    # 'tmp_gradient' 
                    (Array{uEltype, 5}(undef, n_vars, n_nodes, n_nodes, n_nodes, n_elements),
                    Array{uEltype, 5}(undef, n_vars, n_nodes, n_nodes, n_nodes, n_elements),
                    Array{uEltype, 5}(undef, n_vars, n_nodes, n_nodes, n_nodes, n_elements)),
                     # 'tmp_parabolic' 
                    (Array{uEltype, 5}(undef, n_vars, n_nodes, n_nodes, n_nodes, n_elements),
                    Array{uEltype, 5}(undef, n_vars, n_nodes, n_nodes, n_nodes, n_elements),
                    Array{uEltype, 5}(undef, n_vars, n_nodes, n_nodes, n_nodes, n_elements)),
                   # `u_transformed`
                   Vector{uEltype}(undef, n_vars * n_nodes^3 * n_elements),
                   # `_gradients`
                   (Vector{uEltype}(undef, n_vars * n_nodes^3 * n_elements),
                    Vector{uEltype}(undef, n_vars * n_nodes^3 * n_elements),
                    Vector{uEltype}(undef, n_vars * n_nodes^3 * n_elements)),
                   # `_flux_parabolic`
                   (Vector{uEltype}(undef, n_vars * n_nodes^3 * n_elements),
                    Vector{uEltype}(undef, n_vars * n_nodes^3 * n_elements),
                    Vector{uEltype}(undef, n_vars * n_nodes^3 * n_elements)),
                    # `_filtered_gradients`
                   (Vector{uEltype}(undef, n_vars * n_nodes^3 * n_elements),
                    Vector{uEltype}(undef, n_vars * n_nodes^3 * n_elements),
                    Vector{uEltype}(undef, n_vars * n_nodes^3 * n_elements)),
                   # `_filtered_flux_parabolic`
                   (Vector{uEltype}(undef, n_vars * n_nodes^3 * n_elements),
                    Vector{uEltype}(undef, n_vars * n_nodes^3 * n_elements),
                    Vector{uEltype}(undef, n_vars * n_nodes^3 * n_elements)),
                    # `_tmp_gradient`
                   (Vector{uEltype}(undef, n_vars * n_nodes^3 * n_elements),
                    Vector{uEltype}(undef, n_vars * n_nodes^3 * n_elements),
                    Vector{uEltype}(undef, n_vars * n_nodes^3 * n_elements)),
                    # `_tmp_parabolic`
                   (Vector{uEltype}(undef, n_vars * n_nodes^3 * n_elements),
                    Vector{uEltype}(undef, n_vars * n_nodes^3 * n_elements),
                    Vector{uEltype}(undef, n_vars * n_nodes^3 * n_elements)),)
    end
end

function init_parabolic_container_3d(n_vars::Integer, n_nodes::Integer,
                                     n_elements::Integer,
                                     ::Type{uEltype}) where {uEltype <: Real}
    return ParabolicContainer3D{uEltype}(n_vars, n_nodes, n_elements)
end

# Only one-dimensional `Array`s are `resize!`able in Julia.
# Hence, we use `Vector`s as internal storage and `resize!`
# them whenever needed. Then, we reuse the same memory by
# `unsafe_wrap`ping multi-dimensional `Array`s around the
# internal storage.
function Base.resize!(parabolic_container::ParabolicContainer3D, equations, dg, cache)
    capacity = nvariables(equations) * nnodes(dg)^3 * nelements(dg, cache)
    resize!(parabolic_container._u_transformed, capacity)
    for dim in 1:3
        resize!(parabolic_container._gradients[dim], capacity)
        resize!(parabolic_container._flux_parabolic[dim], capacity)
        resize!(parabolic_container._filtered_gradients[dim], capacity)
        resize!(parabolic_container._filtered_flux_parabolic[dim], capacity)
        resize!(parabolic_container._tmp_gradient[dim], capacity)
        resize!(parabolic_container._tmp_parabolic[dim], capacity)
    end

    parabolic_container.u_transformed = unsafe_wrap(Array,
                                                    pointer(parabolic_container._u_transformed),
                                                    (nvariables(equations),
                                                     nnodes(dg), nnodes(dg), nnodes(dg),
                                                     nelements(dg, cache)))

    gradients_1 = unsafe_wrap(Array,
                              pointer(parabolic_container._gradients[1]),
                              (nvariables(equations),
                               nnodes(dg), nnodes(dg), nnodes(dg),
                               nelements(dg, cache)))
    gradients_2 = unsafe_wrap(Array,
                              pointer(parabolic_container._gradients[2]),
                              (nvariables(equations),
                               nnodes(dg), nnodes(dg), nnodes(dg),
                               nelements(dg, cache)))
    gradients_3 = unsafe_wrap(Array,
                              pointer(parabolic_container._gradients[3]),
                              (nvariables(equations),
                               nnodes(dg), nnodes(dg), nnodes(dg),
                               nelements(dg, cache)))

    parabolic_container.gradients = (gradients_1, gradients_2, gradients_3)

    flux_parabolic_1 = unsafe_wrap(Array,
                                   pointer(parabolic_container._flux_parabolic[1]),
                                   (nvariables(equations),
                                    nnodes(dg), nnodes(dg), nnodes(dg),
                                    nelements(dg, cache)))
    flux_parabolic_2 = unsafe_wrap(Array,
                                   pointer(parabolic_container._flux_parabolic[2]),
                                   (nvariables(equations),
                                    nnodes(dg), nnodes(dg), nnodes(dg),
                                    nelements(dg, cache)))
    flux_parabolic_3 = unsafe_wrap(Array,
                                   pointer(parabolic_container._flux_parabolic[3]),
                                   (nvariables(equations),
                                    nnodes(dg), nnodes(dg), nnodes(dg),
                                    nelements(dg, cache)))

    parabolic_container.flux_parabolic = (flux_parabolic_1, flux_parabolic_2,
                                          flux_parabolic_3)
    
    filtered_gradients_1 = unsafe_wrap(Array,
                              pointer(parabolic_container._filtered_gradients[1]),
                              (nvariables(equations),
                               nnodes(dg), nnodes(dg), nnodes(dg),
                               nelements(dg, cache)))
    filtered_gradients_2 = unsafe_wrap(Array,
                              pointer(parabolic_container._filtered_gradients[2]),
                              (nvariables(equations),
                               nnodes(dg), nnodes(dg), nnodes(dg),
                               nelements(dg, cache)))
    filtered_gradients_3 = unsafe_wrap(Array,
                              pointer(parabolic_container._filtered_gradients[3]),
                              (nvariables(equations),
                               nnodes(dg), nnodes(dg), nnodes(dg),
                               nelements(dg, cache)))

    parabolic_container.filtered_gradients = (filtered_gradients_1, filtered_gradients_2, filtered_gradients_3)

    flux_parabolic_1 = unsafe_wrap(Array,
                                   pointer(parabolic_container._flux_parabolic[1]),
                                   (nvariables(equations),
                                    nnodes(dg), nnodes(dg), nnodes(dg),
                                    nelements(dg, cache)))
    flux_parabolic_2 = unsafe_wrap(Array,
                                   pointer(parabolic_container._filtered_flux_parabolic[2]),
                                   (nvariables(equations),
                                    nnodes(dg), nnodes(dg), nnodes(dg),
                                    nelements(dg, cache)))
    flux_parabolic_3 = unsafe_wrap(Array,
                                   pointer(parabolic_container._filtered_flux_parabolic[3]),
                                   (nvariables(equations),
                                    nnodes(dg), nnodes(dg), nnodes(dg),
                                    nelements(dg, cache)))

    parabolic_container.filtered_flux_parabolic = (filtered_flux_parabolic_1, filtered_flux_parabolic_2,
                                          filtered_flux_parabolic_3)
    
    tmp_1 = unsafe_wrap(Array,
                                   pointer(parabolic_container._tmp_gradient[1]),
                                   (nvariables(equations),
                                    nnodes(dg), nnodes(dg), nnodes(dg),
                                    nelements(dg, cache)))
    tmp_2 = unsafe_wrap(Array,
                                   pointer(parabolic_container._tmp_gradient[2]),
                                   (nvariables(equations),
                                    nnodes(dg), nnodes(dg), nnodes(dg),
                                    nelements(dg, cache)))
    tmp_3 = unsafe_wrap(Array,
                                   pointer(parabolic_container._tmp_gradient[3]),
                                   (nvariables(equations),
                                    nnodes(dg), nnodes(dg), nnodes(dg),
                                    nelements(dg, cache)))

    parabolic_container.tmp_gradient = (tmp_1, tmp_2, tmp_3)

    tmp_f1 = unsafe_wrap(Array,
                                   pointer(parabolic_container._tmp_parabolic[1]),
                                   (nvariables(equations),
                                    nnodes(dg), nnodes(dg), nnodes(dg),
                                    nelements(dg, cache)))
    tmp_f2 = unsafe_wrap(Array,
                                   pointer(parabolic_container._tmp_parabolic[2]),
                                   (nvariables(equations),
                                    nnodes(dg), nnodes(dg), nnodes(dg),
                                    nelements(dg, cache)))
    tmp_f3 = unsafe_wrap(Array,
                                   pointer(parabolic_container._tmp_parabolic[3]),
                                   (nvariables(equations),
                                    nnodes(dg), nnodes(dg), nnodes(dg),
                                    nelements(dg, cache)))

    parabolic_container.tmp_parabolic = (tmp_f1, tmp_f2, tmp_f3)

    return nothing
end
