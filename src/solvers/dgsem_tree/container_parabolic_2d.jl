# Container is for both `TreeMesh` and `P4estMesh`
mutable struct ParabolicContainer2D{uEltype <: Real}
    # [variables, nodes, nodes, elements]
    u_transformed::Array{uEltype, 4}
    # ([variables, nodes, nodes, elements],
    #  [variables, nodes, nodes, elements])
    gradients::NTuple{2, Array{uEltype, 4}}
    # ([variables, nodes, nodes, elements],
    #  [variables, nodes, nodes, elements])
    flux_parabolic::NTuple{2, Array{uEltype, 4}}

    # internal `resize!`able storage
    _u_transformed::Vector{uEltype}
    # Use Tuple for outer, fixed-size datastructure
    _gradients::Tuple{Vector{uEltype}, Vector{uEltype}}
    _flux_parabolic::Tuple{Vector{uEltype}, Vector{uEltype}}

    function ParabolicContainer2D{uEltype}(n_vars::Integer, n_nodes::Integer,
                                           n_elements::Integer) where {uEltype <: Real}
        return new(Array{uEltype, 4}(undef, n_vars, n_nodes, n_nodes, n_elements), # `u_transformed`
                   # `gradients`
                   (Array{uEltype, 4}(undef, n_vars, n_nodes, n_nodes, n_elements),
                    Array{uEltype, 4}(undef, n_vars, n_nodes, n_nodes, n_elements)),
                   # `flux_parabolic`
                   (Array{uEltype, 4}(undef, n_vars, n_nodes, n_nodes, n_elements),
                    Array{uEltype, 4}(undef, n_vars, n_nodes, n_nodes, n_elements)),
                   # `_u_transformed`
                   Vector{uEltype}(undef, n_vars * n_nodes^2 * n_elements),
                   # `_gradients`
                   (Vector{uEltype}(undef, n_vars * n_nodes^2 * n_elements),
                    Vector{uEltype}(undef, n_vars * n_nodes^2 * n_elements)),
                   # `_flux_parabolic`
                   (Vector{uEltype}(undef, n_vars * n_nodes^2 * n_elements),
                    Vector{uEltype}(undef, n_vars * n_nodes^2 * n_elements)))
    end
end

function init_parabolic_container_2d(n_vars::Integer, n_nodes::Integer,
                                     n_elements::Integer,
                                     ::Type{uEltype}) where {uEltype <: Real}
    return ParabolicContainer2D{uEltype}(n_vars, n_nodes, n_elements)
end

# Only one-dimensional `Array`s are `resize!`able in Julia.
# Hence, we use `Vector`s as internal storage and `resize!`
# them whenever needed. Then, we reuse the same memory by
# `unsafe_wrap`ping multi-dimensional `Array`s around the
# internal storage.
function Base.resize!(parabolic_container::ParabolicContainer2D, equations, dg, cache)
    capacity = nvariables(equations) * nnodes(dg)^2 * nelements(dg, cache)
    resize!(parabolic_container._u_transformed, capacity)
    for dim in 1:2
        resize!(parabolic_container._gradients[dim], capacity)
        resize!(parabolic_container._flux_parabolic[dim], capacity)
    end

    parabolic_container.u_transformed = unsafe_wrap(Array,
                                                    pointer(parabolic_container._u_transformed),
                                                    (nvariables(equations),
                                                     nnodes(dg), nnodes(dg),
                                                     nelements(dg, cache)))

    gradients_1 = unsafe_wrap(Array,
                              pointer(parabolic_container._gradients[1]),
                              (nvariables(equations),
                               nnodes(dg), nnodes(dg),
                               nelements(dg, cache)))
    gradients_2 = unsafe_wrap(Array,
                              pointer(parabolic_container._gradients[2]),
                              (nvariables(equations),
                               nnodes(dg), nnodes(dg),
                               nelements(dg, cache)))

    parabolic_container.gradients = (gradients_1, gradients_2)

    flux_parabolic_1 = unsafe_wrap(Array,
                                   pointer(parabolic_container._flux_parabolic[1]),
                                   (nvariables(equations),
                                    nnodes(dg), nnodes(dg),
                                    nelements(dg, cache)))
    flux_parabolic_2 = unsafe_wrap(Array,
                                   pointer(parabolic_container._flux_parabolic[2]),
                                   (nvariables(equations),
                                    nnodes(dg), nnodes(dg),
                                    nelements(dg, cache)))

    parabolic_container.flux_parabolic = (flux_parabolic_1, flux_parabolic_2)

    return nothing
end

abstract type AbstractParabolicGradientBoundaryContainer2D <:
              AbstractParabolicGradientBoundaryContainer end

# Container is only for `TreeMesh`
mutable struct TreeParabolicGradientBoundaryContainer2D{uEltype <: Real} <:
               AbstractParabolicGradientBoundaryContainer2D
    # ([leftright, variables, noes, boundaries],
    #  [leftright, variables, nodes, boundaries])
    gradients::NTuple{2, Array{uEltype, 4}}
    # internal `resize!`able storage. Use tuple for outer, fixed-size datastructure.
    _gradients::Tuple{Vector{uEltype}, Vector{uEltype}}

    function TreeParabolicGradientBoundaryContainer2D{uEltype}(n_boundaries::Integer,
                                                               n_variables,
                                                               n_nodes) where {
                                                                               uEltype <:
                                                                               Real}
        _gradients_1 = Vector{uEltype}(undef, 2 * n_variables * n_nodes * n_boundaries)
        _gradients_2 = Vector{uEltype}(undef, 2 * n_variables * n_nodes * n_boundaries)

        _gradients = (_gradients_1, _gradients_2)

        gradients_1 = unsafe_wrap(Array, pointer(_gradients_1),
                                  (2, n_variables, n_nodes, n_boundaries))

        gradients_2 = unsafe_wrap(Array, pointer(_gradients_2),
                                  (2, n_variables, n_nodes, n_boundaries))

        gradients = (gradients_1, gradients_2)

        return new(gradients, _gradients)
    end
end

function init_parabolic_gradient_boundary_container_2d(mesh::TreeMesh2D,
                                                       n_vars::Integer, n_nodes::Integer,
                                                       n_boundaries::Integer,
                                                       ::Type{uEltype}) where {uEltype <:
                                                                               Real}
    return TreeParabolicGradientBoundaryContainer2D{uEltype}(n_boundaries, n_vars, n_nodes)
end

function Base.resize!(gradients_at_boundaries_container::TreeParabolicGradientBoundaryContainer2D,
                      equations, dg, cache)
    @unpack boundaries = cache
    capacity = 2 * nvariables(equations) * nnodes(dg) * nboundaries(boundaries)

    resize!(gradients_at_boundaries_container._gradients[1], capacity)
    resize!(gradients_at_boundaries_container._gradients[2], capacity)

    gadients_1 = unsafe_wrap(Array,
                             pointer(gradients_at_boundaries_container._gradients[1]),
                             (2, nvariables(equations), nnodes(dg),
                              nboundaries(boundaries)))
    gadients_2 = unsafe_wrap(Array,
                             pointer(gradients_at_boundaries_container._gradients[2]),
                             (2, nvariables(equations), nnodes(dg),
                              nboundaries(boundaries)))

    gradients_at_boundaries_container.gradients = (gadients_1, gadients_2)

    return nothing
end
