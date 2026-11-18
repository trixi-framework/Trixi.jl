# Container is only for `P4estMesh`
mutable struct P4estParabolicGradientBoundaryContainer2D{uEltype <: Real} <:
               AbstractParabolicGradientBoundaryContainer2D
    # ([variables, noes, boundaries],
    #  [variables, nodes, boundaries])
    gradients::NTuple{2, Array{uEltype, 3}}
    # internal `resize!`able storage. Use tuple for outer, fixed-size datastructure.
    _gradients::Tuple{Vector{uEltype}, Vector{uEltype}}

    function P4estParabolicGradientBoundaryContainer2D{uEltype}(n_boundaries::Integer,
                                                                n_variables,
                                                                n_nodes) where {
                                                                                uEltype <:
                                                                                Real}
        _gradients_1 = Vector{uEltype}(undef, n_variables * n_nodes * n_boundaries)
        _gradients_2 = Vector{uEltype}(undef, n_variables * n_nodes * n_boundaries)

        _gradients = (_gradients_1, _gradients_2)

        gradients_1 = unsafe_wrap(Array, pointer(_gradients_1),
                                  (n_variables, n_nodes, n_boundaries))

        gradients_2 = unsafe_wrap(Array, pointer(_gradients_2),
                                  (n_variables, n_nodes, n_boundaries))

        gradients = (gradients_1, gradients_2)

        return new(gradients, _gradients)
    end
end

function init_parabolic_gradient_boundary_container_2d(mesh::P4estMesh{2},
                                                       n_vars::Integer, n_nodes::Integer,
                                                       n_boundaries::Integer,
                                                       ::Type{uEltype}) where {uEltype <:
                                                                               Real}
    return P4estParabolicGradientBoundaryContainer2D{uEltype}(n_boundaries, n_vars, n_nodes)
end

function Base.resize!(gradients_at_boundaries_container::P4estParabolicGradientBoundaryContainer2D,
                      equations, dg, cache)
    @unpack boundaries = cache
    capacity = nvariables(equations) * nnodes(dg) * nboundaries(boundaries)

    resize!(gradients_at_boundaries_container._gradients[1], capacity)
    resize!(gradients_at_boundaries_container._gradients[2], capacity)

    gadients_1 = unsafe_wrap(Array,
                             pointer(gradients_at_boundaries_container._gradients[1]),
                             (nvariables(equations), nnodes(dg),
                              nboundaries(boundaries)))
    gadients_2 = unsafe_wrap(Array,
                             pointer(gradients_at_boundaries_container._gradients[2]),
                             (nvariables(equations), nnodes(dg),
                              nboundaries(boundaries)))

    gradients_at_boundaries_container.gradients = (gadients_1, gadients_2)

    return nothing
end
