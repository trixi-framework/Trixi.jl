# Container is for both `TreeMesh` and `P4estMesh`
mutable struct ParabolicContainer2D{uEltype <: Real,
                                    ArrayuEltype4D <: AbstractArray{uEltype, 4},
                                    VectoruEltype <: AbstractVector{uEltype}} <:
               AbstractContainer
    # [variables, nodes, nodes, elements]
    u_transformed::ArrayuEltype4D
    # ([variables, nodes, nodes, elements],
    #  [variables, nodes, nodes, elements])
    gradients::NTuple{2, ArrayuEltype4D}
    # ([variables, nodes, nodes, elements],
    #  [variables, nodes, nodes, elements])
    flux_parabolic::NTuple{2, ArrayuEltype4D}

    # internal `resize!`able storage
    _u_transformed::VectoruEltype
    # Use Tuple for outer, fixed-size datastructure
    _gradients::Tuple{VectoruEltype, VectoruEltype}
    _flux_parabolic::Tuple{VectoruEltype, VectoruEltype}
end

function ParabolicContainer2D{uEltype}(n_vars::Integer, n_nodes::Integer,
                                       n_elements::Integer) where {uEltype <: Real}
    new_vector() = Vector{uEltype}(undef, n_vars * n_nodes^2 * n_elements)
    # Wrap the internal storage to avoid allocating the memory twice; see
    # `unsafe_wrap_or_alloc` for the handling of non-`isbits` element types.
    wrap(vector) = unsafe_wrap_or_alloc(Array, vector,
                                        (n_vars, n_nodes, n_nodes, n_elements))

    _u_transformed = new_vector()
    _gradients = (new_vector(), new_vector())
    _flux_parabolic = (new_vector(), new_vector())
    u_transformed = wrap(_u_transformed)
    gradients = (wrap(_gradients[1]), wrap(_gradients[2]))
    flux_parabolic = (wrap(_flux_parabolic[1]), wrap(_flux_parabolic[2]))

    return ParabolicContainer2D{uEltype, Array{uEltype, 4},
                                Vector{uEltype}}(u_transformed, gradients,
                                                 flux_parabolic, _u_transformed,
                                                 _gradients, _flux_parabolic)
end

function init_parabolic_container_2d(n_vars::Integer, n_nodes::Integer,
                                     n_elements::Integer,
                                     ::Type{uEltype}) where {uEltype <: Real}
    return ParabolicContainer2D{uEltype}(n_vars, n_nodes, n_elements)
end

function Adapt.parent_type(::Type{<:ParabolicContainer2D{<:Any, <:Any,
                                                         VectoruEltype}}) where {VectoruEltype}
    return VectoruEltype
end

# Only one-dimensional `Array`s are `resize!`able in Julia.
# Hence, we use `Vector`s as internal storage and `resize!`
# them whenever needed. Then, we reuse the same memory by
# `unsafe_wrap`ping multi-dimensional `Array`s around the
# internal storage.
function Base.resize!(parabolic_container::ParabolicContainer2D, equations, dg, cache)
    @unpack _u_transformed, _gradients, _flux_parabolic = parabolic_container
    ArrayType = storage_type(parabolic_container)

    capacity = nvariables(equations) * nnodes(dg)^2 * nelements(dg, cache)
    resize!(_u_transformed, capacity)
    for dim in 1:2
        resize!(_gradients[dim], capacity)
        resize!(_flux_parabolic[dim], capacity)
    end

    array_size = (nvariables(equations), nnodes(dg), nnodes(dg), nelements(dg, cache))
    parabolic_container.u_transformed = unsafe_wrap_or_alloc(ArrayType, _u_transformed,
                                                             array_size)
    parabolic_container.gradients = (unsafe_wrap_or_alloc(ArrayType, _gradients[1],
                                                          array_size),
                                     unsafe_wrap_or_alloc(ArrayType, _gradients[2],
                                                          array_size))
    parabolic_container.flux_parabolic = (unsafe_wrap_or_alloc(ArrayType,
                                                               _flux_parabolic[1],
                                                               array_size),
                                          unsafe_wrap_or_alloc(ArrayType,
                                                               _flux_parabolic[2],
                                                               array_size))

    return nothing
end

# Adapt the parabolic container to a different storage type (e.g., for GPUs)
# and/or real type. The (scratch) data of the multi-dimensional arrays is not
# preserved. Instead, they are re-created by wrapping the adapted internal storage,
# similar to the other containers.
function Adapt.adapt_structure(to, parabolic_container::ParabolicContainer2D)
    _u_transformed = adapt(to, parabolic_container._u_transformed)
    _gradients = map(x -> adapt(to, x), parabolic_container._gradients)
    _flux_parabolic = map(x -> adapt(to, x), parabolic_container._flux_parabolic)

    array_size = size(parabolic_container.u_transformed)
    u_transformed = unsafe_wrap_or_alloc(to, _u_transformed, array_size)
    gradients = map(x -> unsafe_wrap_or_alloc(to, x, array_size), _gradients)
    flux_parabolic = map(x -> unsafe_wrap_or_alloc(to, x, array_size), _flux_parabolic)

    return ParabolicContainer2D{eltype(_u_transformed), typeof(u_transformed),
                                typeof(_u_transformed)}(u_transformed, gradients,
                                                        flux_parabolic,
                                                        _u_transformed, _gradients,
                                                        _flux_parabolic)
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
