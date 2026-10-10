mutable struct ParabolicContainer3D{uEltype <: Real,
                                    ArrayuEltype5D <: AbstractArray{uEltype, 5},
                                    VectoruEltype <: AbstractVector{uEltype}} <:
               AbstractContainer
    # [variables, nodes, nodes, nodes, elements]
    u_transformed::ArrayuEltype5D
    # ([variables, nodes, nodes, nodes, elements],
    #  [variables, nodes, nodes, nodes, elements],
    #  [variables, nodes, nodes, nodes, elements])
    gradients::NTuple{3, ArrayuEltype5D}
    # ([variables, nodes, nodes, nodes, elements],
    #  [variables, nodes, nodes, nodes, elements],
    #  [variables, nodes, nodes, nodes, elements])
    flux_parabolic::NTuple{3, ArrayuEltype5D}

    # internal `resize!`able storage
    _u_transformed::VectoruEltype
    # Use Tuple for outer, fixed-size datastructure
    _gradients::Tuple{VectoruEltype, VectoruEltype, VectoruEltype}
    _flux_parabolic::Tuple{VectoruEltype, VectoruEltype, VectoruEltype}
end

function ParabolicContainer3D{uEltype}(n_vars::Integer, n_nodes::Integer,
                                       n_elements::Integer) where {uEltype <: Real}
    new_array() = Array{uEltype, 5}(undef, n_vars, n_nodes, n_nodes, n_nodes, n_elements)
    new_vector() = Vector{uEltype}(undef, n_vars * n_nodes^3 * n_elements)

    u_transformed = new_array()
    gradients = (new_array(), new_array(), new_array())
    flux_parabolic = (new_array(), new_array(), new_array())
    _u_transformed = new_vector()
    _gradients = (new_vector(), new_vector(), new_vector())
    _flux_parabolic = (new_vector(), new_vector(), new_vector())

    return ParabolicContainer3D{uEltype, Array{uEltype, 5},
                                Vector{uEltype}}(u_transformed, gradients,
                                                 flux_parabolic, _u_transformed,
                                                 _gradients, _flux_parabolic)
end

function init_parabolic_container_3d(n_vars::Integer, n_nodes::Integer,
                                     n_elements::Integer,
                                     ::Type{uEltype}) where {uEltype <: Real}
    return ParabolicContainer3D{uEltype}(n_vars, n_nodes, n_elements)
end

function Adapt.parent_type(::Type{<:ParabolicContainer3D{<:Any, <:Any,
                                                         VectoruEltype}}) where {VectoruEltype}
    return VectoruEltype
end

# Only one-dimensional `Array`s are `resize!`able in Julia.
# Hence, we use `Vector`s as internal storage and `resize!`
# them whenever needed. Then, we reuse the same memory by
# `unsafe_wrap`ping multi-dimensional `Array`s around the
# internal storage.
function Base.resize!(parabolic_container::ParabolicContainer3D, equations, dg, cache)
    @unpack _u_transformed, _gradients, _flux_parabolic = parabolic_container
    ArrayType = storage_type(parabolic_container)

    capacity = nvariables(equations) * nnodes(dg)^3 * nelements(dg, cache)
    resize!(_u_transformed, capacity)
    for dim in 1:3
        resize!(_gradients[dim], capacity)
        resize!(_flux_parabolic[dim], capacity)
    end

    array_size = (nvariables(equations), nnodes(dg), nnodes(dg), nnodes(dg),
                  nelements(dg, cache))
    parabolic_container.u_transformed = unsafe_wrap_or_alloc(ArrayType, _u_transformed,
                                                             array_size)
    parabolic_container.gradients = (unsafe_wrap_or_alloc(ArrayType, _gradients[1],
                                                          array_size),
                                     unsafe_wrap_or_alloc(ArrayType, _gradients[2],
                                                          array_size),
                                     unsafe_wrap_or_alloc(ArrayType, _gradients[3],
                                                          array_size))
    parabolic_container.flux_parabolic = (unsafe_wrap_or_alloc(ArrayType,
                                                               _flux_parabolic[1],
                                                               array_size),
                                          unsafe_wrap_or_alloc(ArrayType,
                                                               _flux_parabolic[2],
                                                               array_size),
                                          unsafe_wrap_or_alloc(ArrayType,
                                                               _flux_parabolic[3],
                                                               array_size))

    return nothing
end

# Adapt the parabolic container to a different storage type (e.g., for GPUs)
# and/or real type. The (scratch) data of the multi-dimensional arrays is not
# preserved. Instead, they are re-created by wrapping the adapted internal storage,
# similar to the other containers.
function Adapt.adapt_structure(to, parabolic_container::ParabolicContainer3D)
    _u_transformed = adapt(to, parabolic_container._u_transformed)
    _gradients = map(x -> adapt(to, x), parabolic_container._gradients)
    _flux_parabolic = map(x -> adapt(to, x), parabolic_container._flux_parabolic)

    array_size = size(parabolic_container.u_transformed)
    u_transformed = unsafe_wrap_or_alloc(to, _u_transformed, array_size)
    gradients = map(x -> unsafe_wrap_or_alloc(to, x, array_size), _gradients)
    flux_parabolic = map(x -> unsafe_wrap_or_alloc(to, x, array_size), _flux_parabolic)

    return ParabolicContainer3D{eltype(_u_transformed), typeof(u_transformed),
                                typeof(_u_transformed)}(u_transformed, gradients,
                                                        flux_parabolic,
                                                        _u_transformed, _gradients,
                                                        _flux_parabolic)
end
