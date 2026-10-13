mutable struct ParabolicContainer1D{uEltype <: Real,
                                    ArrayuEltype3D <: AbstractArray{uEltype, 3},
                                    VectoruEltype <: AbstractVector{uEltype}} <:
               AbstractContainer
    u_transformed::ArrayuEltype3D  # [variables, nodes, elements]
    gradients::ArrayuEltype3D      # [variables, nodes, elements]
    flux_parabolic::ArrayuEltype3D # [variables, nodes, elements]

    # internal `resize!`able storage
    _u_transformed::VectoruEltype
    _gradients::VectoruEltype
    _flux_parabolic::VectoruEltype
end

function ParabolicContainer1D{uEltype}(n_vars::Integer, n_nodes::Integer,
                                       n_elements::Integer) where {uEltype <: Real}
    new_vector() = Vector{uEltype}(undef, n_vars * n_nodes * n_elements)
    # Wrap the internal storage to avoid allocating the memory twice; this returns
    # `VectorBackedArray`s (see `wrap_storage`).
    wrap(vector) = wrap_storage(vector, (n_vars, n_nodes, n_elements))

    _u_transformed = new_vector()
    _gradients = new_vector()
    _flux_parabolic = new_vector()
    u_transformed = wrap(_u_transformed)
    gradients = wrap(_gradients)
    flux_parabolic = wrap(_flux_parabolic)

    return ParabolicContainer1D{uEltype, VectorBackedArray{uEltype, 3},
                                Vector{uEltype}}(u_transformed, gradients,
                                                 flux_parabolic, _u_transformed,
                                                 _gradients, _flux_parabolic)
end

function init_parabolic_container_1d(n_vars::Integer, n_nodes::Integer,
                                     n_elements::Integer,
                                     ::Type{uEltype}) where {uEltype <: Real}
    return ParabolicContainer1D{uEltype}(n_vars, n_nodes, n_elements)
end

function Adapt.parent_type(::Type{<:ParabolicContainer1D{<:Any, <:Any,
                                                         VectoruEltype}}) where {VectoruEltype}
    return VectoruEltype
end

# Only one-dimensional `Array`s are `resize!`able in Julia.
# Hence, we use `Vector`s as internal storage and `resize!`
# them whenever needed. Then, we reuse the same memory by
# wrapping multi-dimensional arrays around the internal
# storage (see `wrap_storage`).
function Base.resize!(parabolic_container::ParabolicContainer1D, equations, dg, cache)
    @unpack _u_transformed, _gradients, _flux_parabolic = parabolic_container

    capacity = nvariables(equations) * nnodes(dg) * nelements(dg, cache)
    resize!(_u_transformed, capacity)
    resize!(_gradients, capacity)
    resize!(_flux_parabolic, capacity)

    array_size = (nvariables(equations), nnodes(dg), nelements(dg, cache))
    parabolic_container.u_transformed = wrap_storage(_u_transformed, array_size)
    parabolic_container.gradients = wrap_storage(_gradients, array_size)
    parabolic_container.flux_parabolic = wrap_storage(_flux_parabolic, array_size)

    return nothing
end

# Adapt the parabolic container to a different storage type (e.g., for GPUs)
# and/or real type. The (scratch) data of the multi-dimensional arrays is not
# preserved. Instead, they are re-created by wrapping the adapted internal storage,
# similar to the other containers.
function Adapt.adapt_structure(to, parabolic_container::ParabolicContainer1D)
    _u_transformed = adapt(to, parabolic_container._u_transformed)
    _gradients = adapt(to, parabolic_container._gradients)
    _flux_parabolic = adapt(to, parabolic_container._flux_parabolic)

    array_size = size(parabolic_container.u_transformed)
    u_transformed = wrap_storage(_u_transformed, array_size)
    gradients = wrap_storage(_gradients, array_size)
    flux_parabolic = wrap_storage(_flux_parabolic, array_size)

    return ParabolicContainer1D{eltype(_u_transformed), typeof(u_transformed),
                                typeof(_u_transformed)}(u_transformed, gradients,
                                                        flux_parabolic,
                                                        _u_transformed, _gradients,
                                                        _flux_parabolic)
end
