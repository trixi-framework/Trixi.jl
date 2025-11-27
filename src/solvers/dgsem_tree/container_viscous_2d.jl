mutable struct ViscousContainer2D{uEltype <: Real}
    u_transformed::Array{uEltype, 4}
    gradients::NTuple{2, Array{uEltype, 4}}
    flux_viscous::NTuple{2, Array{uEltype, 4}}

    # internal `resize!`able storage
    _u_transformed::Vector{uEltype}
    # Use Tuple for outer, fixed-size datastructure
    _gradients::Tuple{Vector{uEltype}, Vector{uEltype}}
    _flux_viscous::Tuple{Vector{uEltype}, Vector{uEltype}}

    function ViscousContainer2D{uEltype}(n_vars::Integer, n_nodes::Integer,
                                         n_elements::Integer) where {uEltype <: Real}
        return new(Array{uEltype, 4}(undef, n_vars, n_nodes, n_nodes, n_elements), # `u_transformed`
                   # `gradients`
                   (Array{uEltype, 4}(undef, n_vars, n_nodes, n_nodes, n_elements),
                    Array{uEltype, 4}(undef, n_vars, n_nodes, n_nodes, n_elements)),
                   # `flux_viscous`
                   (Array{uEltype, 4}(undef, n_vars, n_nodes, n_nodes, n_elements),
                    Array{uEltype, 4}(undef, n_vars, n_nodes, n_nodes, n_elements)),
                   # `_u_transformed`
                   Vector{uEltype}(undef, n_vars * n_nodes^2 * n_elements),
                   # `_gradients`
                   (Vector{uEltype}(undef, n_vars * n_nodes^2 * n_elements),
                    Vector{uEltype}(undef, n_vars * n_nodes^2 * n_elements)),
                   # `_flux_viscous`
                   (Vector{uEltype}(undef, n_vars * n_nodes^2 * n_elements),
                    Vector{uEltype}(undef, n_vars * n_nodes^2 * n_elements)))
    end
end

function init_viscous_container_2d(n_vars::Integer, n_nodes::Integer,
                                   n_elements::Integer,
                                   ::Type{uEltype}) where {uEltype <: Real}
    return ViscousContainer2D{uEltype}(n_vars, n_nodes, n_elements)
end

# Only one-dimensional `Array`s are `resize!`able in Julia.
# Hence, we use `Vector`s as internal storage and `resize!`
# them whenever needed. Then, we reuse the same memory by
# `unsafe_wrap`ping multi-dimensional `Array`s around the
# internal storage.
function Base.resize!(viscous_container::ViscousContainer2D, equations, dg, cache)
    capacity = nvariables(equations) * nnodes(dg) * nnodes(dg) *
               nelements(dg, cache)
    resize!(viscous_container._u_transformed, capacity)
    for dim in 1:2
        resize!(viscous_container._gradients[dim], capacity)
        resize!(viscous_container._flux_viscous[dim], capacity)
    end

    viscous_container.u_transformed = unsafe_wrap(Array,
                                                  pointer(viscous_container._u_transformed),
                                                  (nvariables(equations),
                                                   nnodes(dg), nnodes(dg),
                                                   nelements(dg, cache)))

    gradients_1 = unsafe_wrap(Array,
                              pointer(viscous_container._gradients[1]),
                              (nvariables(equations),
                               nnodes(dg), nnodes(dg),
                               nelements(dg, cache)))
    gradients_2 = unsafe_wrap(Array,
                              pointer(viscous_container._gradients[2]),
                              (nvariables(equations),
                               nnodes(dg), nnodes(dg),
                               nelements(dg, cache)))

    viscous_container.gradients = (gradients_1, gradients_2)

    flux_viscous_1 = unsafe_wrap(Array,
                                 pointer(viscous_container._flux_viscous[1]),
                                 (nvariables(equations),
                                  nnodes(dg), nnodes(dg),
                                  nelements(dg, cache)))
    flux_viscous_2 = unsafe_wrap(Array,
                                 pointer(viscous_container._flux_viscous[2]),
                                 (nvariables(equations),
                                  nnodes(dg), nnodes(dg),
                                  nelements(dg, cache)))

    viscous_container.flux_viscous = (flux_viscous_1, flux_viscous_2)

    return nothing
end
