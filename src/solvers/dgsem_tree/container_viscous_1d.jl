mutable struct ViscousContainer1D{uEltype <: Real}
    u_transformed::Array{uEltype, 3}
    gradients::Array{uEltype, 3}
    flux_viscous::Array{uEltype, 3}

    # internal `resize!`able storage
    _u_transformed::Vector{uEltype}
    _gradients::Vector{uEltype}
    _flux_viscous::Vector{uEltype}

    function ViscousContainer1D{uEltype}(n_vars::Integer, n_nodes::Integer,
                                         n_elements::Integer) where {uEltype <: Real}
        new(Array{uEltype, 3}(undef, n_vars, n_nodes, n_elements),
            Array{uEltype, 3}(undef, n_vars, n_nodes, n_elements),
            Array{uEltype, 3}(undef, n_vars, n_nodes, n_elements),
            Vector{uEltype}(undef, n_vars * n_nodes * n_elements),
            Vector{uEltype}(undef, n_vars * n_nodes * n_elements),
            Vector{uEltype}(undef, n_vars * n_nodes * n_elements))
    end
end

function init_viscous_container_1d(n_vars::Integer, n_nodes::Integer,
                                   n_elements::Integer,
                                   ::Type{uEltype}) where {uEltype <: Real}
    return ViscousContainer1D{uEltype}(n_vars, n_nodes, n_elements)
end

# Only one-dimensional `Array`s are `resize!`able in Julia.
# Hence, we use `Vector`s as internal storage and `resize!`
# them whenever needed. Then, we reuse the same memory by
# `unsafe_wrap`ping multi-dimensional `Array`s around the
# internal storage.
function Base.resize!(viscous_container::ViscousContainer1D, equations, dg, cache)
    capacity = nvariables(equations) * nnodes(dg) * nelements(dg, cache)
    resize!(viscous_container._u_transformed, capacity)
    resize!(viscous_container._gradients, capacity)
    resize!(viscous_container._flux_viscous, capacity)

    viscous_container.u_transformed = unsafe_wrap(Array,
                                                  pointer(viscous_container._u_transformed),
                                                  (nvariables(equations),
                                                   nnodes(dg),
                                                   nelements(dg, cache)))

    viscous_container.gradients = unsafe_wrap(Array,
                                              pointer(viscous_container._gradients),
                                              (nvariables(equations),
                                               nnodes(dg),
                                               nelements(dg, cache)))

    viscous_container.flux_viscous = unsafe_wrap(Array,
                                                 pointer(viscous_container._flux_viscous),
                                                 (nvariables(equations),
                                                  nnodes(dg),
                                                  nelements(dg, cache)))

    return nothing
end
