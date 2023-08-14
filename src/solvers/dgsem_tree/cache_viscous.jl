mutable struct CacheViscous1D{uEltype <: Real}
    u_transformed::Array{uEltype, 3}
    gradients::Array{uEltype, 3}
    flux_viscous::Array{uEltype, 3}

    # internal `resize!`able storage
    _u_transformed::Vector{uEltype}
    _gradients::Vector{uEltype}
    _flux_viscous::Vector{uEltype}

    function CacheViscous1D{uEltype}(n_vars::Integer, n_nodes::Integer,
                                     n_elements::Integer) where {uEltype <: Real}
        new(Array{uEltype, 3}(undef, n_vars, n_nodes, n_elements),
            Array{uEltype, 3}(undef, n_vars, n_nodes, n_elements),
            Array{uEltype, 3}(undef, n_vars, n_nodes, n_elements),
            Vector{uEltype}(undef, n_vars * n_nodes * n_elements),
            Vector{uEltype}(undef, n_vars * n_nodes * n_elements),
            Vector{uEltype}(undef, n_vars * n_nodes * n_elements))
    end
end

# Only one-dimensional `Array`s are `resize!`able in Julia.
# Hence, we use `Vector`s as internal storage and `resize!`
# them whenever needed. Then, we reuse the same memory by
# `unsafe_wrap`ping multi-dimensional `Array`s around the
# internal storage.
function Base.resize!(cache_viscous::CacheViscous1D, capacity)
    resize!(cache_viscous._u_transformed, capacity)
    resize!(cache_viscous._gradients, capacity)
    resize!(cache_viscous._flux_viscous, capacity)

    return nothing
end
