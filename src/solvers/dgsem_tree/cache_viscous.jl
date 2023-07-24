mutable struct CacheViscous1D{uEltype <: Real}
  u_transformed::Array{uEltype}
  gradients::Array{uEltype}
  flux_viscous::Array{uEltype}

  # internal `resize!`able storage
  _u_transformed::Vector{uEltype}
  _gradients::Vector{uEltype}
  _flux_viscous::Vector{uEltype}

  function CacheViscous1D{uEltype}(n_vars::Integer, n_nodes::Integer, n_elements::Integer) where {uEltype <: Real}
      new(Array{uEltype}(undef, n_vars, n_nodes, n_elements),
          Array{uEltype}(undef, n_vars, n_nodes, n_elements),
          Array{uEltype}(undef, n_vars, n_nodes, n_elements),
          Vector{uEltype}(undef, n_vars*n_nodes*n_elements),
          Vector{uEltype}(undef, n_vars*n_nodes*n_elements),
          Vector{uEltype}(undef, n_vars*n_nodes*n_elements))
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

mutable struct CacheViscous2D{uEltype <: Real}
    u_transformed::Array{uEltype}
    gradients::Vector{Array{uEltype}}
    flux_viscous::Vector{Array{uEltype}}

    # internal `resize!`able storage
    _u_transformed::Vector{uEltype}
    _gradients::Vector{Vector{uEltype}}
    _flux_viscous::Vector{Vector{uEltype}}

    function CacheViscous2D{uEltype}(n_vars::Integer, n_nodes::Integer, n_elements::Integer) where {uEltype <: Real}
        new(Array{uEltype}(undef, n_vars, n_nodes, n_nodes, n_elements),
            [Array{uEltype}(undef, n_vars, n_nodes, n_nodes, n_elements) for _ in 1:2],
            [Array{uEltype}(undef, n_vars, n_nodes, n_nodes, n_elements) for _ in 1:2],
            Vector{uEltype}(undef, n_vars * n_nodes * n_nodes * n_elements),
            [Vector{uEltype}(undef, n_vars * n_nodes * n_nodes * n_elements) for _ in 1:2],
            [Vector{uEltype}(undef, n_vars * n_nodes * n_nodes * n_elements) for _ in 1:2])
    end
end

# Only one-dimensional `Array`s are `resize!`able in Julia.
# Hence, we use `Vector`s as internal storage and `resize!`
# them whenever needed. Then, we reuse the same memory by
# `unsafe_wrap`ping multi-dimensional `Array`s around the
# internal storage.
function Base.resize!(cache_viscous::CacheViscous2D, capacity)
    resize!(cache_viscous._u_transformed, capacity)
    for dim in 1:2
      resize!(cache_viscous._gradients[dim], capacity)
      resize!(cache_viscous._flux_viscous[dim], capacity)
    end

    return nothing
end