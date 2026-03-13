export get_u, get_coordinates

@inline function _build_svector_array(raw, ::Val{N}) where N
    components = ntuple(i -> selectdim(raw, 1, i), N)
    return StructArray{SVector{N, eltype(raw)}}(components)
end

function get_u(sol)
    semi  = sol.prob.p
    u_ode = sol.u[end]
    return get_u(u_ode, semi)
end

function get_u(u_ode, semi::AbstractSemidiscretization)
    u_raw  = wrap_array(u_ode, semi)
    n_vars = nvariables(semi)
    @assert ndims(u_raw) >= 3 "Unexpected wrap_array shape: $(size(u_raw))"
    return _build_svector_array(u_raw, Val(n_vars))
end

function get_coordinates(sol)
    return get_coordinates(sol.prob.p)
end

function get_coordinates(semi::AbstractSemidiscretization)
    x_raw  = semi.cache.elements.node_coordinates
    n_dims = ndims(semi)
    return _build_svector_array(x_raw, Val(n_dims))
end