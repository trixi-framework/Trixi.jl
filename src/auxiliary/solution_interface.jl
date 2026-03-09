export get_u, get_coordinates

"""
    get_u(sol)
    get_u(u_ode, semi)

Extracts the nodal solution values from an `ODESolution` (or raw ODE vector and semidiscretization)
and returns them as a `StructArray` of `SVector`s. The output shape is purely spatial 
(e.g., `(n_nodes_x, n_nodes_y, n_elements)` in 2D), where each element contains the physics variables.
"""
function get_u(sol)
    semi = sol.prob.p
    u_ode = sol.u[end]
    return get_u(u_ode, semi)
end

function get_u(u_ode, semi::SemidiscretizationHyperbolic)
    u_raw = wrap_array(u_ode, semi)
    n_vars = size(u_raw, 1)
    
    component_arrays = ntuple(v -> view(u_raw, v, ntuple(_ -> Colon(), ndims(u_raw)-1)...), n_vars)
    
    return StructArray{SVector{n_vars, eltype(u_raw)}}(component_arrays)
end

"""
    get_coordinates(sol)
    get_coordinates(semi)

Extracts the nodal physical coordinates from an `ODESolution` or semidiscretization and returns 
them as a `StructArray` of `SVector`s representing the spatial dimensions.
"""
function get_coordinates(sol)
    semi = sol.prob.p
    return get_coordinates(semi)
end

function get_coordinates(semi::SemidiscretizationHyperbolic)
    x_raw = semi.cache.elements.node_coordinates
    n_dims = size(x_raw, 1)
    
    component_arrays = ntuple(d -> view(x_raw, d, ntuple(_ -> Colon(), ndims(x_raw)-1)...), n_dims)
    
    return StructArray{SVector{n_dims, eltype(x_raw)}}(component_arrays)
end