"""
    get_u(sol::ODESolution)
    get_u(u_ode, semi::AbstractSemidiscretization)

Extract the state variable `u` from an `ODESolution` or a raw `u_ode` array
and wrap it into a multidimensional array based on the underlying semidiscretization.
This function uses multiple dispatch to handle different mesh types
(e.g., `TreeMesh`, `StructuredMesh`, `DGMultiMesh`).
"""
function get_u(sol::ODESolution)
    semi = sol.prob.p
    u_ode = sol.u[end]
    return get_u(u_ode, semi)
end

function get_u(u_ode, semi::AbstractSemidiscretization)
    return wrap_array(u_ode, semi)
end

"""
    get_coordinates(sol::ODESolution)
    get_coordinates(semi::AbstractSemidiscretization)

Extract the nodal coordinates from a semidiscretization or an `ODESolution`.
Returns a multidimensional array containing the coordinates for each node.
This function uses multiple dispatch to handle different mesh types 
(e.g., `TreeMesh`, `StructuredMesh`, `DGMultiMesh`).
"""
function get_coordinates(sol::ODESolution)
    return get_coordinates(sol.prob.p)
end

function get_coordinates(semi::AbstractSemidiscretization)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    return get_coordinates(mesh, equations, solver, cache)
end

function get_coordinates(mesh, equations, solver, cache)
    return cache.elements.node_coordinates
end

function get_coordinates(mesh::DGMultiMesh, equations, solver, cache)
    return mesh.md.xyz
end
