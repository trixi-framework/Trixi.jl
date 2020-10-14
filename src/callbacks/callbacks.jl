
# overload this function for specific callbacks which use element element variables
# that should be saved
get_element_variables!(element_variables, u, mesh, equations, solver, cache, callback) = nothing

@inline function get_element_variables!(element_variables, u_ode::AbstractVector,
                                        semi::AbstractSemidiscretization, cb::DiscreteCallback)
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
  u = wrap_array(u_ode, mesh, equations, solver, cache)
  get_element_variables!(element_variables, u, mesh, equations, solver, cache, cb.affect!)
end


# `include` callback definitions in the order that we currently prefer
# when combining them into a `CallbackSet` which is called after a complete step
include("summary.jl")
include("steady_state.jl")
include("amr.jl")
include("stepsize.jl")
include("save_solution.jl")
include("analysis.jl")
include("alive.jl")
