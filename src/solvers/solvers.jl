
# Base type from which all solvers inherit from
abstract type AbstractSolver end


# Create an instance of a solver based on a given name
function make_solver(name::String, equations::AbstractEquation, mesh::TreeMesh;
                     surface_flux_function=nothing, volume_flux_function=nothing)
  if name == "dg"
    N = parameter("N")

    # "eval is evil"
    # This is a temporary hack until we have switched to a library based approach
    # with pure Julia code instead of parameter files.
    if isnothing(surface_flux_function)
      surface_flux_type = Symbol(parameter("surface_flux", "flux_lax_friedrichs"))
      surface_flux_function = eval(surface_flux_type)
    end
    if isnothing(volume_flux_function)
      volume_flux_type = Symbol(parameter("volume_flux", "flux_central"))
      volume_flux_function = eval(volume_flux_type)
    end

    initial_conditions_type = Symbol(parameter("initial_conditions"))
    initial_conditions = eval(initial_conditions_type)

    source_terms_type = Symbol(parameter("source_terms", "nothing"))
    source_terms = eval(source_terms_type)

    return Dg(equations, surface_flux_function, volume_flux_function, initial_conditions, source_terms, mesh, N)
  else
    error("'$name' does not name a valid solver")
  end
end


####################################################################################################
# Include files with actual implementations for different systems of equations.

# DG
include("dg.jl")
