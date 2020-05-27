
# Base type from which all solvers inherit from
abstract type AbstractSolver end


# Create an instance of a solver based on a given name
function make_solver(name::String, equations::AbstractEquation, mesh::TreeMesh)
  if name == "dg"
    N = parameter("N")
    return Dg(equations, mesh, N)
  else
    error("'$name' does not name a valid system of equations")
  end
end


####################################################################################################
# Include files with actual implementations for different systems of equations.

# DG
include("dg.jl")
