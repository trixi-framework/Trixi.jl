module Solvers

using ..Jul1dge.Mesh.Trees: Tree
using ..Jul1dge.Equations: AbstractEquation
using ..Jul1dge.Auxiliary: parameter

export AbstractSolver
export make_solver
export setinitialconditions
export analyze_solution
export calcdt
export equations
export rhs!

# Base type from which all solvers inherit from
abstract type AbstractSolver end


# Create an instance of a solver based on a given name
function make_solver(name::String, equations::AbstractEquation, mesh::Tree)
  if name == "dg"
    N = parameter("N")
    return Dg(equations, mesh, N)
  else
    error("'$name' does not name a valid system of equations")
  end
end


####################################################################################################
# Include files with actual implementations for different systems of equations.

# First, add generic functions for which the submodules can create own methods
function setinitialconditions end
function analyze_solution end
function calcdt end
function equations end
function rhs! end

# Next, include module files and make symbols available. Here we employ an
# unqualified "using" to avoid boilerplate code.

# DG
include("dg.jl")
using .DgSolver

end
