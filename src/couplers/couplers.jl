module Couplers

using ..Trixi.Mesh: TreeMesh
using ..Trixi.Solvers: AbstractSolver
using ..Trixi.Auxiliary: parameter

export AbstractCoupler
export make_coupler
export couple_post_rhs!

# Base type from which all couplers inherit from
abstract type AbstractCoupler end


# Create an instance of a coupler based on a given name
function make_coupler(name::String, solver_a::AbstractSolver, solver_b::AbstractSolver,
                      mesh::TreeMesh)
  if name == "dg_source"
    return DgSource(solver_a, solver_b, mesh)
  else
    error("'$name' does not name a valid coupler")
  end
end


####################################################################################################
# Include files with actual implementations for different systems of couplers.

# First, add generic functions for which the submodules can create own methods
function couple_post_rhs! end

# Next, include module files and make symbols available. Here we employ an
# unqualified "using" to avoid boilerplate code.

# DG source coupler
include("dg_source.jl")
using .DgSourceCoupler

end
