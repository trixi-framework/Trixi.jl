module Equations

using ..Jul1dge

export make_equations
export nvars
export AbstractEquation
export initial_conditions
export sources
export calcflux
export riemann!
export maxdt
export cons2prim


# Base type from which all systems of equations types inherit from
abstract type AbstractEquation{V} end


# Retrieve number of variables from equation type
nvars(::Type{AbstractEquation{V}}) where V = V

# Retrieve number of variables from equation instance
nvars(::AbstractEquation{V}) where V = V


# Retrieve name of system of equations
name(s::AbstractEquation) = s.name


# Add method to show some information on system of equations
function Base.show(io::IO, s::AbstractEquation)
  print("name = $(s.name), nvars = $(nvars(s))")
end


# Create an instance of a system of equation type based on a given name
function make_equations(name::String)
  if name == "linearscalaradvection"
    return LinearScalarAdvection()
  elseif name == "euler"
    return Euler()
  else
    error("'$name' does not name a valid system of equations")
  end
end


####################################################################################################
# Include files with actual implementations for different systems of equations.

# First, add generic functions for which the submodules can create own methods
function initial_conditions end
function sources end
function calcflux end
function riemann! end
function maxdt end
function cons2prim end

# Next, include module files and make symbols available. Here we employ an
# unqualified "using" to avoid boilerplate code.

# Linear scalar advection
include("linearscalaradvection.jl")
using .LinearScalarAdvectionEquations

# Euler
include("euler.jl")
using .EulerEquations

end # module
