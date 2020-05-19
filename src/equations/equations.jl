module Equations

using ..Trixi

export make_equations
export nvariables
export AbstractEquation
export initial_conditions
export sources
export calcflux!
export calcflux_twopoint!
export riemann!
export noncons_surface_flux!
export calc_max_dt
export cons2prim
export cons2indicator
export cons2indicator!


# Base type from which all systems of equations types inherit from
abstract type AbstractEquation{V} end


# Retrieve number of variables from equation type
nvariables(::Type{AbstractEquation{V}}) where V = V

# Retrieve number of variables from equation instance
nvariables(::AbstractEquation{V}) where V = V


# Retrieve name of system of equations
name(equation::AbstractEquation) = equation.name


# Add method to show some information on system of equations
function Base.show(io::IO, equation::AbstractEquation)
  print(io, "name = $(equation.name), n_vars = $(nvariables(equation))")
end


# Create an instance of a system of equation type based on a given name
function make_equations(name::String)
  if name == "linearscalaradvection"
    return LinearScalarAdvection()
  elseif name == "euler"
    return Euler()
  elseif name == "mhd"
    return Mhd()
  elseif name == "hyperbolicdiffusion"
    return HyperbolicDiffusion()
  else
    error("'$name' does not name a valid system of equations")
  end
end


####################################################################################################
# Include files with actual implementations for different systems of equations.

# First, add generic functions for which the submodules can create own methods
function initial_conditions end
function sources end
function calcflux! end
function calcflux_twopoint! end
function riemann! end
function noncons_surface_flux! end
function calc_max_dt end
function cons2prim end
function cons2indicator end
function cons2indicator! end
function cons2entropy end

# Next, include module files and make symbols available. Here we employ an
# unqualified "using" to avoid boilerplate code.

# Linear scalar advection
include("linearscalaradvection.jl")
using .LinearScalarAdvectionEquations

# Euler
include("euler.jl")
using .EulerEquations

# Ideal MHD
include("mhd.jl")
using .MhdEquations

# Diffusion equation: first order hyperbolic system
include("hyperbolicdiffusion.jl")
using .HyperbolicDiffusionEquations

end # module
