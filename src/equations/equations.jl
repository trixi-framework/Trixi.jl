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
export central_flux


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
    return LinearScalarAdvectionEquation()
  elseif name == "euler"
    return CompressibleEulerEquations()
  elseif name == "mhd"
    return IdealMhdEquations()
  elseif name == "hyperbolicdiffusion"
    return HyperbolicDiffusionEquations()
  else
    error("'$name' does not name a valid system of equations")
  end
end


# Calculate 2D two-point flux (decide which volume flux type to use)
@inline function calcflux_twopoint!(f1, f2, f1_diag, f2_diag,
                                    equation, u, element_id, n_nodes)
  calcflux_twopoint!(f1, f2, f1_diag, f2_diag,
                     equation.volume_flux, equation, u, element_id, n_nodes)
end


####################################################################################################
# Include files with actual implementations for different systems of equations.

# Linear scalar advection
include("linearscalaradvection.jl")

# CompressibleEulerEquations
include("euler.jl")

# Ideal MHD
include("mhd.jl")

# Diffusion equation: first order hyperbolic system
include("hyperbolicdiffusion.jl")

end # module
