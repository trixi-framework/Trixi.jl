
# Base type from which all systems of equations types inherit from
abstract type AbstractEquation{V} end


# Retrieve number of variables from equation type
nvariables(::Type{AbstractEquation{V}}) where V = V

# Retrieve number of variables from equation instance
nvariables(::AbstractEquation{V}) where V = V


# Add method to show some information on system of equations
function Base.show(io::IO, equation::AbstractEquation)
  print(io, "name = ", get_name(equation), ", n_vars = ", nvariables(equation))
end


# Create an instance of a system of equation type based on a given name
function make_equations(name::String)
  if name == "LinearScalarAdvection"
    return LinearScalarAdvectionEquation()
  elseif name == "CompressibleEuler"
    return CompressibleEulerEquations()
  elseif name == "IdealMhd"
    return IdealMhdEquations()
  elseif name == "HyperbolicDiffusion"
    return HyperbolicDiffusionEquations()
  else
    error("'$name' does not name a valid system of equations")
  end
end


have_nonconservative_terms(::AbstractEquation) = Val(false)


####################################################################################################
# Include files with actual implementations for different systems of equations.

# Linear scalar advection
include("linear_scalar_advection.jl")

# CompressibleEulerEquations
include("compressible_euler.jl")

# Ideal MHD
include("ideal_mhd.jl")

# Diffusion equation: first order hyperbolic system
include("hyperbolic_diffusion.jl")
