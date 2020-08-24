
# Base type from which all systems of equations types inherit from
abstract type AbstractEquation{NDIMS, NVARS} end


# Retrieve number of variables from equation type
@inline nvariables(::Type{AbstractEquation{NDIMS, NVARS}}) where {NDIMS, NVARS} = NVARS

# Retrieve number of variables from equation instance
@inline nvariables(::AbstractEquation{NDIMS, NVARS}) where {NDIMS, NVARS} = NVARS


# Add method to show some information on system of equations
function Base.show(io::IO, equation::AbstractEquation)
  print(io, "name = ", get_name(equation), ", n_vars = ", nvariables(equation))
end


@inline Base.ndims(::AbstractEquation{NDIMS}) where NDIMS = NDIMS


# Create an instance of a system of equation type based on a given name
function make_equations(name::String, ndims_)
  if name == "LinearScalarAdvectionEquation"
    if ndims_ == 2
      return LinearScalarAdvectionEquation2D()
    else
      return LinearScalarAdvectionEquation3D()
    end
  elseif name == "CompressibleEulerEquations"
    if ndims_ == 2
      return CompressibleEulerEquations2D()
    else
      error("CompressibleEulerEquations is not implemented for 3D")
    end
  elseif name == "IdealGlmMhdEquations"
    if ndims_ == 2
      return IdealGlmMhdEquations2D()
    else
      error("IdealGlmMhdEquations is not implemented for 3D")
    end
  elseif name == "HyperbolicDiffusionEquations"
    if ndims_ == 2
      return HyperbolicDiffusionEquations2D()
    else
      error("HyperbolicDiffusionEquations is not implemented for 3D")
    end
  else
    error("'$name' does not name a valid system of equations")
  end
end


have_nonconservative_terms(::AbstractEquation) = Val(false)
default_analysis_quantities(::AbstractEquation) = (:l2_error, :linf_error, :dsdu_ut)


"""
    flux_central(u_ll, u_rr, orientation, equation::AbstractEquation)

The classical central numerical flux `f((u_ll) + f(u_rr)) / 2`. When this flux is
used as volume flux, the discretization is equivalent to the classical weak form
DG method (except floating point errors).
"""
@inline function flux_central(u_ll, u_rr, orientation, equation::AbstractEquation)
  # Calculate regular 1D fluxes
  f_ll = calcflux(u_ll, orientation, equation)
  f_rr = calcflux(u_rr, orientation, equation)

  # Average regular fluxes
  return 0.5 * (f_ll + f_rr)
end


####################################################################################################
# Include files with actual implementations for different systems of equations.

# Linear scalar advection
abstract type AbstractLinearScalarAdvectionEquation{NDIMS, NVARS} <: AbstractEquation{NDIMS, NVARS} end
include("2d/linear_scalar_advection.jl")
include("3d/linear_scalar_advection.jl")

# CompressibleEulerEquations
include("2d/compressible_euler.jl")

# Ideal MHD
include("2d/ideal_glm_mhd.jl")

# Diffusion equation: first order hyperbolic system
include("2d/hyperbolic_diffusion.jl")
