
# Retrieve number of variables from equation instance
@inline nvariables(::AbstractEquations{NDIMS, NVARS}) where {NDIMS, NVARS} = NVARS

# TODO: Taal performance, 1:NVARS vs. Base.OneTo(NVARS) vs. SOneTo(NVARS)
@inline eachvariable(equations::AbstractEquations) = Base.OneTo(nvariables(equations))


# Add method to show some information on system of equations
function Base.show(io::IO, equations::AbstractEquations)
  print(io, get_name(equations), " with ")
  if nvariables(equations) == 1
    print(io, "one variable")
  else
    print(io, nvariables(equations), " variables")
  end
end


@inline Base.ndims(::AbstractEquations{NDIMS}) where NDIMS = NDIMS


"""
    calcflux(u, orientation, equations)

Given the conservative variables `u`, calculate the (physical) flux in spatial
direction `orientation` for the coressponding set of governing `equations`
`orientation` is `1`, `2`, and `3` for the x-, y, and z-directions, respectively.
"""
calcflux(u, orientation, equations)


# TODO: Taal remove method below
# Create an instance of a system of equation type based on a given name
function make_equations(name::String, ndims_)
  if name == "LinearScalarAdvectionEquation"
    if ndims_ == 1
      return LinearScalarAdvectionEquation1D()
    elseif ndims_ == 2
      return LinearScalarAdvectionEquation2D()
    elseif ndims_ == 3
      return LinearScalarAdvectionEquation3D()
    else
      error("Unsupported number of spatial dimensions: ", ndims_)
    end
  elseif name == "CompressibleEulerEquations"
    if ndims_ == 1
      return CompressibleEulerEquations1D()
    elseif ndims_ == 2
      return CompressibleEulerEquations2D()
    elseif ndims_ == 3
      return CompressibleEulerEquations3D()
    else
      error("Unsupported number of spatial dimensions: ", ndims_)
    end
  elseif name == "IdealGlmMhdEquations"
    if ndims_ == 2
      return IdealGlmMhdEquations2D()
    elseif ndims_ == 3
      return IdealGlmMhdEquations3D()
    else
      error("Unsupported number of spatial dimensions: ", ndims_)
    end
  elseif name == "HyperbolicDiffusionEquations"
    if ndims_ == 2
      return HyperbolicDiffusionEquations2D()
    elseif ndims_ == 3
      return HyperbolicDiffusionEquations3D()
    else
      error("Unsupported number of spatial dimensions: ", ndims_)
    end
  else
    error("'$name' does not name a valid system of equations")
  end
end


have_nonconservative_terms(::AbstractEquations) = Val(false)
have_constant_speed(::AbstractEquations) = Val(false)

# TODO: Taal refactor, remove default_analysis_quantities
default_analysis_quantities(::AbstractEquations) = (:l2_error, :linf_error, :dsdu_ut)
default_analysis_errors(::AbstractEquations)     = (:l2_error, :linf_error)
default_analysis_integrals(::AbstractEquations)  = (entropy_timederivative,)


"""
    flux_central(u_ll, u_rr, orientation, equations::AbstractEquations)

The classical central numerical flux `f((u_ll) + f(u_rr)) / 2`. When this flux is
used as volume flux, the discretization is equivalent to the classical weak form
DG method (except floating point errors).
"""
@inline function flux_central(u_ll, u_rr, orientation, equations::AbstractEquations)
  # Calculate regular 1D fluxes
  f_ll = calcflux(u_ll, orientation, equations)
  f_rr = calcflux(u_rr, orientation, equations)

  # Average regular fluxes
  return 0.5 * (f_ll + f_rr)
end


@inline cons2cons(u, ::AbstractEquations) = u
@inline Base.first(u, ::AbstractEquations) = first(u)


####################################################################################################
# Include files with actual implementations for different systems of equations.

# Linear scalar advection
abstract type AbstractLinearScalarAdvectionEquation{NDIMS, NVARS} <: AbstractEquations{NDIMS, NVARS} end
include("1d/linear_scalar_advection.jl")
include("2d/linear_scalar_advection.jl")
include("3d/linear_scalar_advection.jl")

# CompressibleEulerEquations
abstract type AbstractCompressibleEulerEquations{NDIMS, NVARS} <: AbstractEquations{NDIMS, NVARS} end
include("1d/compressible_euler.jl")
include("2d/compressible_euler.jl")
include("3d/compressible_euler.jl")

# Ideal MHD
abstract type AbstractIdealGlmMhdEquations{NDIMS, NVARS} <: AbstractEquations{NDIMS, NVARS} end
include("2d/ideal_glm_mhd.jl")
include("3d/ideal_glm_mhd.jl")

# Diffusion equation: first order hyperbolic system
abstract type AbstractHyperbolicDiffusionEquations{NDIMS, NVARS} <: AbstractEquations{NDIMS, NVARS} end
include("2d/hyperbolic_diffusion.jl")
include("3d/hyperbolic_diffusion.jl")
