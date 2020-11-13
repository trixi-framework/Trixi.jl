
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
`orientation` is `1`, `2`, and `3` for the x-, y-, and z-directions, respectively.
"""
function calcflux(u, orientation, equations) end


# set sensible default values that may be overwritten by specific equations
have_nonconservative_terms(::AbstractEquations) = Val(false)
have_constant_speed(::AbstractEquations) = Val(false)

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
