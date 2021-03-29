
# Retrieve number of variables from equation instance
@inline nvariables(::AbstractEquations{NDIMS, NVARS}) where {NDIMS, NVARS} = NVARS

# TODO: Taal performance, 1:NVARS vs. Base.OneTo(NVARS) vs. SOneTo(NVARS)
@inline eachvariable(equations::AbstractEquations) = Base.OneTo(nvariables(equations))

"""
    get_name(equations::AbstractEquations)

Returns the canonical, human-readable name for the given system of equations.

# Examples
```jldoctest
julia> Trixi.get_name(CompressibleEulerEquations1D(1.4))
"CompressibleEulerEquations1D"
```
"""
get_name(equations::AbstractEquations) = equations |> typeof |> nameof |> string


# Add methods to show some information on systems of equations.
function Base.show(io::IO, equations::AbstractEquations)
  # Since this is not performance-critical, we can use `@nospecialize` to reduce latency.
  @nospecialize equations # reduce precompilation time

  print(io, get_name(equations), " with ")
  if nvariables(equations) == 1
    print(io, "one variable")
  else
    print(io, nvariables(equations), " variables")
  end
end

function Base.show(io::IO, ::MIME"text/plain", equations::AbstractEquations)
  # Since this is not performance-critical, we can use `@nospecialize` to reduce latency.
  @nospecialize equations # reduce precompilation time

  if get(io, :compact, false)
    show(io, equations)
  else
    summary_header(io, get_name(equations))
    summary_line(io, "#variables", nvariables(equations))
    for variable in eachvariable(equations)
      summary_line(increment_indent(io),
                   "variable " * string(variable),
                   varnames(cons2cons, equations)[variable])
    end
    summary_footer(io)
  end
end


@inline Base.ndims(::AbstractEquations{NDIMS}) where NDIMS = NDIMS


"""
    flux(u, orientation, equations)

Given the conservative variables `u`, calculate the (physical) flux in spatial
direction `orientation` for the coressponding set of governing `equations`
`orientation` is `1`, `2`, and `3` for the x-, y-, and z-directions, respectively.
"""
function flux end


# set sensible default values that may be overwritten by specific equations
have_nonconservative_terms(::AbstractEquations) = Val(false)
have_constant_speed(::AbstractEquations) = Val(false)

default_analysis_errors(::AbstractEquations)     = (:l2_error, :linf_error)
default_analysis_integrals(::AbstractEquations)  = (entropy_timederivative,)


@inline cons2cons(u, ::AbstractEquations) = u
function cons2prim#=(u, ::AbstractEquations)=# end
@inline Base.first(u, ::AbstractEquations) = first(u)

# FIXME: Deprecations introduced in v0.3
@deprecate varnames_cons(equations) varnames(cons2cons, equations)
@deprecate varnames_prim(equations) varnames(cons2prim, equations)
@deprecate flux_upwind(u_ll, u_rr, orientation, equations) flux_godunov(u_ll, u_rr, orientation, equations)
@deprecate calcflux(u, orientation, equations) flux(u, orientation, equations)


####################################################################################################
# Include files with actual implementations for different systems of equations.

# Numerical flux formulations that are independent of the specific system of equations
include("numerical_fluxes.jl")

# Linear scalar advection
abstract type AbstractLinearScalarAdvectionEquation{NDIMS, NVARS} <: AbstractEquations{NDIMS, NVARS} end
include("linear_scalar_advection_1d.jl")
include("linear_scalar_advection_2d.jl")
include("linear_scalar_advection_3d.jl")

# Inviscid Burgers
abstract type AbstractInviscidBurgersEquation{NDIMS, NVARS} <: AbstractEquations{NDIMS, NVARS} end
include("inviscid_burgers_1d.jl")

# CompressibleEulerEquations
abstract type AbstractCompressibleEulerEquations{NDIMS, NVARS} <: AbstractEquations{NDIMS, NVARS} end
include("compressible_euler_1d.jl")
include("compressible_euler_2d.jl")
include("compressible_euler_3d.jl")

# CompressibleEulerMulticomponentEquations
abstract type AbstractCompressibleEulerMulticomponentEquations{NDIMS, NVARS, NCOMP} <: AbstractEquations{NDIMS, NVARS} end
include("compressible_euler_multicomponent_1d.jl")
include("compressible_euler_multicomponent_2d.jl")

# Retrieve number of components from equation instance for the multicomponent case
@inline ncomponents(::AbstractCompressibleEulerMulticomponentEquations{NDIMS, NVARS, NCOMP}) where {NDIMS, NVARS, NCOMP} = NCOMP
@inline eachcomponent(equations::AbstractCompressibleEulerMulticomponentEquations) = Base.OneTo(ncomponents(equations))

# Ideal MHD
abstract type AbstractIdealGlmMhdEquations{NDIMS, NVARS} <: AbstractEquations{NDIMS, NVARS} end
include("ideal_glm_mhd_1d.jl")
include("ideal_glm_mhd_2d.jl")
include("ideal_glm_mhd_3d.jl")

# IdealGlmMhdMulticomponentEquations
abstract type AbstractIdealGlmMhdMulticomponentEquations{NDIMS, NVARS, NCOMP} <: AbstractEquations{NDIMS, NVARS} end
include("ideal_glm_mhd_multicomponent_1d.jl")
include("ideal_glm_mhd_multicomponent_2d.jl")

# Retrieve number of components from equation instance for the multicomponent case
@inline ncomponents(::AbstractIdealGlmMhdMulticomponentEquations{NDIMS, NVARS, NCOMP}) where {NDIMS, NVARS, NCOMP} = NCOMP
@inline eachcomponent(equations::AbstractIdealGlmMhdMulticomponentEquations) = Base.OneTo(ncomponents(equations))

# Diffusion equation: first order hyperbolic system
abstract type AbstractHyperbolicDiffusionEquations{NDIMS, NVARS} <: AbstractEquations{NDIMS, NVARS} end
include("hyperbolic_diffusion_1d.jl")
include("hyperbolic_diffusion_2d.jl")
include("hyperbolic_diffusion_3d.jl")

# Lattice-Boltzmann equation (advection part only)
abstract type AbstractLatticeBoltzmannEquations{NDIMS, NVARS} <: AbstractEquations{NDIMS, NVARS} end
include("lattice_boltzmann_2d.jl")
include("lattice_boltzmann_3d.jl")

# Acoustic perturbation equations
abstract type AbstractAcousticPerturbationEquations{NDIMS, NVARS} <: AbstractEquations{NDIMS, NVARS} end
include("acoustic_perturbation_2d.jl")
