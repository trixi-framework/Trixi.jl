# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


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

"""
    varnames(conversion_function, equations)

Return the list of variable names when applying `conversion_function` to the
conserved variables associated to `equations`. This function is mainly used
internally to determine output to screen and for IO, e.g., for the
[`AnalysisCallback`](@ref) and the [`SaveSolutionCallback`](@ref).
Common choices of the `conversion_function` are [`cons2cons`](@ref) and
[`cons2prim`](@ref).
"""
function varnames end


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

# equations act like scalars in broadcasting
Base.broadcastable(equations::AbstractEquations) = Ref(equations)


"""
    flux(u, orientation_or_normal, equations)

Given the conservative variables `u`, calculate the (physical) flux in Cartesian
direction `orientation::Integer` or in arbitrary direction `normal::AbstractVector`
for the corresponding set of governing `equations`.
`orientation` is `1`, `2`, and `3` for the x-, y-, and z-directions, respectively.
"""
function flux end

"""
    flux(u, normal_direction::AbstractVector, equations::AbstractEquations{1})

Enables calling `flux` with a non-integer argument `normal_direction` for one-dimensional
equations. Returns the value of `flux(u, 1, equations)` scaled by `normal_direction[1]`.
"""
@inline function flux(u, normal_direction::AbstractVector, equations::AbstractEquations{1})
  # Call `flux` with `orientation::Int = 1` for dispatch. Note that the actual
  # `orientation` argument is ignored.
  return normal_direction[1] * flux(u, 1, equations)
end


"""
    rotate_to_x(u, normal, equations)

Apply the rotation that maps `normal` onto the x-axis to the convservative variables `u`.
This is used by [`FluxRotated`](@ref) to calculate the numerical flux of rotationally
invariant equations in arbitrary normal directions.

See also: [`rotate_from_x`](@ref)
"""
function rotate_to_x end

"""
    rotate_from_x(u, normal, equations)

Apply the rotation that maps the x-axis onto `normal` to the convservative variables `u`.
This is used by [`FluxRotated`](@ref) to calculate the numerical flux of rotationally
invariant equations in arbitrary normal directions.

See also: [`rotate_to_x`](@ref)
"""
function rotate_from_x end


"""
    BoundaryConditionDirichlet(boundary_value_function)

Create a Dirichlet boundary condition that uses the function `boundary_value_function`
to specify the values at the boundary.
This can be used to create a boundary condition that specifies exact boundary values
by passing the exact solution of the equation.
The passed boundary value function will be called with the same arguments as an initial condition function is called, i.e., as
```julia
boundary_value_function(x, t, equations)
```
where `x` specifies the coordinates, `t` is the current time, and `equation` is the corresponding system of equations.

# Examples
```julia
julia> BoundaryConditionDirichlet(initial_condition_convergence_test)
```
"""
struct BoundaryConditionDirichlet{B}
  boundary_value_function::B
end

# Dirichlet-type boundary condition for use with TreeMesh or StructuredMesh
@inline function (boundary_condition::BoundaryConditionDirichlet)(u_inner, orientation_or_normal,
                                                                  direction,
                                                                  x, t,
                                                                  surface_flux_function, equations)
  u_boundary = boundary_condition.boundary_value_function(x, t, equations)

  # Calculate boundary flux
  if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation_or_normal, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation_or_normal, equations)
  end

  return flux
end

# Dirichlet-type boundary condition for use with UnstructuredMesh2D
# Note: For unstructured we lose the concept of an "absolute direction"
@inline function (boundary_condition::BoundaryConditionDirichlet)(u_inner,
                                                                  normal_direction::AbstractVector,
                                                                  x, t,
                                                                  surface_flux_function, equations)
  # get the external value of the solution
  u_boundary = boundary_condition.boundary_value_function(x, t, equations)

  # Calculate boundary flux
  flux = surface_flux_function(u_inner, u_boundary, normal_direction, equations)

  return flux
end

# operator types used for dispatch on parabolic boundary fluxes
struct Gradient end
struct Divergence end

"""
    BoundaryConditionNeumann(boundary_normal_flux_function)

Similar to `BoundaryConditionDirichlet`, but creates a Neumann boundary condition for parabolic
equations that uses the function `boundary_normal_flux_function` to specify the values of the normal
flux at the boundary.
The passed boundary value function will be called with the same arguments as an initial condition function is called, i.e., as
```julia
boundary_normal_flux_function(x, t, equations)
```
where `x` specifies the coordinates, `t` is the current time, and `equation` is the corresponding system of equations.
"""
struct BoundaryConditionNeumann{B}
  boundary_normal_flux_function::B
end

# set sensible default values that may be overwritten by specific equations
"""
    have_nonconservative_terms(equations)

Trait function determining whether `equations` represent a conservation law
with or without nonconservative terms. Classical conservation laws such as the
[`CompressibleEulerEquations2D`](@ref) do not have nonconservative terms. The
[`ShallowWaterEquations2D`](@ref) with non-constant bottom topography are an
example of equations with nonconservative terms.
The return value will be `True()` or `False()` to allow dispatching on the return type.
"""
have_nonconservative_terms(::AbstractEquations) = False()
have_constant_speed(::AbstractEquations) = False()

default_analysis_errors(::AbstractEquations)     = (:l2_error, :linf_error)
"""
    default_analysis_integrals(equations)

Default analysis integrals used by the [`AnalysisCallback`](@ref).
"""
default_analysis_integrals(::AbstractEquations)  = (entropy_timederivative,)


"""
    cons2cons(u, equations)

Return the conserved variables `u`. While this function is as trivial as `identity`,
it is also as useful.
"""
@inline cons2cons(u, ::AbstractEquations) = u

@inline Base.first(u, ::AbstractEquations) = first(u)

"""
    cons2prim(u, equations)

Convert the conserved variables `u` to the primitive variables for a given set of
`equations`. `u` is a vector type of the correct length `nvariables(equations)`.
Notice the function doesn't include any error checks for the purpose of efficiency,
so please make sure your input is correct.
The inverse conversion is performed by [`prim2cons`](@ref).
"""
function cons2prim end

"""
    prim2cons(u, equations)

Convert the primitive variables `u` to the conserved variables for a given set of
`equations`. `u` is a vector type of the correct length `nvariables(equations)`.
Notice the function doesn't include any error checks for the purpose of efficiency,
so please make sure your input is correct.
The inverse conversion is performed by [`cons2prim`](@ref).
"""
function prim2cons end

"""
    entropy(u, equations)

Return the chosen entropy of the conserved variables `u` for a given set of
`equations`.
"""
function entropy end

"""
    cons2entropy(u, equations)

Convert the conserved variables `u` to the entropy variables for a given set of
`equations` with chosen standard [`entropy`](@ref).
`u` is a vector type of the correct length `nvariables(equations)`.
Notice the function doesn't include any error checks for the purpose of efficiency,
so please make sure your input is correct.
The inverse conversion is performed by [`entropy2cons`](@ref).
"""
function cons2entropy end

"""
    entropy2cons(w, equations)

Convert the entropy variables `w` based on a standard [`entropy`](@ref) to the
conserved variables for a given set of `equations`.
`u` is a vector type of the correct length `nvariables(equations)`.
Notice the function doesn't include any error checks for the purpose of efficiency,
so please make sure your input is correct.
The inverse conversion is performed by [`cons2entropy`](@ref).
"""
function entropy2cons end

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

# Shallow water equations
abstract type AbstractShallowWaterEquations{NDIMS, NVARS} <: AbstractEquations{NDIMS, NVARS} end
include("shallow_water_1d.jl")
include("shallow_water_2d.jl")

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

# Linearized Euler equations
abstract type AbstractLinearizedEulerEquations{NDIMS, NVARS} <: AbstractEquations{NDIMS, NVARS} end
include("linearized_euler_2d.jl")

abstract type AbstractEquationsParabolic{NDIMS, NVARS} <: AbstractEquations{NDIMS, NVARS} end

end # @muladd
