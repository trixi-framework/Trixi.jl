# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


@doc raw"""
    LinearScalarAdvectionEquation1D

The linear scalar advection equation
```math
\partial_t u + a \partial_1 u  = 0
```
in one space dimension with constant velocity `a`.
"""
struct LinearScalarAdvectionEquation1D{RealT<:Real} <: AbstractLinearScalarAdvectionEquation{1, 1}
  advectionvelocity::SVector{1, RealT}
end

function LinearScalarAdvectionEquation1D(a::Real)
  LinearScalarAdvectionEquation1D(SVector(a))
end


varnames(::typeof(cons2cons), ::LinearScalarAdvectionEquation1D) = ("scalar", )
varnames(::typeof(cons2prim), ::LinearScalarAdvectionEquation1D) = ("scalar", )


# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_constant(x, t, equations::LinearScalarAdvectionEquation1D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t, equation::LinearScalarAdvectionEquation1D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  return SVector(2.0)
end


"""
    initial_condition_convergence_test(x, t, equations::LinearScalarAdvectionEquation1D)

A smooth initial condition used for convergence tests
(in combination with [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref)
in non-periodic domains).
"""
function initial_condition_convergence_test(x, t, equation::LinearScalarAdvectionEquation1D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  c = 1.0
  A = 0.5
  L = 2
  f = 1/L
  omega = 2 * pi * f
  scalar = c + A * sin(omega * sum(x_trans))
  return SVector(scalar)
end


"""
    initial_condition_gauss(x, t, equations::LinearScalarAdvectionEquation1D)

A Gaussian pulse used together with
[`BoundaryConditionDirichlet(initial_condition_gauss)`](@ref).
"""
function initial_condition_gauss(x, t, equation::LinearScalarAdvectionEquation1D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  scalar = exp(-(x_trans[1]^2))
  return SVector(scalar)
end


"""
    initial_condition_sin(x, t, equations::LinearScalarAdvectionEquation1D)

A sine wave in the conserved variable.
"""
function initial_condition_sin(x, t, equation::LinearScalarAdvectionEquation1D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  scalar = sinpi(2 * x_trans[1])
  return SVector(scalar)
end


# Pre-defined source terms should be implemented as
# function source_terms_WHATEVER(u, x, t, equations::LinearScalarAdvectionEquation1D)


# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer, equation::LinearScalarAdvectionEquation1D)
  a = equation.advectionvelocity[orientation]
  return a * u
end


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation, equation::LinearScalarAdvectionEquation1D)
  λ_max = abs(equation.advectionvelocity[orientation])
end


@inline have_constant_speed(::LinearScalarAdvectionEquation1D) = Val(true)

@inline function max_abs_speeds(equation::LinearScalarAdvectionEquation1D)
  return abs.(equation.advectionvelocity)
end


# Convert conservative variables to primitive
@inline cons2prim(u, equation::LinearScalarAdvectionEquation1D) = u

# Convert conservative variables to entropy variables
@inline cons2entropy(u, equation::LinearScalarAdvectionEquation1D) = u


# Calculate entropy for a conservative state `cons`
@inline entropy(u::Real, ::LinearScalarAdvectionEquation1D) = 0.5 * u^2
@inline entropy(u, equation::LinearScalarAdvectionEquation1D) = entropy(u[1], equation)


# Calculate total energy for a conservative state `cons`
@inline energy_total(u::Real, ::LinearScalarAdvectionEquation1D) = 0.5 * u^2
@inline energy_total(u, equation::LinearScalarAdvectionEquation1D) = energy_total(u[1], equation)


end # @muladd
