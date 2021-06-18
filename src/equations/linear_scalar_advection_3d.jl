
@doc raw"""
    LinearScalarAdvectionEquation3D

The linear scalar advection equation
```math
\partial_t u + a_1 \partial_1 u + a_2 \partial_2 u + a_3 \partial_3 u = 0
```
in three space dimensions with constant velocity `a`.
"""
struct LinearScalarAdvectionEquation3D{RealT<:Real} <: AbstractLinearScalarAdvectionEquation{3, 1}
  advectionvelocity::SVector{3, RealT}
end

function LinearScalarAdvectionEquation3D(a::NTuple{3,<:Real})
  LinearScalarAdvectionEquation3D(SVector(a))
end

function LinearScalarAdvectionEquation3D(a1::Real, a2::Real, a3::Real)
  LinearScalarAdvectionEquation3D(SVector(a1, a2, a3))
end


varnames(::typeof(cons2cons), ::LinearScalarAdvectionEquation3D) = ("scalar", )
varnames(::typeof(cons2prim), ::LinearScalarAdvectionEquation3D) = ("scalar", )


# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_constant(x, t, equations::LinearScalarAdvectionEquation1D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t, equation::LinearScalarAdvectionEquation3D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  return SVector(2.0)
end


"""
    initial_condition_convergence_test(x, t, equations::LinearScalarAdvectionEquation1D)

A smooth initial condition used for convergence tests.
"""
function initial_condition_convergence_test(x, t, equation::LinearScalarAdvectionEquation3D)
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

A Gaussian pulse.
"""
function initial_condition_gauss(x, t, equation::LinearScalarAdvectionEquation3D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  scalar = exp(-(x_trans[1]^2 + x_trans[2]^2 + x_trans[3]^2))
  return SVector(scalar)
end


"""
    initial_condition_sin(x, t, equations::LinearScalarAdvectionEquation1D)

A sine wave in the conserved variable.
"""
function initial_condition_sin(x, t, equation::LinearScalarAdvectionEquation3D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  scalar = sin(2 * pi * x_trans[1]) * sin(2 * pi * x_trans[2]) * sin(2 * pi * x_trans[3])
  return SVector(scalar)
end


"""
    initial_condition_linear_z(x, t, equations::LinearScalarAdvectionEquation1D)

A linear function of `x[3]` used together with
[`boundary_condition_linear_z`](@ref).
"""
function initial_condition_linear_z(x, t, equation::LinearScalarAdvectionEquation3D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  return SVector(x_trans[3])
end

"""
    boundary_condition_linear_z(u_inner, orientation, direction, x, t,
                                surface_flux_function,
                                equation::LinearScalarAdvectionEquation1D)

Boundary conditions for
[`boundary_condition_linear_z`](@ref).
"""
function boundary_condition_linear_z(u_inner, orientation, direction, x, t,
                                     surface_flux_function,
                                     equation::LinearScalarAdvectionEquation3D)
  u_boundary = initial_condition_linear_z(x, t, equation)

  # Calculate boundary flux
  if direction in (2, 4, 6) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equation)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equation)
  end

  return flux
end


# Pre-defined source terms should be implemented as
# function source_terms_WHATEVER(u, x, t, equation::LinearScalarAdvectionEquation3D)


# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer, equation::LinearScalarAdvectionEquation3D)
  a = equation.advectionvelocity[orientation]
  return a * u
end


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equation::LinearScalarAdvectionEquation3D)
  λ_max = abs(equation.advectionvelocity[orientation])
end


# Calculate 1D flux for a single point in the normal direction
# Note, this directional vector is not normalized
@inline function flux(u, normal_direction::AbstractVector, equation::LinearScalarAdvectionEquation3D)
  a = dot(equation.advectionvelocity, normal_direction) # velocity in normal direction
  return a * u
end


# Calculate maximum wave speed in the normal direction for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equation::LinearScalarAdvectionEquation3D)
  a = dot(equation.advectionvelocity, normal_direction) # velocity in normal direction
  return abs(a)
end


@inline have_constant_speed(::LinearScalarAdvectionEquation3D) = Val(true)

@inline function max_abs_speeds(equation::LinearScalarAdvectionEquation3D)
  return abs.(equation.advectionvelocity)
end


# Convert conservative variables to primitive
@inline cons2prim(u, equation::LinearScalarAdvectionEquation3D) = u

# Convert conservative variables to entropy variables
@inline cons2entropy(u, equation::LinearScalarAdvectionEquation3D) = u


# Calculate entropy for a conservative state `cons`
@inline entropy(u::Real, ::LinearScalarAdvectionEquation3D) = 0.5 * u^2
@inline entropy(u, equation::LinearScalarAdvectionEquation3D) = entropy(u[1], equation)


# Calculate total energy for a conservative state `cons`
@inline energy_total(u::Real, ::LinearScalarAdvectionEquation3D) = 0.5 * u^2
@inline energy_total(u, equation::LinearScalarAdvectionEquation3D) = energy_total(u[1], equation)
