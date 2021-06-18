
@doc raw"""
    LinearScalarAdvectionEquation2D

The linear scalar advection equation
```math
\partial_t u + a_1 \partial_1 u + a_2 \partial_2 u = 0
```
in two space dimensions with constant velocity `a`.
"""
struct LinearScalarAdvectionEquation2D{RealT<:Real} <: AbstractLinearScalarAdvectionEquation{2, 1}
  advectionvelocity::SVector{2, RealT}
end

function LinearScalarAdvectionEquation2D(a::NTuple{2,<:Real})
  LinearScalarAdvectionEquation2D(SVector(a))
end

function LinearScalarAdvectionEquation2D(a1::Real, a2::Real)
  LinearScalarAdvectionEquation2D(SVector(a1, a2))
end


varnames(::typeof(cons2cons), ::LinearScalarAdvectionEquation2D) = ("scalar", )
varnames(::typeof(cons2prim), ::LinearScalarAdvectionEquation2D) = ("scalar", )

# Calculates translated coordinates `x` for a periodic domain
function x_trans_periodic_2d(x, domain_length = SVector(10, 10), center = SVector(0, 0))
  x_normalized = x .- center
  x_shifted = x_normalized .% domain_length
  x_offset = ((x_shifted .< -0.5*domain_length) - (x_shifted .> 0.5*domain_length)) .* domain_length
  return center + x_shifted + x_offset
end

# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_constant(x, t, equations::LinearScalarAdvectionEquation2D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t, equation::LinearScalarAdvectionEquation2D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x_trans_periodic_2d(x - equation.advectionvelocity * t)

  return SVector(2.0)
end


"""
    initial_condition_convergence_test(x, t, equations::LinearScalarAdvectionEquation2D)

A smooth initial condition used for convergence tests.
"""
function initial_condition_convergence_test(x, t, equation::LinearScalarAdvectionEquation2D)
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
    initial_condition_gauss(x, t, equation::LinearScalarAdvectionEquation2D)

A Gaussian pulse used together with
[`BoundaryConditionDirichlet(initial_condition_gauss)`](@ref).
"""
function initial_condition_gauss(x, t, equation::LinearScalarAdvectionEquation2D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x_trans_periodic_2d(x - equation.advectionvelocity * t)

  scalar = exp(-(x_trans[1]^2 + x_trans[2]^2))
  return SVector(scalar)
end


"""
    initial_condition_sin_sin(x, t, equations::LinearScalarAdvectionEquation2D)

A sine wave in the conserved variable.
"""
function initial_condition_sin_sin(x, t, equation::LinearScalarAdvectionEquation2D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  scalar = sinpi(2 * x_trans[1]) * sinpi(2 * x_trans[2])
  return SVector(scalar)
end


"""
    initial_condition_linear_x_y(x, t, equations::LinearScalarAdvectionEquation2D)

A linear function of `x[1] + x[2]` used together with
[`boundary_condition_linear_x_y`](@ref).
"""
function initial_condition_linear_x_y(x, t, equation::LinearScalarAdvectionEquation2D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  return SVector(sum(x_trans))
end

"""
    boundary_condition_linear_x_y(u_inner, orientation, direction, x, t,
                                  surface_flux_function,
                                  equation::LinearScalarAdvectionEquation2D)

Boundary conditions for
[`initial_condition_linear_x_y`](@ref).
"""
function boundary_condition_linear_x_y(u_inner, orientation, direction, x, t,
                                       surface_flux_function,
                                       equation::LinearScalarAdvectionEquation2D)
  u_boundary = initial_condition_linear_x_y(x, t, equation)

  # Calculate boundary flux
  if direction in (2, 4) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equation)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equation)
  end

  return flux
end


"""
    initial_condition_linear_x(x, t, equations::LinearScalarAdvectionEquation2D)

A linear function of `x[1]` used together with
[`boundary_condition_linear_x`](@ref).
"""
function initial_condition_linear_x(x, t, equation::LinearScalarAdvectionEquation2D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  return SVector(x_trans[1])
end

"""
    boundary_condition_linear_x(u_inner, orientation, direction, x, t,
                                surface_flux_function,
                                equation::LinearScalarAdvectionEquation2D)

Boundary conditions for
[`initial_condition_linear_x`](@ref).
"""
function boundary_condition_linear_x(u_inner, orientation, direction, x, t,
                                     surface_flux_function,
                                     equation::LinearScalarAdvectionEquation2D)
  u_boundary = initial_condition_linear_x(x, t, equation)

  # Calculate boundary flux
  if direction in (2, 4) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equation)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equation)
  end

  return flux
end


"""
    initial_condition_linear_y(x, t, equations::LinearScalarAdvectionEquation2D)

A linear function of `x[1]` used together with
[`boundary_condition_linear_y`](@ref).
"""
function initial_condition_linear_y(x, t, equation::LinearScalarAdvectionEquation2D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  return SVector(x_trans[2])
end

"""
    boundary_condition_linear_y(u_inner, orientation, direction, x, t,
                                surface_flux_function,
                                equation::LinearScalarAdvectionEquation2D)

Boundary conditions for
[`initial_condition_linear_y`](@ref).
"""
function boundary_condition_linear_y(u_inner, orientation, direction, x, t,
                                     surface_flux_function,
                                     equation::LinearScalarAdvectionEquation2D)
  u_boundary = initial_condition_linear_y(x, t, equation)

  # Calculate boundary flux
  if direction in (2, 4) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equation)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equation)
  end

  return flux
end


# Pre-defined source terms should be implemented as
# function source_terms_WHATEVER(u, x, t, equations::LinearScalarAdvectionEquation2D)


# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer, equation::LinearScalarAdvectionEquation2D)
  a = equation.advectionvelocity[orientation]
  return a * u
end


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equation::LinearScalarAdvectionEquation2D)
  λ_max = abs(equation.advectionvelocity[orientation])
end


# Calculate 1D flux for a single point in the normal direction
# Note, this directional vector is not normalized
@inline function flux(u, normal_direction::AbstractVector, equation::LinearScalarAdvectionEquation2D)
  a = dot(equation.advectionvelocity, normal_direction) # velocity in normal direction
  return a * u
end


# Calculate maximum wave speed in the normal direction for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equation::LinearScalarAdvectionEquation2D)
  a = dot(equation.advectionvelocity, normal_direction) # velocity in normal direction
  return abs(a)
end


@inline have_constant_speed(::LinearScalarAdvectionEquation2D) = Val(true)

@inline function max_abs_speeds(equation::LinearScalarAdvectionEquation2D)
  return abs.(equation.advectionvelocity)
end


# Convert conservative variables to primitive
@inline cons2prim(u, equation::LinearScalarAdvectionEquation2D) = u

# Convert conservative variables to entropy variables
@inline cons2entropy(u, equation::LinearScalarAdvectionEquation2D) = u


# Calculate entropy for a conservative state `cons`
@inline entropy(u::Real, ::LinearScalarAdvectionEquation2D) = 0.5 * u^2
@inline entropy(u, equation::LinearScalarAdvectionEquation2D) = entropy(u[1], equation)


# Calculate total energy for a conservative state `cons`
@inline energy_total(u::Real, ::LinearScalarAdvectionEquation2D) = 0.5 * u^2
@inline energy_total(u, equation::LinearScalarAdvectionEquation2D) = energy_total(u[1], equation)
