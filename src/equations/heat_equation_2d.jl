
@doc raw"""
    HeatEquation2D

The linear scalar advection equation
```math
\partial_t u + a_1 \partial_1 u + a_2 \partial_2 u = 0
```
in two space dimensions with constant velocity `a`.
"""
struct HeatEquation2D{RealT<:Real} <: AbstractHeatEquation{2, 1}
  nu::RealT
end

get_name(::HeatEquation2D) = "HeatEquation2D"
varnames(::typeof(cons2cons), ::HeatEquation2D) = SVector("scalar")
varnames(::typeof(cons2prim), ::HeatEquation2D) = SVector("scalar")

have_parabolic_terms(::HeatEquation2D) = Val(true)


# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_constant(x, t, equations::HeatEquation2D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t, equation::HeatEquation2D)
  return @SVector [2.0]
end

function initial_condition_linear_xy(x, t, equation::HeatEquation2D)
  return @SVector [2*x[1] + 3*x[2]]
end


"""
    initial_condition_convergence_test(x, t, equations::HeatEquation2D)

A smooth initial condition used for convergence tests.
"""
function initial_condition_convergence_test(x, t, equation::HeatEquation2D)
  @unpack nu = equation
  c = 1.0
  A = 0.5
  L = 2
  f = 1/L
  omega = 2 * pi * f
  scalar = c + A * sin(omega * sum(x)) * exp(-2 * nu * omega^2 * t)
  return @SVector [scalar]
end


# Pre-defined source terms should be implemented as
# function source_terms_WHATEVER(u, x, t, equations::HeatEquation2D)


# Calculate parabolic 1D flux in axis `orientation` for a single point
@inline function calcflux(u, gradients, orientation, equation::HeatEquation2D)
  return equation.nu*gradients[orientation]
end


@inline have_constant_speed(::HeatEquation2D) = Val(true)

@inline function max_abs_speeds(equation::HeatEquation2D)
  return NaN
end


# Convert conservative variables to primitive
@inline cons2prim(u, equation::HeatEquation2D) = u

# Convert conservative variables to entropy variables
@inline cons2entropy(u, equation::HeatEquation2D) = u


# Calculate entropy for a conservative state `cons`
@inline entropy(u::Real, ::HeatEquation2D) = 0.5 * u^2
@inline entropy(u, equation::HeatEquation2D) = entropy(u[1], equation)


# Calculate total energy for a conservative state `cons`
@inline energy_total(u::Real, ::HeatEquation2D) = 0.5 * u^2
@inline energy_total(u, equation::HeatEquation2D) = energy_total(u[1], equation)
