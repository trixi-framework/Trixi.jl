
@doc raw"""
    LinearAdvectionDiffusionEquation2D

The linear scalar advection equation
```math
\partial_t u + a_1 \partial_1 u + a_2 \partial_2 u = 0
```
in two space dimensions with constant velocity `a`.
"""
struct LinearAdvectionDiffusionEquation2D{RealT<:Real} <: AbstractLinearAdvectionDiffusionEquation{2, 1}
  advectionvelocity::SVector{2, RealT}
  nu::RealT
end

function LinearAdvectionDiffusionEquation2D(a::NTuple{2,<:Real}, nu::Real)
  LinearAdvectionDiffusionEquation2D(SVector(a), nu)
end

function LinearAdvectionDiffusionEquation2D(a1::Real, a2::Real, nu::Real)
  a1, a2, nu = promote(a1, a2, nu)
  LinearAdvectionDiffusionEquation2D(SVector(a1, a2), nu)
end

get_name(::LinearAdvectionDiffusionEquation2D) = "LinearAdvectionDiffusionEquation2D"
varnames(::typeof(cons2cons), ::LinearAdvectionDiffusionEquation2D) = SVector("scalar")
varnames(::typeof(cons2prim), ::LinearAdvectionDiffusionEquation2D) = SVector("scalar")

have_parabolic_terms(::LinearAdvectionDiffusionEquation2D) = Val(true)


# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_constant(x, t, equations::LinearAdvectionDiffusionEquation2D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t, equation::LinearAdvectionDiffusionEquation2D)
  return @SVector [2.0]
end


"""
    initial_condition_convergence_test(x, t, equations::LinearAdvectionDiffusionEquation2D)

A smooth initial condition used for convergence tests.
"""
function initial_condition_convergence_test(x, t, equation::LinearAdvectionDiffusionEquation2D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  @unpack nu = equation
  c = 1.0
  A = 0.5
  L = 2
  f = 1/L
  omega = 2 * pi * f
  scalar = c + A * sin(omega * sum(x_trans)) * exp(-2 * nu * omega^2 * t)
  return @SVector [scalar]
end


# Pre-defined source terms should be implemented as
# function source_terms_WHATEVER(u, x, t, equations::LinearAdvectionDiffusionEquation2D)


# Calculate 1D flux in for a single point
@inline function calcflux(u, orientation, equation::LinearAdvectionDiffusionEquation2D)
  a = equation.advectionvelocity[orientation]
  return a * u
end


function flux_lax_friedrichs(u_ll, u_rr, orientation, equation::LinearAdvectionDiffusionEquation2D)
  a = equation.advectionvelocity[orientation]
  return 0.5 * ( a * (u_ll + u_rr) - abs(a) * (u_rr - u_ll) )
end



@inline have_constant_speed(::LinearAdvectionDiffusionEquation2D) = Val(true)

@inline function max_abs_speeds(equation::LinearAdvectionDiffusionEquation2D)
  return abs.(equation.advectionvelocity)
end


# Convert conservative variables to primitive
@inline cons2prim(u, equation::LinearAdvectionDiffusionEquation2D) = u

# Convert conservative variables to entropy variables
@inline cons2entropy(u, equation::LinearAdvectionDiffusionEquation2D) = u


# Calculate entropy for a conservative state `cons`
@inline entropy(u::Real, ::LinearAdvectionDiffusionEquation2D) = 0.5 * u^2
@inline entropy(u, equation::LinearAdvectionDiffusionEquation2D) = entropy(u[1], equation)


# Calculate total energy for a conservative state `cons`
@inline energy_total(u::Real, ::LinearAdvectionDiffusionEquation2D) = 0.5 * u^2
@inline energy_total(u, equation::LinearAdvectionDiffusionEquation2D) = energy_total(u[1], equation)
