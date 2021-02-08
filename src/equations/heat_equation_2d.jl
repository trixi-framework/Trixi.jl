
@doc raw"""
    HeatEquation2D

The heat equation
```math
\partial_t u - \nu (\partial^2_1 u + \partial^2_2 u) = 0
```
in two space dimensions with constant viscosity ``\nu``.

!!! warning "Experimental code"
    This system of equations is experimental and can change any time.
"""
struct HeatEquation2D{RealT<:Real} <: AbstractHeatEquation{2, 1}
  nu::RealT
end

get_name(::HeatEquation2D) = "HeatEquation2D"
varnames(::typeof(cons2cons), ::HeatEquation2D) = SVector("scalar")
varnames(::typeof(cons2prim), ::HeatEquation2D) = SVector("scalar")


# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_constant(x, t, equations::HeatEquation2D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t, equation::HeatEquation2D)
  return @SVector [2.0]
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


"""
    initial_condition_gauss(x, t, equation::HeatEquation2D)

A Gaussian pulse used together with
[`boundary_condition_gauss`](@ref).
"""
function initial_condition_gauss(x, t, equation::HeatEquation2D)
  return @SVector [exp(-(x[1]^2 + x[2]^2))]
end


function initial_condition_sin_x(x, t, equation::HeatEquation2D)
  @unpack nu = equation

  omega = pi
  scalar = sin(omega * x[1]) * exp(-nu * omega^2 * t)

  return @SVector [scalar]
end


function boundary_condition_sin_x(u_inner, gradients_inner, orientation, direction, x, t,
                                  surface_flux_function,
                                  equation::HeatEquation2D)
  # OBS! This is specifically implemented for BR1
  return calcflux(u_inner, gradients_inner, orientation, equation)
end


function initial_condition_poisson_periodic(x, t, equation::HeatEquation2D)
  # elliptic equation: -νΔϕ = f
  # depending on initial constant state, c, for phi this converges to the solution ϕ + c
  @unpack nu = equation

  phi = sin(2.0*pi*x[1])*sin(2.0*pi*x[2])*(1 - exp(-8*nu*pi^2*t))
  return @SVector [phi]
end


@inline function source_terms_poisson_periodic(u, x, t, equation::HeatEquation2D)
  # elliptic equation: -νΔϕ = f
  # analytical solution: phi = sin(2πx)*sin(2πy) and f = -8νπ^2 sin(2πx)*sin(2πy)
  C = -8 * equation.nu * pi^2

  x1, x2 = x
  tmp1 = sinpi(2 * x1)
  tmp2 = sinpi(2 * x2)
  du1 = -C*tmp1*tmp2

  return SVector(du1)
end


# Calculate parabolic 1D flux in axis `orientation` for a single point
@inline function calcflux(u, gradients, orientation, equation::HeatEquation2D)
  return -equation.nu*gradients[orientation]
end


@inline have_constant_diffusion(::HeatEquation2D) = Val(true)

# FIXME: Find a better name than `max_abs_diffusions` or `max_abs_speeds_viscous`
@inline function max_abs_diffusions(equation::HeatEquation2D)
  @unpack nu = equation
  return (nu, nu)
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
