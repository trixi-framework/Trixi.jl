
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

# TODO Taal refactor, remove old constructors and replace them with default values
function LinearScalarAdvectionEquation1D()
  a = convert(SVector{1,Float64}, parameter("advectionvelocity"))
  LinearScalarAdvectionEquation1D(a)
end


get_name(::LinearScalarAdvectionEquation1D) = "LinearScalarAdvectionEquation1D"
varnames_cons(::LinearScalarAdvectionEquation1D) = SVector("scalar")
varnames_prim(::LinearScalarAdvectionEquation1D) = SVector("scalar")


# Set initial conditions at physical location `x` for time `t`
function initial_condition_gauss(x, t, equation::LinearScalarAdvectionEquation1D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  return @SVector [exp(-(x_trans[1]^2 ))]
end

function initial_condition_convergence_test(x, t, equation::LinearScalarAdvectionEquation1D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  c = 1.0
  A = 0.5
  L = 2
  f = 1/L
  omega = 2 * pi * f
  scalar = c + A * sin(omega * sum(x_trans))
  return @SVector [scalar]
end

function initial_condition_sin(x, t, equation::LinearScalarAdvectionEquation1D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  scalar = sin(2 * pi * x_trans[1])
  return @SVector [scalar]
end

function initial_condition_constant(x, t, equation::LinearScalarAdvectionEquation1D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  return @SVector [2.0]
end


function initial_condition_linear_x(x, t, equation::LinearScalarAdvectionEquation1D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  return @SVector [x_trans[1]]
end

# Apply boundary conditions
function boundary_condition_linear_x(u_inner, orientation, direction, x, t, surface_flux_function,
                                      equation::LinearScalarAdvectionEquation1D)
  u_boundary = initial_condition_linear_x(x, t, equation)

  # Calculate boundary flux
  if direction == 2  # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equation)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equation)
  end

  return flux
end


function boundary_condition_gauss(u_inner, orientation, direction, x, t, surface_flux_function,
                                   equation::LinearScalarAdvectionEquation1D)
  u_boundary = initial_condition_gauss(x, t, equation)

  # Calculate boundary flux
  if direction == 2  # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equation)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equation)
  end

  return flux
end


function boundary_condition_convergence_test(u_inner, orientation, direction, x, t,
                                              surface_flux_function,
                                              equation::LinearScalarAdvectionEquation1D)
  u_boundary = initial_condition_convergence_test(x, t, equation)

  # Calculate boundary flux
  if direction == 2  # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equation)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equation)
  end

  return flux
end


# Pre-defined source terms should be implemented as
# function source_terms_WHATEVER(ut, u, x, element_id, t, n_nodes, equation::LinearScalarAdvectionEquation2D)


# Calculate 1D flux in for a single point
@inline function calcflux(u, orientation, equation::LinearScalarAdvectionEquation1D)
  a = equation.advectionvelocity[orientation]
  return a * u
end


function flux_lax_friedrichs(u_ll, u_rr, orientation, equation::LinearScalarAdvectionEquation1D)
  a = equation.advectionvelocity[orientation]
  return 0.5 * ( a * (u_ll + u_rr) - abs(a) * (u_rr - u_ll) )
end


# Determine maximum stable time step based on polynomial degree and CFL number
function calc_max_dt(u, element_id, invjacobian, cfl,
                     equation::LinearScalarAdvectionEquation1D, dg)
  λ_max = maximum(abs, equation.advectionvelocity)
  return cfl * 2 / (nnodes(dg) * invjacobian * λ_max)
end

@inline have_constant_speed(::LinearScalarAdvectionEquation1D) = Val(true)

@inline function max_abs_speeds(eq::LinearScalarAdvectionEquation1D)
  return abs.(eq.advectionvelocity)
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
