
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

# TODO Taal refactor, remove old constructors and replace them with default values
function LinearScalarAdvectionEquation3D()
  a = convert(SVector{3,Float64}, parameter("advectionvelocity"))
  LinearScalarAdvectionEquation3D(a)
end


get_name(::LinearScalarAdvectionEquation3D) = "LinearScalarAdvectionEquation3D"
varnames_cons(::LinearScalarAdvectionEquation3D) = SVector("scalar")
varnames_prim(::LinearScalarAdvectionEquation3D) = SVector("scalar")


# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_constant(x, t, equations::LinearScalarAdvectionEquation1D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t, equation::LinearScalarAdvectionEquation3D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  return @SVector [2.0]
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
  return @SVector [scalar]
end


"""
    initial_condition_gauss(x, t, equations::LinearScalarAdvectionEquation1D)

A Gaussian pulse.
"""
function initial_condition_gauss(x, t, equation::LinearScalarAdvectionEquation3D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  return @SVector [exp(-(x_trans[1]^2 + x_trans[2]^2 + x_trans[3]^2))]
end


"""
    initial_condition_sin(x, t, equations::LinearScalarAdvectionEquation1D)

A sine wave in the conserved variable.
"""
function initial_condition_sin(x, t, equation::LinearScalarAdvectionEquation3D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  scalar = sin(2 * pi * x_trans[1]) * sin(2 * pi * x_trans[2]) * sin(2 * pi * x_trans[3])
  return @SVector [scalar]
end


"""
    initial_condition_linear_z(x, t, equations::LinearScalarAdvectionEquation1D)

A linear function of `x[3]` used together with
[`boundary_condition_linear_z`](@ref).
"""
function initial_condition_linear_z(x, t, equation::LinearScalarAdvectionEquation3D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  return @SVector [x_trans[3]]
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
@inline function calcflux(u, orientation, equation::LinearScalarAdvectionEquation3D)
  a = equation.advectionvelocity[orientation]
  return a * u
end


function flux_lax_friedrichs(u_ll, u_rr, orientation, equation::LinearScalarAdvectionEquation3D)
  a = equation.advectionvelocity[orientation]
  return 0.5 * ( a * (u_ll + u_rr) - abs(a) * (u_rr - u_ll) )
end


# Determine maximum stable time step based on polynomial degree and CFL number
function calc_max_dt(u, element_id, invjacobian, cfl,
                     equation::LinearScalarAdvectionEquation3D, dg)
  λ_max = maximum(abs, equation.advectionvelocity)
  return cfl * 2 / (nnodes(dg) * invjacobian * λ_max)
end

@inline have_constant_speed(::LinearScalarAdvectionEquation3D) = Val(true)

@inline function max_abs_speeds(equation::LinearScalarAdvectionEquation3D)
  # FIXME Taal restore after Taam sync
  # return abs.(equation.advectionvelocity)
  return maximum(abs.(equation.advectionvelocity)), 0.0, 0.0
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
