
@doc raw"""
    LinearScalarAdvectionEquation3D

The linear scalar advection equation
```math
\partial_t u + a_1 \partial_1 u + a_2 \partial_2 u + a_3 \partial_3 u = 0
```
in two space dimensions with constant velocity `a`.
"""
struct LinearScalarAdvectionEquation3D <: AbstractLinearScalarAdvectionEquation{3, 1}
  sources::String
  advectionvelocity::SVector{3, Float64}
end

function LinearScalarAdvectionEquation3D()
  sources = parameter("sources", "none")
  a = convert(SVector{3,Float64}, parameter("advectionvelocity"))
  LinearScalarAdvectionEquation3D(sources, a)
end


get_name(::LinearScalarAdvectionEquation3D) = "LinearScalarAdvectionEquation3D"
varnames_cons(::LinearScalarAdvectionEquation3D) = SVector("scalar")
varnames_prim(::LinearScalarAdvectionEquation3D) = SVector("scalar")


# Set initial conditions at physical location `x` for time `t`
function initial_conditions_gauss(x, t, equation::LinearScalarAdvectionEquation3D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  return @SVector [exp(-(x_trans[1]^2 + x_trans[2]^2 + x_trans[3]^2))]
end

function initial_conditions_convergence_test(x, t, equation::LinearScalarAdvectionEquation3D)
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

function initial_conditions_sin_periodic(x, t, equation::LinearScalarAdvectionEquation3D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  scalar = sin(2 * pi * x_trans[1]) * sin(2 * pi * x_trans[2]) * sin(2 * pi * x_trans[3])
  return @SVector [scalar]
end

function initial_conditions_constant(x, t, equation::LinearScalarAdvectionEquation3D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  return @SVector [2.0]
end

function initial_conditions_linear_z(x, t, equation::LinearScalarAdvectionEquation3D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  return @SVector [x_trans[3]]
end


# Pre-defined source terms should be implemented as
# function source_terms_WHATEVER(ut, u, x, element_id, t, n_nodes, equation::LinearScalarAdvectionEquation3D)


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


# Convert conservative variables to primitive
cons2prim(cons, equation::LinearScalarAdvectionEquation3D) = cons

# Convert conservative variables to entropy variables
cons2entropy(u, equation::LinearScalarAdvectionEquation3D) = u


# Calculate entropy for a conservative state `cons`
@inline entropy(cons::Real, ::LinearScalarAdvectionEquation3D) = cons^2 / 2
@inline entropy(cons, equation::LinearScalarAdvectionEquation3D) = entropy(cons[1], equation)


# Calculate total energy for a conservative state `cons`
@inline energy_total(cons::Real, ::LinearScalarAdvectionEquation3D) = cons^2 / 2
@inline energy_total(cons, equation::LinearScalarAdvectionEquation3D) = energy_total(cons[1], equation)
