
@doc raw"""
    LinearScalarAdvectionEquation

The linear scalar advection equation
```math
\partial_t u + a_1 \partial_1 u + a_2 \partial_2 u = 0
```
in two space dimensions with constant velocity `a`.
"""
struct LinearScalarAdvectionEquation <: AbstractEquation{1}
  sources::String
  advectionvelocity::SVector{2, Float64}
end

function LinearScalarAdvectionEquation()
  sources = parameter("sources", "none")
  a = convert(SVector{2,Float64}, parameter("advectionvelocity"))
  LinearScalarAdvectionEquation(sources, a)
end


get_name(::LinearScalarAdvectionEquation) = "LinearScalarAdvectionEquation"
varnames_cons(::LinearScalarAdvectionEquation) = SVector("scalar")
varnames_prim(::LinearScalarAdvectionEquation) = SVector("scalar")


# Set initial conditions at physical location `x` for time `t`
function initial_conditions_gauss(x, t, equation::LinearScalarAdvectionEquation)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  return @SVector [exp(-(x_trans[1]^2 + x_trans[2]^2))]
end

function initial_conditions_convergence_test(x, t, equation::LinearScalarAdvectionEquation)
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

function initial_conditions_sin_sin(x, t, equation::LinearScalarAdvectionEquation)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  scalar = sin(2 * pi * x_trans[1]) * sin(2 * pi * x_trans[2])
  return @SVector [scalar]
end

function initial_conditions_constant(x, t, equation::LinearScalarAdvectionEquation)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  return @SVector [2.0]
end

function initial_conditions_linear_x_y(x, t, equation::LinearScalarAdvectionEquation)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  return @SVector [sum(x_trans)]
end

function initial_conditions_linear_x(x, t, equation::LinearScalarAdvectionEquation)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  return @SVector [x_trans[1]]
end

function initial_conditions_linear_y(x, t, equation::LinearScalarAdvectionEquation)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  return @SVector [x_trans[2]]
end


# Pre-defined source terms should be implemented as
# function source_terms_WHATEVER(ut, u, x, element_id, t, n_nodes, equation::LinearScalarAdvectionEquation)


# Calculate 1D flux in for a single point
@inline function calcflux(u, orientation, equation::LinearScalarAdvectionEquation)
  a = equation.advectionvelocity[orientation]
  return a * u
end


function flux_lax_friedrichs(u_ll, u_rr, orientation, equation::LinearScalarAdvectionEquation)
  a = equation.advectionvelocity[orientation]
  return 0.5 * ( a * (u_ll + u_rr) - abs(a) * (u_rr - u_ll) )
end


# Determine maximum stable time step based on polynomial degree and CFL number
function calc_max_dt(u, element_id, n_nodes, invjacobian, cfl,
                     equation::LinearScalarAdvectionEquation)
  return cfl * 2 / (invjacobian * maximum(abs.(equation.advectionvelocity))) / n_nodes
end


# Convert conservative variables to primitive
cons2prim(cons, equation::LinearScalarAdvectionEquation) = cons

# Convert conservative variables to entropy variables
cons2entropy(cons, n_nodes, n_elements, equation::LinearScalarAdvectionEquation) = cons


# Calculate entropy for a conservative state `cons`
@inline entropy(cons::Real, ::LinearScalarAdvectionEquation) = cons^2 / 2
@inline entropy(cons, equation::LinearScalarAdvectionEquation) = entropy(cons[1], equation)


# Calculate total energy for a conservative state `cons`
@inline energy_total(cons::Real, ::LinearScalarAdvectionEquation) = cons^2 / 2
@inline energy_total(cons, equation::LinearScalarAdvectionEquation) = energy_total(cons[1], equation)
