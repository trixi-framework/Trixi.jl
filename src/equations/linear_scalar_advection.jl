
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
function initial_conditions_gauss(equation::LinearScalarAdvectionEquation, x, t)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  return @SVector [exp(-(x_trans[1]^2 + x_trans[2]^2))]
end

function initial_conditions_convergence_test(equation::LinearScalarAdvectionEquation, x, t)
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

function initial_conditions_sin_sin(equation::LinearScalarAdvectionEquation, x, t)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  scalar = sin(2 * pi * x_trans[1]) * sin(2 * pi * x_trans[2])
  return @SVector [scalar]
end

function initial_conditions_constant(equation::LinearScalarAdvectionEquation, x, t)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  return @SVector [2.0]
end

function initial_conditions_linear_x_y(equation::LinearScalarAdvectionEquation, x, t)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  return @SVector [sum(x_trans)]
end

function initial_conditions_linear_x(equation::LinearScalarAdvectionEquation, x, t)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  return @SVector [x_trans[1]]
end

function initial_conditions_linear_y(equation::LinearScalarAdvectionEquation, x, t)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  return @SVector [x_trans[2]]
end


# Pre-defined source terms should be implemented as
# function source_terms_WHATEVER(equation::LinearScalarAdvectionEquation, ut, u, x, element_id, t, n_nodes)


# Calculate 1D flux in for a single point
@inline function calcflux(equation::LinearScalarAdvectionEquation, orientation, u)
  a = equation.advectionvelocity[orientation]
  a * u
end


# Calculate 2D flux (element version)
@inline function calcflux!(f1::AbstractArray{Float64},
                           f2::AbstractArray{Float64},
                           equation::LinearScalarAdvectionEquation,
                           u::AbstractArray{Float64}, element_id::Int,
                           n_nodes::Int)
  for j = 1:n_nodes
    for i = 1:n_nodes
      @views calcflux!(f1[:, i, j], f2[:, i, j], equation, u[:, i, j, element_id])
    end
  end
end


# Calculate 2D flux (pointwise version)
@inline function calcflux!(f1::AbstractArray{Float64},
                           f2::AbstractArray{Float64},
                           equation::LinearScalarAdvectionEquation,
                           u::AbstractArray{Float64})
  f1[1] = u[1] * equation.advectionvelocity[1]
  f2[1] = u[1] * equation.advectionvelocity[2]
  return nothing
end


function flux_lax_friedrichs(equation::LinearScalarAdvectionEquation, orientation, u_ll, u_rr)
  a = equation.advectionvelocity[orientation]
  return 0.5 * ( a * (u_ll + u_rr) - abs(a) * (u_rr - u_ll) )
end


# Determine maximum stable time step based on polynomial degree and CFL number
function calc_max_dt(equation::LinearScalarAdvectionEquation,
                     u::Array{Float64, 4}, element_id::Int,
                     n_nodes::Int, invjacobian::Float64,
                     cfl::Float64)
  return cfl * 2 / (invjacobian * maximum(abs.(equation.advectionvelocity))) / n_nodes
end


# Convert conservative variables to primitive
function cons2prim(equation::LinearScalarAdvectionEquation, cons::Array{Float64, 4})
  return cons
end

# Convert conservative variables to entropy variables
function cons2entropy(equation::LinearScalarAdvectionEquation,
                      cons::Array{Float64, 4}, n_nodes::Int,
                      n_elements::Int)
  return cons
end


# Calculate entropy for a conservative state `cons`
@inline entropy(cons::Real, ::LinearScalarAdvectionEquation) = cons^2 / 2
@inline entropy(cons, equation::LinearScalarAdvectionEquation) = entropy(cons[1], equation)


# Calculate total energy for a conservative state `cons`
@inline energy_total(cons::Real, ::LinearScalarAdvectionEquation) = cons^2 / 2
@inline energy_total(cons, equation::LinearScalarAdvectionEquation) = energy_total(cons[1], equation)
