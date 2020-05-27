
@doc raw"""
    LinearScalarAdvectionEquation

The linear scalar advection equation
```math
\partial_t u + a_1 \partial_1 u + a_2 \partial_2 u = 0
```
in two space dimensions with constant velocity `a`.
"""
struct LinearScalarAdvectionEquation <: AbstractEquation{1}
  initial_conditions::String
  sources::String
  advectionvelocity::SVector{2, Float64}
end

function LinearScalarAdvectionEquation()
  initial_conditions = parameter("initial_conditions")
  sources = parameter("sources", "none")
  a = convert(SVector{2,Float64}, parameter("advectionvelocity"))
  LinearScalarAdvectionEquation(initial_conditions, sources, a)
end


get_name(::LinearScalarAdvectionEquation) = "LinearScalarAdvection"
varnames_cons(::LinearScalarAdvectionEquation) = SVector("scalar")
varnames_prim(::LinearScalarAdvectionEquation) = SVector("scalar")


# Set initial conditions at physical location `x` for time `t`
function initial_conditions(equation::LinearScalarAdvectionEquation, x, t)
  name = equation.initial_conditions

  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advectionvelocity * t

  if name == "gauss"
    return [exp(-(x_trans[1]^2 + x_trans[2]^2))]
  elseif name == "convergence_test"
    c = 1.0
    A = 0.5
    L = 2
    f = 1/L
    omega = 2 * pi * f
    scalar = c + A * sin(omega * sum(x_trans))
    return [scalar]
  elseif name == "sin-sin"
    scalar = sin(2 * pi * x_trans[1]) * sin(2 * pi * x_trans[2])
    return [scalar]
  elseif name == "constant"
    return [2.0]
  elseif name == "linear-x-y"
    return [sum(x_trans)]
  elseif name == "linear-x"
    return [x_trans[1]]
  elseif name == "linear-y"
    return [x_trans[2]]
  else
    error("Unknown initial condition '$name'")
  end
end


# Apply source terms
function sources(equation::LinearScalarAdvectionEquation, ut, u, x, element_id, t, n_nodes)
  name = equation.sources
  error("Unknown source terms '$name'")
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
  nothing
end


# Calculate flux across interface with different states on both sides (surface version)
function riemann!(destination, surface_flux, u_surfaces, surface_id,
                  equation::LinearScalarAdvectionEquation, n_nodes::Int,
                  orientations::Vector{Int})
  for i = 1:n_nodes
    flux = surface_flux(equation, orientations[surface_id],
                        u_surfaces[1, 1, i, surface_id],
                        u_surfaces[2, 1, i, surface_id])

    # Copy flux back to actual flux array
    for v in 1:nvariables(equation)
      destination[v, i] = flux[v]
    end
  end
end


function lax_friedrichs_flux(equation::LinearScalarAdvectionEquation, orientation, u_ll, u_rr)
  a = equation.advectionvelocity[orientation]
  0.5 * (a + abs(a)) * u_ll + (a - abs(a)) * u_rr
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

# Convert conservative variables to entropy
function cons2entropy(equation::LinearScalarAdvectionEquation,
                      cons::Array{Float64, 4}, n_nodes::Int,
                      n_elements::Int)
  return cons
end
