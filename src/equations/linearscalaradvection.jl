module LinearScalarAdvectionEquations

using ...Trixi
using ..Equations # Use everything to allow method extension via "function <parent_module>.<method>"
using ...Auxiliary: parameter
using StaticArrays: SVector, MVector, MMatrix

# Export all symbols that should be available from Equations
export LinearScalarAdvection
export initial_conditions
export sources
export calcflux!
export riemann!
export calc_max_dt
export cons2entropy
export cons2prim


# Main data structure for system of equations "linear scalar advection"
struct LinearScalarAdvection{SurfaceFlux} <: AbstractEquation{1}
  name::String
  initial_conditions::String
  sources::String
  varnames_cons::SVector{1, String}
  varnames_prim::SVector{1, String}
  advectionvelocity::SVector{2, Float64}
  surface_flux::SurfaceFlux
  volume_flux_type::Symbol
  have_nonconservative_terms::Bool
end

function LinearScalarAdvection()
  name = "linearscalaradvection"
  initial_conditions = parameter("initial_conditions")
  sources = parameter("sources", "none")
  varnames_cons = SVector("scalar")
  varnames_prim = SVector("scalar")
  a = convert(SVector{2,Float64}, parameter("advectionvelocity"))
  surface_flux_type = Symbol(parameter("surface_flux", "lax_friedrichs_flux",
                                       valid=["lax_friedrichs_flux", "central_flux"]))
  surface_flux = eval(surface_flux_type)
  volume_flux_type = Symbol(parameter("volume_flux_type", "central_flux",
                                      valid=["central_flux"]))
  have_nonconservative_terms = false
  LinearScalarAdvection(name, initial_conditions, sources, varnames_cons, varnames_prim, a, surface_flux, volume_flux_type,
                        have_nonconservative_terms)
end


# Set initial conditions at physical location `x` for time `t`
function Equations.initial_conditions(equation::LinearScalarAdvection,
                                      x::AbstractArray{Float64}, t::Real)
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
function Equations.sources(equation::LinearScalarAdvection, ut, u, x, element_id, t, n_nodes)
  name = equation.sources
  error("Unknown source terms '$name'")
end


# Calculate 2D flux (element version)
@inline function Equations.calcflux!(f1::AbstractArray{Float64},
                                     f2::AbstractArray{Float64},
                                     equation::LinearScalarAdvection,
                                     u::AbstractArray{Float64}, element_id::Int,
                                     n_nodes::Int)
  for j = 1:n_nodes
    for i = 1:n_nodes
      @views calcflux!(f1[:, i, j], f2[:, i, j], equation, u[:, i, j, element_id])
    end
  end
end


# Calculate 2D flux (pointwise version)
@inline function Equations.calcflux!(f1::AbstractArray{Float64},
                                     f2::AbstractArray{Float64},
                                     equation::LinearScalarAdvection,
                                     u::AbstractArray{Float64})
  f1[1] = u[1] * equation.advectionvelocity[1]
  f2[1] = u[1] * equation.advectionvelocity[2]
end


# Calculate flux across interface with different states on both sides (surface version)
function Equations.riemann!(surface_flux::AbstractMatrix{Float64}, ::AbstractVector{Float64},
                            u_surfaces::AbstractArray{Float64, 4}, surface_id::Int,
                            equation::LinearScalarAdvection, n_nodes::Int,
                            orientations::Vector{Int})
  for i = 1:n_nodes
    @views riemann!(surface_flux[:, i], u_surfaces[:, :, i, surface_id],
                    equation, orientations[surface_id])
  end
end


# Calculate flux across interface with different states on both sides (pointwise version)
function Equations.riemann!(surface_flux::AbstractArray{Float64},
                            u_surfaces::AbstractArray{Float64},
                            equation::LinearScalarAdvection, orientation::Int)
  surface_flux[1] = equation.surface_flux(equation, orientation, u_surfaces[1, 1], u_surfaces[2, 1])
  return nothing
end


function lax_friedrichs_flux(equation::LinearScalarAdvection, orientation, u_ll, u_rr)
  a = equation.advectionvelocity[orientation]
  0.5 * (a + abs(a)) * u_ll + (a - abs(a)) * u_rr
end


# Determine maximum stable time step based on polynomial degree and CFL number
function Equations.calc_max_dt(equation::LinearScalarAdvection,
                               u::Array{Float64, 4}, element_id::Int,
                               n_nodes::Int, invjacobian::Float64,
                               cfl::Float64)
  return cfl * 2 / (invjacobian * maximum(abs.(equation.advectionvelocity))) / n_nodes
end


# Convert conservative variables to primitive
function Equations.cons2prim(equation::LinearScalarAdvection, cons::Array{Float64, 4})
  return cons
end

# Convert conservative variables to entropy
function Equations.cons2entropy(equation::LinearScalarAdvection,
                                cons::Array{Float64, 4}, n_nodes::Int,
                                n_elements::Int)
  return cons
end

end # module
