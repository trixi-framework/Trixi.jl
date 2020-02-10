# Main data structure for system of equations "linear scalar advection"
struct LinearScalarAdvection <: AbstractSysEqn{1}
  name::String
  initialconditions::String
  sources::String
  varnames_cons::SVector{1, String}
  varnames_prim::SVector{1, String}
  advectionvelocity::Float64

  function LinearScalarAdvection()
    name = "linearscalaradvection"
    initialconditions = parameter("initialconditions")
    sources = parameter("sources", "none")
    varnames_cons = ["scalar"]
    varnames_prim = ["scalar"]
    a = parameter("advectionvelocity")
    new(name, initialconditions, sources, varnames_cons, varnames_prim, a)
  end
end


# Set initial conditions at physical location `x` for time `t`
function initialconditions(s::LinearScalarAdvection, x, t)
  name = s.initialconditions
  if name == "gauss"
    return [exp(-(x - s.advectionvelocity * t)^2)]
  elseif name == "convergence_test"
    c = 1.0
    A = 0.5
    a = 0.3
    L = 2 
    f = 1/L
    omega = 2 * pi * f
    u = a
    p = 1.0
    scalar = c + A * sin(omega * (x - a * t))
    return [scalar]
  elseif name == "constant"
    return [2.0]
  else
    error("Unknown initial condition '$name'")
  end
end


# Apply source terms
function sources(s::LinearScalarAdvection, ut, u, x, cell_id, t, nnodes)
  name = s.sources
  error("Unknown source terms '$name'")
end


# Calculate flux at a given cell id
function calcflux(s::LinearScalarAdvection, u::Array{Float64, 3}, cell_id::Int, nnodes::Int)
  f = zeros(MMatrix{1, nnodes})
  a = s.advectionvelocity

  for i = 1:nnodes
    f[1, i]  = u[1, i, cell_id] * a
  end

  return f
end


# Calculate flux across interface with different states on both sides (Riemann problem)
function riemann!(fsurf, usurf, s, ss::LinearScalarAdvection, nnodes)
  a = ss.advectionvelocity
  fsurf[1, s] = 1/2 * ((a + abs(a)) * usurf[1, 1, s] + (a - abs(a)) * usurf[2, 1, s])
end


# Determine maximum stable time step based on polynomial degree and CFL number
function maxdt(s::LinearScalarAdvection, u::Array{Float64, 3}, cell_id::Int, nnodes::Int,
               invjacobian::Float64, cfl::Float64)
  return cfl * 2 / (invjacobian * s.advectionvelocity) / (2 * (nnodes - 1) + 1)
end


# Convert conservative variables to primitive
function cons2prim(s::LinearScalarAdvection, cons::Array{Float64, 3})
  return cons
end
