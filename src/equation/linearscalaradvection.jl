struct LinearScalarAdvection <: AbstractSysEqn{1}
  name::String
  initialconditions::String
  sources::String
  varnames::SVector{1, String}
  advectionvelocity::Float64

  function LinearScalarAdvection(initialconditions, sources, a)
    name = "linearscalaradvection"
    varnames = ["scalar"]
    new(name, initialconditions, sources, varnames, a)
  end
end


function initialconditions(s::LinearScalarAdvection, x, t)
  name = s.initialconditions
  if name == "gauss"
    return [exp(-(x - s.advectionvelocity * t)^2)]
  elseif name == "constant"
    return [2.0]
  else
    error("Unknown initial condition '$name'")
  end
end


function sources(s::LinearScalarAdvection, ut, u, x, cell_id, t, nnodes)
  name = s.sources
  error("Unknown source terms '$name'")
end


function calcflux(s::LinearScalarAdvection, u, cell_id, nnodes)
  f = zeros(MMatrix{1, nnodes})
  a = s.advectionvelocity

  for i = 1:nnodes
    f[1, i]  = u[1, i, cell_id] * a
  end

  return f
end


function riemann!(fsurf, usurf, s, ss::LinearScalarAdvection, nnodes)
  a = ss.advectionvelocity
  fsurf[1, s] = 1/2 * ((a + abs(a)) * usurf[1, 1, s] + (a - abs(a)) * usurf[2, 1, s])
end


function maxdt(s::LinearScalarAdvection, u::Array{Float64, 3}, cell_id::Int, nnodes::Int,
               invjacobian::Float64, cfl::Float64)
  return cfl * 2 / (invjacobian * s.advectionvelocity) / (2 * (nnodes - 1) + 1)
end

