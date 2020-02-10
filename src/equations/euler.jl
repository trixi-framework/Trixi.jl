module EulerEquations

using ...Jul1dge
using ..Equations # Use everything to allow method extension via "function Equations.<method>"
using ...Auxiliary: parameter
using StaticArrays: SVector, MVector, MMatrix

# Export all symbols that should be available from Equations
export Euler
export initialconditions
export sources
export calcflux
export riemann!
export maxdt
export cons2prim


# Main data structure for system of equations "Euler"
struct Euler <: AbstractEquation{3}
  name::String
  initialconditions::String
  sources::String
  varnames_cons::SVector{3, String}
  varnames_prim::SVector{3, String}
  gamma::Float64
  riemann_solver::String

  function Euler()
    name = "euler"
    initialconditions = parameter("initialconditions")
    sources = parameter("sources", "none")
    varnames_cons = ["rho", "rho_u", "rho_e"]
    varnames_prim = ["rho", "u", "p"]
    gamma = 1.4
    riemann_solver = parameter("riemann_solver", "hllc", valid=["hllc", "laxfriedrichs"])
    new(name, initialconditions, sources, varnames_cons, varnames_prim, gamma, riemann_solver)
  end
end


# Set initial conditions at physical location `x` for time `t`
function Equations.initialconditions(s::Euler, x, t)
  name = s.initialconditions
  if name == "density_pulse"
    rho = 1 + exp(-x^2)/2
    v = 1
    rho_v = rho * v
    p = 1
    # p = 1 + exp(-x^2)/2
    rho_e = p/(s.gamma - 1) + 1/2 * rho * v^2
    return [rho, rho_v, rho_e] 
  elseif name == "pressure_pulse"
    rho = 1
    v = 1
    rho_v = rho * v
    p = 1 + exp(-x^2)/2
    # p = 1 + exp(-x^2)/2
    rho_e = p/(s.gamma - 1) + 1/2 * rho * v^2
    return [rho, rho_v, rho_e] 
    # return [1.0 + exp(-x^2)/2, 1.0, 1.0] 
  elseif name == "density_pressure_pulse"
    rho = 1 + exp(-x^2)/2
    v = 1
    rho_v = rho * v
    p = 1 + exp(-x^2)/2
    rho_e = p/(s.gamma - 1) + 1/2 * rho * v^2
    return [rho, rho_v, rho_e] 
  elseif name == "constant"
    return [1.0, 0.0, 1.0]
  elseif name == "convergence_test"
    c = 1.0
    A = 0.5
    a = 0.3
    L = 2 
    f = 1/L
    omega = 2 * pi * f
    u = a
    p = 1.0
    rho = c + A * sin(omega * (x - a * t))
    rho_u = rho * u
    rho_e = p/(s.gamma - 1) + 1/2 * rho * u^2
    return [rho, rho_u, rho_e]
  elseif name == "sod"
    if x < 0.0
      return [1.0, 0.0, 2.5]
    else
      return [0.125, 0.0, 0.25]
    end
  else
    error("Unknown initial condition '$name'")
  end
end


# Apply source terms
function Equations.sources(s::Euler, ut, u, x, cell_id, t, nnodes)
  name = s.sources
  error("Unknown source term '$name'")
end


# Calculate flux at a given cell id
function Equations.calcflux(s::Euler, u::Array{Float64, 3}, cell_id::Int, nnodes::Int)
  f = zeros(MMatrix{3, nnodes})
  @inbounds for i = 1:nnodes
    rho   = u[1, i, cell_id]
    rho_v = u[2, i, cell_id]
    rho_e = u[3, i, cell_id]
    f[:, i] .= calcflux(s, rho, rho_v, rho_e)
  end

  return f
end

# Calculate flux for a give state
function Equations.calcflux(s::Euler, rho::Float64, rho_v::Float64, rho_e::Float64)
  f = zeros(MVector{3})
  v = rho_v/rho
  p = (s.gamma - 1) * (rho_e - 1/2 * rho * v^2)

  f[1]  = rho_v
  f[2]  = rho_v * v + p
  f[3]  = (rho_e + p) * v

  return f
end


# Calculate flux across interface with different states on both sides (Riemann problem)
function Equations.riemann!(fsurf, usurf, s::Int, ss::Euler, nnodes)
  u_ll     = usurf[1, :, s]
  u_rr     = usurf[2, :, s]

  rho_ll   = u_ll[1]
  rho_v_ll = u_ll[2]
  rho_e_ll = u_ll[3]
  rho_rr   = u_rr[1]
  rho_v_rr = u_rr[2]
  rho_e_rr = u_rr[3]

  v_ll = rho_v_ll / rho_ll
  p_ll = (ss.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * v_ll^2)
  c_ll = sqrt(ss.gamma * p_ll / rho_ll)
  v_rr = rho_v_rr / rho_rr
  p_rr = (ss.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * v_rr^2)
  c_rr = sqrt(ss.gamma * p_rr / rho_rr)

  f_ll = calcflux(ss, rho_ll, rho_v_ll, rho_e_ll)
  f_rr = calcflux(ss, rho_rr, rho_v_rr, rho_e_rr)

  if ss.riemann_solver == "laxfriedrichs"
    λ_max = max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr)

    @. fsurf[:, s] = 1/2 * (f_ll + f_rr) - 1/2 * λ_max * (u_rr - u_ll)
  elseif ss.riemann_solver == "hllc"
    v_tilde = (sqrt(rho_ll) * v_ll + sqrt(rho_rr) * v_rr) / (sqrt(rho_ll) + sqrt(rho_rr))
    h_ll = (rho_e_ll + p_ll) / rho_ll
    h_rr = (rho_e_rr + p_rr) / rho_rr
    h_tilde = (sqrt(rho_ll) * h_ll + sqrt(rho_rr) * h_rr) / (sqrt(rho_ll) + sqrt(rho_rr))
    c_tilde = sqrt((ss.gamma - 1) * (h_tilde - 1/2 * v_tilde^2))
    s_ll = v_tilde - c_tilde
    s_rr = v_tilde + c_tilde

    if s_ll > 0
      @. fsurf[:, s] = f_ll
    elseif s_rr < 0
      @. fsurf[:, s] = f_rr
    else
      s_star = ((p_rr - p_ll + rho_ll * v_ll * (s_ll - v_ll) - rho_rr * v_rr * (s_rr - v_rr))
                / (rho_ll * (s_ll - v_ll) - rho_rr * (s_rr - v_rr)))
      if s_ll <= 0 && 0 <= s_star
        u_star_ll = rho_ll * (s_ll - v_ll)/(s_ll - s_star) .* (
            [1, s_star,
             rho_e_ll/rho_ll + (s_star - v_ll) * (s_star + rho_ll/(rho_ll * (s_ll - v_ll)))])
        @. fsurf[:, s] = f_ll + s_ll * (u_star_ll - u_ll)
      else
        u_star_rr = rho_rr * (s_rr - v_rr)/(s_rr - s_star) .* (
            [1, s_star,
             rho_e_rr/rho_rr + (s_star - v_rr) * (s_star + rho_rr/(rho_rr * (s_rr - v_rr)))])
        @. fsurf[:, s] = f_rr + s_rr * (u_star_rr - u_rr)
      end
    end
  else
    error("unknown Riemann solver '$(s.riemann_solver)'")
  end
end


# Determine maximum stable time step based on polynomial degree and CFL number
function Equations.maxdt(s::Euler, u::Array{Float64, 3}, cell_id::Int,
                         nnodes::Int, invjacobian::Float64, cfl::Float64)
  λ_max = 0.0
  for i = 1:nnodes
    rho   = u[1, i, cell_id]
    rho_v = u[2, i, cell_id]
    rho_e = u[3, i, cell_id]
    v = rho_v/rho
    p = (s.gamma - 1) * (rho_e - 1/2 * rho * v^2)
    c = sqrt(s.gamma * p / rho)
    λ_max = max(λ_max, abs(v) + c)
  end

  dt = cfl * 2 / (invjacobian * λ_max) / (2 * (nnodes - 1) + 1)

  return dt
end


# Convert conservative variables to primitive
function Equations.cons2prim(s::Euler, cons::Array{Float64, 3})
  prim = similar(cons)
  @. prim[1, :, :] = cons[1, :, :]
  @. prim[2, :, :] = cons[2, :, :] / cons[1, :, :]
  @. prim[3, :, :] = (s.gamma - 1) * (cons[3, :, :] - 1/2 * cons[2, :, :] * prim[2, :, :])
  return prim
end

end # module
