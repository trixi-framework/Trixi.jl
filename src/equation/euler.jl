struct Euler <: AbstractSysEqn{3}
  name::String
  initialconditions::String
  sources::String
  varnames::SVector{3, String}
  gamma::Float64

  function Euler(initialconditions, sources)
    name = "euler"
    varnames = ["rho", "rho_u", "rho_e"]
    gamma = 1.4
    new(name, initialconditions, sources, varnames, gamma)
  end
end


function initialconditions(s::Euler, x, t)
  name = s.initialconditions
  if name == "gauss"
    return [1.0 + exp(-x^2)/2, 1.0, 1.0] 
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


function sources(s::Euler, ut, u, x, cell_id, t, nnodes)
  name = s.sources
  error("Unknown source term '$name'")
end


function calcflux(s::Euler, u, cell_id::Int, nnodes::Int)
  f = zeros(MMatrix{3, nnodes})
  for i = 1:nnodes
    rho   = u[1, i, cell_id]
    rho_v = u[2, i, cell_id]
    rho_e = u[3, i, cell_id]
    f[:, i] .= calcflux(s, rho, rho_v, rho_e)
  end

  return f
end

function calcflux(s::Euler, rho::Float64, rho_v::Float64, rho_e::Float64)
  f = zeros(MVector{3})
  v = rho_v/rho
  p = (s.gamma - 1) * (rho_e - 1/2 * rho * v^2)

  f[1]  = rho_v
  f[2]  = rho_v * v + p
  f[3]  = (rho_e + p) * v

  return f
end

function riemann!(fsurf, usurf, s, ss::Euler, nnodes)
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
  λ_max = max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr)

  @. fsurf[:, s] = 1/2 * (f_ll + f_rr) - 1/2 * λ_max * (u_rr - u_ll)
end

function maxdt(s::Euler, u::Array{Float64, 3}, cell_id::Int, nnodes::Int,
               invjacobian::Float64, cfl::Float64)
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

