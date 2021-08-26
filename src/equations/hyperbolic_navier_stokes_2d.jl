# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


@doc raw"""
    HyperbolicNavierStokesEquations2D

The hyperbolic Navier-Stokes equations in one space dimension.
A description of this system can be found in the book
- Masatsuka (2013)
  I Do Like CFD, Too: Vol 1.
  Freely available at [http://www.cfdbooks.com/](http://www.cfdbooks.com/)
Further analysis can be found in the paper
- Nishikawa (2011)
  New-Generation Hyperbolic Navier-Stokes Schemes: O(1/h) Speed-Up and Accurate
  Viscous/Heat Fluxes
  20th AIAA Computational Fluid Dynamics Conference
"""
struct HyperbolicNavierStokesEquations2D{RealT<:Real} <: AbstractHyperbolicNavierStokesEquations{2, 9}
  gamma::RealT  # ratio of specific heats
  Pr::RealT     # Prandtl number
  L::RealT      # length scale
  Tinf::RealT   # free stream temperature
  C::RealT      # Sutherland constant
  Minf::RealT   # Mach number
  Reinf::RealT  # Reynolds number
end

function HyperbolicNavierStokesEquations2D(;gamma=1.4, Pr=0.75, L=inv(sqrt(2pi)), Tinf = 400.0, C = 110.5, Minf = 3.5, Reinf = 25.0)
  HyperbolicNavierStokesEquations2D(promote(gamma, Pr, L, Tinf, C, Minf, Reinf)...)
end


varnames(::typeof(cons2cons), ::HyperbolicNavierStokesEquations2D) = ("rho", "rho_v1", "rho_v2", "rho_E", "tau_xx", "tau_xy", "tau_yy", "q_x", "q_y")
varnames(::typeof(cons2prim), ::HyperbolicNavierStokesEquations2D) = ("rho", "v1", "v2", "E", "p", "T", "tau_xx", "tau_xy", "tau_yy")
default_analysis_errors(::HyperbolicNavierStokesEquations2D) = (:l2_error, :linf_error, :residual)

@inline function residual_steady_state(du, ::HyperbolicNavierStokesEquations2D)
  abs(du[1])
end

"""
    initial_condition_constructed_lin(x, t, equations::HyperbolicNavierStokesEquations2D)

A non-priodic smooth initial condition.
Can be used in combination with [`source_terms_constructed_lin`](@ref) and BoundaryConditionDirichlet.
The primal variables density ρ and specific total energy E follow linear functions.
"""
@inline function initial_condition_constructed_lin(x, t, equations::HyperbolicNavierStokesEquations2D)
  @unpack gamma, Pr, L, Tinf, C, Minf, Reinf = equations

  rho = x[1] + x[2]
  v1 = sqrt(x[1])
  v2 = sqrt(x[2])
  E = rho
  p = (gamma - 1)/2 * rho^2
  T = gamma * (gamma-1)/2 * rho
  mu = Minf/Reinf * (1+C/Tinf)/(T+C/Tinf) * T^(1.5)
  tau_xx = 2/3 * mu * (1/v1 - 0.5/v2)
  tau_xy = zero(rho)
  tau_yy = 2/3 * mu * (1/v2 - 0.5/v1)
  q_x = -mu*gamma/Pr * 0.5
  q_y = q_x

  if false #iszero(t)
    delta = 0.01
    rho *= (1+delta)
    v1 *=(1-delta)
    v2 *=(1-delta)
    E *= (1+delta)
    tau_xx *= (1+delta/2)
    tau_xy *= (1+delta/2)
    tau_yy *= (1+delta/2)
    q_x *= (1-delta/2)
    q_y *= (1-delta/2)
  end

  return SVector(rho, rho*v1, rho*v2, rho*E, tau_xx, tau_xy, tau_yy, q_x, q_y)
end

"""
    source_terms_constructed_lin(u, x, t, equations::HyperbolicNavierStokesEquations2D)

Source terms that include the forcing function `f(x)` and right hand side for the hyperbolic
Navier-Stokes system that is used with [`initial_condition_constructed_lin](@ref) and
BoundaryConditionDirichlet.
"""
@inline function source_terms_constructed_lin(u, x, t, equations::HyperbolicNavierStokesEquations2D)
  u1, u2, u3, u4, u5 = u
  @unpack gamma, Pr, L, Tinf, C, Minf, Reinf = equations

  rho = x[1] + x[2]
  v1 = sqrt(x[1])
  v2 = sqrt(x[2])
  E = rho
  p = (gamma - 1)/2 * rho^2
  pdx = (gamma - 1) * rho
  pdy = pdx
  T = gamma * (gamma-1)/2 * rho
  Tdx = gamma * (gamma-1)/2
  Tdy = Tdx
  mu = Minf/Reinf * (1+C/Tinf)/(T+C/Tinf) * T^(1.5)
  mudx = mu * (1.5/T - 1/(T+C/Tinf)) * Tdx
  mudy = mudx
  tau_xx = 2/3 * mu * (1/v1 - 0.5/v2)
  tau_xxdx = 1/3 * (mudx*(2/v1 - 1/v2) - x[1]^(-1.5)*mu)
  tau_xy = zero(rho)
  tau_yy = 2/3 * mu * (1/v2 - 0.5/v1)
  tau_yydy = 1/3 * (mudy*(2/v2 - 1/v1) - x[2]^(-1.5)*mu)
  q_x = -mu*gamma/Pr * 0.5
  q_xdx = -mudx*gamma/Pr * 0.5
  q_y = q_x
  q_ydy = -mudy*gamma/Pr * 0.5

  mu_v = 4/3 * mu     # viscosity of stress
  mu_h = gamma*mu/Pr  # viscosity of heat flux

  dq1 = 1.5 * (v1+v2) + 0.5 * (x[2]/v1 + x[1]/v2)
  dq2 = 2*x[1]+x[2] + pdx - tau_xxdx + v1 * (v2 + 0.5*rho/v2)
  dq3 = 2*x[2]+x[1] + pdy - tau_yydy + v2 * (v1 + 0.5*rho/v1)
  dq4 = ((gamma+1) * rho * (v1+v2 + 0.25*(rho/v1 + rho/v2))
        - tau_xxdx*v1 - tau_yydy*v2 - 0.5*(tau_xx/v1 + tau_yy/v2) + q_xdx + q_ydy)
  dq5 = -tau_xx / mu_v
  dq6 = -tau_xy / mu_v
  dq7 = -tau_yy / mu_v
  dq8 = -q_x / mu_h
  dq9 = -q_y / mu_h

  return SVector(dq1, dq2, dq3, dq4, dq5, dq6, dq7, dq8, dq9)
end

"""
    initial_condition_constructed_square(x, t, equations::HyperbolicNavierStokesEquations2D)

A non-priodic smooth initial condition.
Can be used in combination with [`source_terms_constructed_square`](@ref) and BoundaryConditionDirichlet.
The primal variables density ρ and specific total energy E follow second order polynomials.
"""
@inline function initial_condition_constructed_square(x, t, equations::HyperbolicNavierStokesEquations2D)
  @unpack gamma, Pr, L, Tinf, C, Minf, Reinf = equations

  rho = x[1]^2 + x[2]^2
  v1 = x[1]
  v2 = x[2]
  E = rho
  p = (gamma - 1)/2 * rho^2
  T = gamma * p / rho
  mu = Minf/Reinf * (1+C/Tinf)/(T+C/Tinf) * T^(1.5)
  tau_xx = 2/3 * mu
  tau_xy = zero(rho)
  tau_yy = tau_xx
  q_x = -mu*gamma/Pr * v1
  q_y = -mu*gamma/Pr * v2

  if false #iszero(t)
    delta = 0.01
    rho *= (1+delta)
    v1 *=(1-delta)
    v2 *=(1-delta)
    E *= (1+delta)
    tau_xx *= (1+delta/2)
    tau_xy *= (1+delta/2)
    tau_yy *= (1+delta/2)
    q_x *= (1-delta/2)
    q_y *= (1-delta/2)
  end

  return SVector(rho, rho*v1, rho*v2, rho*E, tau_xx, tau_xy, tau_yy, q_x, q_y)
end

"""
    source_terms_constructed_square(u, x, t, equations::HyperbolicNavierStokesEquations2D)

Source terms that include the forcing function `f(x)` and right hand side for the hyperbolic
Navier-Stokes system that is used with [`initial_condition_constructed_square](@ref) and
BoundaryConditionDirichlet.
"""
@inline function source_terms_constructed_square(u, x, t, equations::HyperbolicNavierStokesEquations2D)
  @unpack gamma, Pr, L, Tinf, C, Minf, Reinf = equations

  rho = x[1]^2 + x[2]^2
  v1 = x[1]
  v2 = x[2]
  E = rho
  p = (gamma - 1)/2 * rho^2
  pdx = (gamma - 1) * rho * 2*v1
  pdy = (gamma - 1) * rho * 2*v2
  T = gamma * p / rho
  Tdx = gamma * (gamma-1) * v1
  Tdy = gamma * (gamma-1) * v2
  mu = Minf/Reinf * (1+C/Tinf)/(T+C/Tinf) * T^(1.5)
  mudx = mu * (1.5/T - 1/(T+C/Tinf)) * Tdx
  mudy = mu * (1.5/T - 1/(T+C/Tinf)) * Tdy
  tau_xx = 2/3 * mu
  tau_xxdx = 2/3 * mudx
  tau_xy = zero(rho)
  tau_yy = tau_xx
  tau_yydy = 2/3 * mudy
  q_x = -mu*gamma/Pr * v1
  q_xdx = -gamma/Pr * (mudx*v1 + mu)
  q_y = -mu*gamma/Pr * v2
  q_ydy = -gamma/Pr * (mudy*v2 + mu)

  mu_v = 4/3 * mu     # viscosity of stress
  mu_h = gamma*mu/Pr  # viscosity of heat flux

  dq1 = 4*rho
  dq2 = 5*rho*v1 + pdx - tau_xxdx
  dq3 = 5*rho*v2 + pdy - tau_yydy
  dq4 = (gamma+1) * 3*rho^2 - tau_xx - tau_yy - tau_xxdx*v1 - tau_yydy*v2 + q_xdx + q_ydy
  dq5 = -tau_xx / mu_v
  dq6 = -tau_xy / mu_v
  dq7 = -tau_yy / mu_v
  dq8 = -q_x / mu_h
  dq9 = -q_y / mu_h

  return SVector(dq1, dq2, dq3, dq4, dq5, dq6, dq7, dq8, dq9)
end

"""
    initial_condition_constructed_exp(x, t, equations::HyperbolicNavierStokesEquations2D)

A non-priodic smooth initial condition. In the initial guess the primal variables
density ρ, velocity v and specific total energy E follow linear functions.
Can be used in combination with [`source_terms_constructed_exp`](@ref) and BoundaryConditionDirichlet.
The primal variables follow exponential functions.
"""
@inline function initial_condition_constructed_exp(x, t, equations::HyperbolicNavierStokesEquations2D)
  @unpack gamma, Pr, L, Tinf, C, Minf, Reinf = equations

  if iszero(t)
    rho = (exp(1.0) - 1.0)*x[2] + (exp(1.0) - 1.0)*x[1] + 2.0
    v1 = (exp(0.5) - 1.0)*x[1] + 1.0
    v2 = (exp(0.5) - 1.0)*x[2] + 1.0
    E = rho
    T = gamma * (gamma-1) * rho/2
    mu = Minf/Reinf * (1+C/Tinf)/(T+C/Tinf) * T^(1.5)
    tau_xx = 2/3 * mu * (v1 - v2/2)
    tau_xy = zero(rho)
    tau_yy = 2/3 * mu * (v2 - v1/2)
    q_x = -mu*gamma/Pr * exp(x[1])/2
    q_y = -mu*gamma/Pr * exp(x[2])/2

  else

    rho = exp(x[1]) + exp(x[2])
    v1 = exp(0.5*x[1])
    v2 = exp(0.5*x[2])
    E = rho
    p = (gamma - 1)/2 * rho^2
    T = gamma * (gamma-1)/2 * rho
    mu = Minf/Reinf * (1+C/Tinf)/(T+C/Tinf) * T^(1.5)
    tau_xx = 2/3 * mu * (v1 - v2/2)
    tau_xy = zero(rho)
    tau_yy = 2/3 * mu * (v2 - v1/2)
    q_x = -mu*gamma/Pr * exp(x[1])/2
    q_y = -mu*gamma/Pr * exp(x[2])/2

  end

  return SVector(rho, rho*v1, rho*v2, rho*E, tau_xx, tau_xy, tau_yy, q_x, q_y)
end

"""
    source_terms_constructed_exp(u, x, t, equations::HyperbolicNavierStokesEquations2D)

Source terms that include the forcing function `f(x)` and right hand side for the hyperbolic
Navier-Stokes system that is used with [`initial_condition_constructed_exp](@ref) and
BoundaryConditionDirichlet.
"""
@inline function source_terms_constructed_exp(u, x, t, equations::HyperbolicNavierStokesEquations2D)
  @unpack gamma, Pr, L, Tinf, C, Minf, Reinf = equations

  rho = exp(x[1]) + exp(x[2])
  v1 = exp(0.5*x[1])
  v2 = exp(0.5*x[2])
  E = rho
  p = (gamma - 1)/2 * rho^2
  pdx = (gamma - 1) * exp(x[1]) * rho
  pdy = (gamma - 1) * exp(x[2]) * rho
  T = gamma * (gamma-1)/2 * rho
  Tdx = gamma * (gamma-1)/2 * exp(x[1])
  Tdy = gamma * (gamma-1)/2 * exp(x[2])
  mu = Minf/Reinf * (1+C/Tinf)/(T+C/Tinf) * T^(1.5)
  mudx = mu * (1.5/T - 1/(T+C/Tinf)) * Tdx
  mudy = mu * (1.5/T - 1/(T+C/Tinf)) * Tdy
  tau_xx = 2/3 * mu * (v1 - v2/2)
  tau_xxdx = 1/3 * (mu * v1 + mudx * (2*v1 - v2))
  tau_xy = zero(rho)
  tau_yy = 2/3 * mu * (v2 - v1/2)
  tau_yydy = 1/3 * (mu * v2 + mudy * (2*v2 - v1))
  q_x = -mu*gamma/Pr * exp(x[1])/2
  q_xdx = -gamma/Pr * exp(x[1])/2 * (mu + mudx)
  q_y = -mu*gamma/Pr * exp(x[2])/2
  q_ydy = -gamma/Pr * exp(x[2])/2 * (mu + mudy)

  mu_v = 4/3 * mu     # viscosity of stress
  mu_h = gamma*mu/Pr  # viscosity of heat flux

  dq1 = v1 * (rho/2 + exp(x[1])) + v2*(rho/2 + exp(x[2]))
  dq2 = exp(x[1])*rho + exp(2*x[1]) + pdx - tau_xxdx + v1*v2*(rho/2 + exp(x[2]))
  dq3 = exp(x[2])*rho + exp(2*x[2]) + pdy - tau_yydy + v1*v2*(rho/2 + exp(x[1]))
  dq4 = ((gamma+1)*rho * (v1*(rho/4 + exp(x[1])) + v2*(rho/4 + exp(x[2])))
        - v1*(tau_xx/2 + tau_xxdx) - v2*(tau_yy/2 + tau_yydy) + q_xdx + q_ydy)
  dq5 = -tau_xx / mu_v
  dq6 = -tau_xy / mu_v
  dq7 = -tau_yy / mu_v
  dq8 = -q_x / mu_h
  dq9 = -q_y / mu_h

  return SVector(dq1, dq2, dq3, dq4, dq5, dq6, dq7, dq8, dq9)
end

@inline function initial_condition_constructed_periodic(x, t, equations::HyperbolicNavierStokesEquations2D)
  @unpack gamma, Pr, L, Tinf, C, Minf, Reinf = equations

  sinx, cosx = sincos(x[1])
  siny, cosy = sincos(x[2])

  rho = (sinx + 2.0)^2 + (siny + 2.0)^2
  v1 = sinx + 2.0
  v2 = siny + 2.0
  E = rho
  p = (gamma - 1)/2 * rho^2
  T = gamma * p /  rho
  mu = Minf/Reinf * (1+C/Tinf)/(T+C/Tinf) * T^(1.5)
  tau_xx = 2/3 * mu * (2*cosx - cosy)
  tau_xy = zero(rho)
  tau_yy = 2/3 * mu * (2*cosy - cosx)
  q_x = -mu*gamma/Pr * v1 * cosx
  q_y = -mu*gamma/Pr * v2 * cosy

  if false #iszero(t)
    delta = 0.01
    rho *= (1+delta)
    v1 *=(1-delta)
    v2 *=(1-delta)
    E *= (1+delta)
    tau_xx *= (1+delta/2)
    tau_xy *= (1+delta/2)
    tau_yy *= (1+delta/2)
    q_x *= (1-delta/2)
    q_y *= (1-delta/2)
  end

  return SVector(rho, rho*v1, rho*v2, rho*E, tau_xx, tau_xy, tau_yy, q_x, q_y)
end

@inline function source_terms_constructed_periodic(u, x, t, equations::HyperbolicNavierStokesEquations2D)
  @unpack gamma, Pr, L, Tinf, C, Minf, Reinf = equations

  sinx, cosx = sincos(x[1])
  siny, cosy = sincos(x[2])

  rho = (sinx + 2.0)^2 + (siny + 2.0)^2
  v1 = sinx + 2.0
  v2 = siny + 2.0
  E = rho
  p = (gamma - 1)/2 * rho^2
  pdx = (gamma - 1) * rho * 2*v1*cosx
  pdy = (gamma - 1) * rho * 2*v2*cosy
  T = gamma * p /  rho
  Tdx = gamma * (gamma-1) * v1*cosx
  Tdy = gamma * (gamma-1) * v2*cosy
  mu = Minf/Reinf * (1+C/Tinf)/(T+C/Tinf) * T^(1.5)
  mudx = mu * (1.5/T - 1/(T+C/Tinf)) * Tdx
  mudy = mu * (1.5/T - 1/(T+C/Tinf)) * Tdy
  tau_xx = 2/3 * mu * (2*cosx - cosy)
  tau_xxdx = 2/3 * (mudx * (2*cosx - cosy) - 2*mu*sinx)
  tau_xy = zero(rho)
  tau_yy = 2/3 * mu * (2*cosy - cosx)
  tau_yydy = 2/3 * (mudy * (2*cosy - cosx) - 2*mu*siny)
  q_x = -mu*gamma/Pr * v1 * cosx
  q_xdx = -gamma/Pr * (mu * (cosx^2 - v1*sinx) + mudx*v1*cosx)
  q_y = -mu*gamma/Pr * v2 * cosy
  q_ydy = -gamma/Pr * (mu * (cosy^2 - v2*siny) + mudy*v2*cosy)

  mu_v = 4/3 * mu     # viscosity of stress
  mu_h = gamma*mu/Pr  # viscosity of heat flux

  dq1 = cosx*(2*v1^2 + rho) + cosy*(2*v2^2 + rho)
  dq2 = 2*cosx*v1*(v1^2 + rho) + pdx - tau_xxdx + cosy * v1 * (2*v2^2 + rho)
  dq3 = 2*cosy*v2*(v2^2 + rho) + pdy - tau_yydy + cosx * v2 * (2*v1^2 + rho)
  dq4 = ((gamma+1)/2 * rho * (cosx * (4*v1^2 + rho) + cosy * (4*v2^2 + rho))
        - (tau_xx*cosx + tau_xxdx*v1) - (tau_yy*cosy + tau_yydy*v2) + q_xdx + q_ydy)
  dq5 = -tau_xx / mu_v
  dq6 = -tau_xy / mu_v
  dq7 = -tau_yy / mu_v
  dq8 = -q_x / mu_h
  dq9 = -q_y / mu_h

  return SVector(dq1, dq2, dq3, dq4, dq5, dq6, dq7, dq8, dq9)
end

@inline function source_terms_harmonic(u, x, t, equations::HyperbolicNavierStokesEquations2D)
  rho, rho_v1, rho_v2, rho_e, tau_xx, tau_xy, tau_yy, q_x, q_y = u
  @unpack gamma, Pr, L, Tinf, C, Minf, Reinf = equations
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  p = (gamma - 1) * (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2))
  T = gamma * p / rho
  mu = Minf / Reinf * (1 + C / Tinf) / (T + C / Tinf) * T^(1.5)
  mu_v = 4/3 * mu     # viscosity of stress
  mu_h = gamma * mu / Pr  # viscosity of heat flux

  dq5 = -tau_xx / mu_v
  du6 = -tau_xy / mu_v
  du7 = -tau_yy / mu_v
  dq8 = -q_x / mu_h
  dq9 = -q_y / mu_h

  return SVector(zero(dq5), zero(dq5), zero(dq5), zero(dq4), dq5, dq6, dq7, dq8, dq9)
end

# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer, equations::HyperbolicNavierStokesEquations2D)
  rho, rho_v1, rho_v2, rho_e, tau_xx, tau_xy, tau_yy, q_x, q_y = u
  @unpack gamma, Pr = equations
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  p = (gamma - 1) * (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2))
  T = gamma * p / rho

  if orientation == 1
    f1 = rho_v1
    f2 = rho_v1 * v1 + p - tau_xx
    f3 = rho_v1 * v2 - tau_xy
    f4 = v1 * (rho_e + p) - tau_xx * v1 - tau_xy * v2 + q_x
    f5 = -v1
    f6 = -0.75 * v2
    f7 = 0.5* v1
    f8 = T / (gamma * (gamma - 1))
    f9 = zero(rho)
  else
    f1 = rho_v2
    f2 = rho_v1 * v2 - tau_xy
    f3 = rho_v2 * v2 + p - tau_yy
    f4 = v2 * (rho_e + p) - tau_yy * v2 - tau_xy * v1 + q_y
    f5 = 0.5 * v2
    f6 = -0.75 * v1
    f7 = -v2
    f8 = zero(rho)
    f9 = T / (gamma * (gamma - 1))
  end

  return SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
end


@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::HyperbolicNavierStokesEquations2D)
  @unpack gamma, Pr, L, Tinf, C, Minf, Reinf = equations
  u1_ll, u2_ll, u3_ll, u4_ll, u5_ll, u6_ll, u7_ll, u8_ll, u9_ll = u_ll
  u1_rr, u2_rr, u3_rr, u4_rr, u5_rr, u6_rr, u7_rr, u8_rr, u9_rr = u_rr
  v1_ll = u2_ll / u1_ll
  v2_ll = u3_ll / u1_ll
  v1_rr = u2_rr / u1_rr
  v2_rr = u3_rr / u1_rr
  p_ll = (gamma-1)*(u4_ll-0.5*u1_ll*(v1_ll^2+v2_ll^2))
  p_rr = (gamma-1)*(u4_rr-0.5*u1_rr*(v1_rr^2+v2_rr^2))
  T_ll = abs(gamma*p_ll/u1_ll)
  T_rr = abs(gamma*p_rr/u1_rr)
  mu_ll = Minf/Reinf * (1+C/Tinf)/(T_ll+C/Tinf) * T_ll^(1.5)
  mu_v_ll = 4/3 * mu_ll     # viscosity of stress
  mu_h_ll = gamma*mu_ll/Pr  # viscosity of heat flux
  mu_rr = Minf/Reinf * (1+C/Tinf)/(T_rr+C/Tinf) * T_rr^(1.5)
  mu_v_rr = 4/3 * mu_rr     # viscosity of stress
  mu_h_rr = gamma*mu_rr/Pr  # viscosity of heat flux

  # λ_max_ll = (abs(v1_ll)+abs(v2_ll))/2 + sqrt(T_ll) + abs(mu_h_ll / (u1_ll * L))
  # λ_max_rr = (abs(v1_rr)+abs(v2_rr))/2 + sqrt(T_rr) + abs(mu_h_rr / (u1_rr * L))
  λ_max_ll = sqrt(v1_ll^2+v2_ll^2) + sqrt(T_ll) + abs(mu_h_ll / (u1_ll * L))
  λ_max_rr = sqrt(v1_rr^2+v2_rr^2) + sqrt(T_rr) + abs(mu_h_rr / (u1_rr * L))
  return max(λ_max_ll, λ_max_rr)
end


@inline have_constant_speed(::HyperbolicNavierStokesEquations2D) = Val(false)

@inline function max_abs_speeds(u, eq::HyperbolicNavierStokesEquations2D)
  @unpack gamma, Pr, L, Tinf, C, Minf, Reinf = eq
  rho, rho_v1, rho_v2, rho_e, tau_xx, tau_xy, tau_yy, q_x, q_y = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  p = (gamma-1)*(rho_e - 0.5 * rho * (v1^2 + v2^2))
  T = abs(gamma*p/rho)
  mu = Minf/Reinf * (1+C/Tinf)/(T+C/Tinf) * T^(1.5)
  mu_v = 4/3 * mu     # viscosity of stress
  mu_h = gamma*mu/Pr  # viscosity of heat flux

  # λ_max = (abs(v1)+abs(v2))/2 + sqrt(T) + abs(mu_h / (rho * L))
  λ_max1 = abs(v1) + sqrt(T) + abs(mu_h / (rho * L))
  λ_max2 = abs(v2) + sqrt(T) + abs(mu_h / (rho * L))
  return λ_max1, λ_max2
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::HyperbolicNavierStokesEquations2D)
  @unpack gamma, Pr, L, Tinf, C, Minf, Reinf = equations
  rho, rho_v1, rho_v2, rho_e, tau_xx, tau_xy, tau_yy, q_x, q_y = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  E = rho_e / rho
  p = (gamma - 1) * (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2))
  T = gamma * p / rho

  return SVector(rho, v1, v2, E, p, T, tau_xx, tau_xy, tau_yy)
end

# Convert conservative variables to entropy found in I Do Like CFD, Too, Vol. 1
@inline function cons2entropy(u, equations::HyperbolicNavierStokesEquations2D)
  @unpack gamma, Pr, L, Tinf, C, Minf, Reinf = equations
  u1, u2, u3, u4, u5 = u

  p = (gamma-1)*(u4-0.5*(u2^2+u3^2)/u1)  # pressure

  s = (log(p) - gamma*log(u1))*u1 / (gamma-1)

  w1 = (gamma - s) / (gamma-1) - 0.5 * u1/p * (u2^2 + u3^2)/u1^2
  w2 = u2/p
  w3 = u3/p
  w4 = -u1/p

  return SVector(w1, w2, w3, w4, zero(w1), zero(w1), zero(w1), zero(w1), zero(w1))
end


# Calculate entropy for a conservative state `u` (here: same as total energy)
@inline entropy(u, equations::HyperbolicNavierStokesEquations2D) = energy_total(u, equations)


# Calculate total energy for a conservative state `u`
@inline function energy_total(u, equations::HyperbolicNavierStokesEquations2D)
  return u[4]/u[1]
end


end # @muladd
