# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


@doc raw"""
    HyperbolicNavierStokesEquations1D

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
struct HyperbolicNavierStokesEquations1D{RealT<:Real} <: AbstractHyperbolicNavierStokesEquations{1, 5}
  gamma::RealT  # ratio of specific heats
  Pr::RealT     # Prandtl number
  L::RealT      # length scale
  Tinf::RealT   # free stream temperature
  C::RealT      # Sutherland constant
  Minf::RealT   # Mach number
  Reinf::RealT  # Reynolds number
end

function HyperbolicNavierStokesEquations1D(;gamma=1.4, Pr=0.75, L=inv(sqrt(2pi)), Tinf = 400.0, C = 110.5, Minf = 3.5, Reinf = 25.0)
  HyperbolicNavierStokesEquations1D(promote(gamma, Pr, L, Tinf, C, Minf, Reinf)...)
end


varnames(::typeof(cons2cons), ::HyperbolicNavierStokesEquations1D) = ("rho", "rho_v1", "rho_E", "tau", "q")
varnames(::typeof(cons2prim), ::HyperbolicNavierStokesEquations1D) = ("rho", "v1", "E", "p", "T")
default_analysis_errors(::HyperbolicNavierStokesEquations1D) = (:l2_error, :linf_error, :residual)

@inline function residual_steady_state(du, ::HyperbolicNavierStokesEquations1D)
  abs(du[1])
end

@inline function initial_condition_constructed_periodic(x, t, equations::HyperbolicNavierStokesEquations1D)
  @unpack gamma, Pr, L, Tinf, C, Minf, Reinf = equations

  sinx, cosx = sincos(x[1])

  rho = (sinx + 2.0)^2
  v1 = sinx + 2.0
  E = rho
  p = (gamma - 1) * rho^2 / 2
  T = gamma * p / rho
  mu = Minf/Reinf * (1+C/Tinf)/(T+C/Tinf) * T^(1.5)
  tau = 4/3 * mu * cosx
  q = -mu*gamma/Pr * v1*cosx

  return SVector(rho, rho*v1, rho*E, tau, q)
end

@inline function source_terms_constructed_periodic(u, x, t, equations::HyperbolicNavierStokesEquations1D)
  u1, u2, u3, u4, u5 = u
  @unpack gamma, Pr, L, Tinf, C, Minf, Reinf = equations

  sinx, cosx = sincos(x[1])

  rho = (sinx + 2.0)^2
  v1 = sinx + 2.0
  E = rho
  p = (gamma - 1) * rho^2/2
  pd = 2 * (gamma - 1) * v1^3 *cosx
  T = gamma * p / rho
  Td = gamma * (gamma - 1) * v1 * cosx
  mu = Minf/Reinf * (1+C/Tinf)/(T+C/Tinf) * T^(1.5)
  mud = mu * Td * (1.5/T - 1/(T+C/Tinf))
  tau = 4/3 * mu * cosx
  taud = 4/3 * (mud*cosx - mu*sinx)
  q = -mu*gamma/Pr * v1*cosx
  qd = -gamma/Pr * (mud * v1*cosx + mu * (cosx^2-v1*sinx))

  mu_v = 4/3 * mu     # viscosity of stress
  mu_h = gamma*mu/Pr  # viscosity of heat flux

  dq1 = 3 * rho * cosx
  dq2 = 4 * v1^3 * cosx + pd - taud
  dq3 = 2.5 * (gamma + 1) * v1^4 * cosx - taud*v1 - tau*cosx + qd
  dq4 = -tau / mu_v
  dq5 = -q / mu_h

  return SVector(dq1, dq2, dq3, dq4, dq5)
end

"""
    initial_condition_constructed_exp(x, t, equations::HyperbolicNavierStokesEquations1D)

A non-priodic smooth initial condition. In the initial guess the primal variables
density ρ, velocity v and specific total energy E follow linear functions.
The function can be used in combination with [`source_terms_constructed_exp`](@ref) and BoundaryConditionDirichlet.
The primal variables follow exponential functions.
"""
@inline function initial_condition_constructed_exp(x, t, equations::HyperbolicNavierStokesEquations1D)
  @unpack gamma, Pr, L, Tinf, C, Minf, Reinf = equations

  if iszero(t)
    rho = (exp(1.0) - 1.0)*x[1] + 1.0
    v1 = (exp(0.5) - 1.0)*x[1] + 1.0
    E = rho
    T = gamma * (gamma-1) * 0.5 * exp(x[1])
    mu = Minf/Reinf * (1+C/Tinf)/(T+C/Tinf) * T^(1.5)
    tau = 2/3 * mu * v1
    q = -mu/((gamma-1)*Pr) * T
  else

    rho = exp(x[1])
    v1 = exp(0.5*x[1])
    E = rho
    p = (gamma - 1) * 0.5 * exp(2*x[1])
    T = gamma * (gamma-1) * rho/2
    mu = Minf/Reinf * (1+C/Tinf)/(T+C/Tinf) * T^(1.5)
    tau = 2/3 * mu * v1
    q = -mu/((gamma-1)*Pr) * T

  end

  return SVector(rho, rho*v1, rho*E, tau, q)
end

"""
    source_terms_constructed_exp(u, x, t, equations::HyperbolicNavierStokesEquations1D)

Source terms that include the forcing function `f(x)` and right hand side for the hyperbolic
Navier-Stokes system that is used with [`initial_condition_constructed_exp`](@ref) and
BoundaryConditionDirichlet.
"""
@inline function source_terms_constructed_exp(u, x, t, equations::HyperbolicNavierStokesEquations1D)
  u1, u2, u3, u4, u5 = u
  @unpack gamma, Pr, L, Tinf, C, Minf, Reinf = equations

  rho = exp(x[1])
  v1 = exp(0.5*x[1])
  E = rho
  p = (gamma - 1) * 0.5 * exp(2*x[1])
  T = gamma * (gamma-1) * rho/2
  mu = Minf/Reinf * (1+C/Tinf)/(T+C/Tinf) * T^(1.5)
  mud = mu * (1.5 - T/(T+C/Tinf))
  tau = 2/3 * mu * v1
  taud = 1/3 * v1 * (mu + 2 * mud)
  q = -mu/((gamma-1)*Pr) * T
  qd = -1/((gamma-1)*Pr) * T * (mu + mud)

  mu_v = 4/3 * mu     # viscosity of stress
  mu_h = gamma*mu/Pr  # viscosity of heat flux

  dq1 = 3/2 * exp(3/2 * x[1])
  dq2 = exp(2*x[1])*(gamma+1) - taud
  dq3 = 5/4*(gamma+1)*exp(5/2*x[1]) - v1 * (taud + 0.5*tau) + qd
  dq4 = -tau / mu_v
  dq5 = -q / mu_h

  return SVector(dq1, dq2, dq3, dq4, dq5)
end

"""
    initial_condition_constructed_exp2(x, t, equations::HyperbolicNavierStokesEquations1D)

A non-priodic smooth initial condition.
Can be used in combination with [`source_terms_constructed_exp`](@ref) and BoundaryConditionDirichlet.
The primal variables follow exponential functions.
"""
@inline function initial_condition_constructed_exp2(x, t, equations::HyperbolicNavierStokesEquations1D)
  @unpack gamma, Pr, L, Tinf, C, Minf, Reinf = equations

  rho = exp(x[1])
  v1 = exp(-0.5*x[1])
  E = 1/rho
  p = (gamma - 1) * 0.5
  T = gamma * (gamma-1) * E/2
  mu = Minf/Reinf * (1+C/Tinf)/(T+C/Tinf) * T^(1.5)
  tau = -2/3 * mu * v1
  q = mu*gamma/Pr * 0.5 * E

  if iszero(t)
    delta = 0.01
    rho *= (1+delta)
    v1 *=(1-delta)
    E *= (1+delta)
    tau *= (1+delta/2)
    q *= (1-delta/2)
  end

  return SVector(rho, rho*v1, rho*E, tau, q)
end

"""
    source_terms_constructed_exp2(u, x, t, equations::HyperbolicNavierStokesEquations1D)

Source terms that include the forcing function `f(x)` and right hand side for the hyperbolic
Navier-Stokes system that is used with [`initial_condition_constructed_exp`](@ref) and
BoundaryConditionDirichlet.
"""
@inline function source_terms_constructed_exp2(u, x, t, equations::HyperbolicNavierStokesEquations1D)
  u1, u2, u3, u4, u5 = u
  @unpack gamma, Pr, L, Tinf, C, Minf, Reinf = equations

  rho = exp(x[1])
  v1 = exp(-0.5*x[1])
  E = 1/rho
  p = (gamma - 1) * 0.5
  T = gamma * (gamma-1) * E/2
  Td = -T
  mu = Minf/Reinf * (1+C/Tinf)/(T+C/Tinf) * T^(1.5)
  mud = mu * Td * (1.5/T - 1/(T+C/Tinf))
  tau = -2/3 * mu * v1
  taud = 1/3 * v1 * (mu - 2 * mud)
  q = mu*gamma/Pr * 0.5 * E
  qd = gamma/Pr * 0.5 * E * (mud - mu)

  mu_v = 4/3 * mu     # viscosity of stress
  mu_h = gamma*mu/Pr  # viscosity of heat flux

  dq1 = exp(1/2 * x[1])/2
  dq2 = -taud
  dq3 = -0.25*(gamma+1)*v1 - v1 * (taud - 0.5*tau) + qd
  dq4 = -u4 / mu_v
  dq5 = -u5 / mu_h

  return SVector(dq1, dq2, dq3, dq4, dq5)
end

"""
  boundary_condition_viscous_shock(u_inner, orientation, direction, x, t,
                                                surface_flux_function,
                                                equations::HyperbolicNavierStokesEquations1D)

Boundary conditions used for the Navier-Stokes viscous shock example.
The exact solution at the boundarys is calculated as explained in
  Masatsuka (2013)
  I Do Like CFD, Too: Vol 1.
  Freely available at [http://www.cfdbooks.com/](http://www.cfdbooks.com/)
"""
function boundary_condition_viscous_shock(u_inner, orientation, direction, x, t,
                                                surface_flux_function,
                                                equations::HyperbolicNavierStokesEquations1D)
  @unpack gamma = equations
  M_left = 3.5 # Mach number for the left boundary

  if direction == 2 # u_inner is "left" of boundary, u_boundary is "right" of boundary

    p = 1/gamma + 2/(gamma+1)*(M_left^2-1) # pressure in the right boundary

    u1 = (gamma+1)*M_left^2 / ((gamma-1)*M_left^2+2)
    u2 = M_left
    u3 = p/(gamma-1) + 0.5*u2^2/u1
    u4 = 0.0
    u5 = 0.0
    u_boundary = SVector(u1, u2, u3, u4, u5)

    # Calculate boundary flux
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary

    u1 = 1.0
    u2 = M_left
    u3 = 1.0/(gamma*(gamma-1)) + 0.5*u2^2/u1
    u4 = 0.0
    u5 = 0.0
    u_boundary = SVector(u1, u2, u3, u4, u5)

    # Calculate boundary flux
    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  end

  return flux
end

@inline function source_terms_harmonic(u, x, t, equations::HyperbolicNavierStokesEquations1D)
  u1, u2, u3, u4, u5 = u
  @unpack gamma, Pr, L, Tinf, C, Minf, Reinf = equations
  p = (gamma-1)*(u3-0.5*u2^2/u1)
  T = gamma*p/u1
  mu = Minf/Reinf * (1+C/Tinf)/(T+C/Tinf) * T^(1.5)
  mu_v = 4/3 * mu     # viscosity of stress
  mu_h = gamma*mu/Pr  # viscosity of heat flux

  dq4 = -u4 / mu_v
  dq5 = -u5 / mu_h
  return SVector(zero(dq4), zero(dq4), zero(dq4), dq4, dq5)
end

# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer, equations::HyperbolicNavierStokesEquations1D)
  rho, rho_v1, rho_e, tau, q = u
  @unpack gamma, Pr, L, Tinf, C, Minf, Reinf = equations
  p = (gamma-1)*(rho_e-0.5*rho_v1^2/rho)
  T = gamma*p/rho

  # Ignore orientation since it is always "1" in 1D
  f1 = rho_v1
  f2 = rho_v1^2/rho + p - tau
  f3 = rho_v1*(rho_e/rho+p/rho) - tau*rho_v1/rho + q
  f4 = -rho_v1/rho
  f5 = T/(gamma*(gamma-1))

  return SVector(f1, f2, f3, f4, f5)
end


@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::HyperbolicNavierStokesEquations1D)
  @unpack gamma, Pr, L, Tinf, C, Minf, Reinf = equations
  u1_ll, u2_ll, u3_ll, u4_ll, u5_ll = u_ll
  u1_rr, u2_rr, u3_rr, u4_rr, u5_rr = u_rr
  p_ll = (gamma-1)*(u3_ll-0.5*u2_ll^2/u1_ll)
  p_rr = (gamma-1)*(u3_rr-0.5*u2_rr^2/u1_rr)
  T_ll = gamma*p_ll/u1_ll
  T_rr = gamma*p_rr/u1_rr
  mu_ll = Minf/Reinf * (1+C/Tinf)/(T_ll+C/Tinf) * T_ll^(1.5)
  mu_v_ll = 4/3 * mu_ll     # viscosity of stress
  mu_h_ll = gamma*mu_ll/Pr  # viscosity of heat flux
  mu_rr = Minf/Reinf * (1+C/Tinf)/(T_rr+C/Tinf) * T_rr^(1.5)
  mu_v_rr = 4/3 * mu_rr     # viscosity of stress
  mu_h_rr = gamma*mu_rr/Pr  # viscosity of heat flux

  # λ_max = max(abs(u2_ll/u1_ll), abs(u2_rr/u1_rr)) + sqrt(max(T_ll, T_rr)) + max(abs(L*u1_ll/min(mu_v_ll, mu_h_ll)), abs(L*u1_rr/min(mu_v_rr, mu_h_rr)))
  λ_max = max(abs(u2_ll/u1_ll), abs(u2_rr/u1_rr)) + sqrt(max(T_ll, T_rr)) + max(abs(mu_h_ll / u1_ll), abs(mu_h_rr / u1_ll)) / L
  return λ_max
end


@inline have_constant_speed(::HyperbolicNavierStokesEquations1D) = Val(false)

@inline function max_abs_speeds(u, eq::HyperbolicNavierStokesEquations1D)
  @unpack gamma, Pr, L, Tinf, C, Minf, Reinf = eq
  rho, rho_v1, rho_e, tau, q = u
  p = (gamma-1)*(rho_e-0.5*rho_v1^2/rho)
  T = gamma*p/rho
  mu = Minf/Reinf * (1+C/Tinf)/(T+C/Tinf) * T^(1.5)
  mu_v = 4/3 * mu     # viscosity of stress
  mu_h = gamma*mu/Pr  # viscosity of heat flux

  # λ_max = abs(rho_v1/rho) + sqrt(T) + abs(L*rho/min(mu_v, mu_h))
  λ_max = abs(rho_v1/rho) + sqrt(T) + abs(mu_h / (rho * L))
  return λ_max
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::HyperbolicNavierStokesEquations1D)
  @unpack gamma, Pr, L, Tinf, C, Minf, Reinf = equations
  rho, rho_v1, rho_e, tau, q = u

  v1 = rho_v1/rho
  E = rho_e/rho
  p = (gamma-1)*(rho_e-0.5*rho*v1^2)
  T = gamma*p/rho

  return SVector(rho, v1, E, p, T)
end

# Convert conservative variables to entropy found in I Do Like CFD, Too, Vol. 1
@inline function cons2entropy(u, equations::HyperbolicNavierStokesEquations1D)
  @unpack gamma, Pr, L, Tinf, C, Minf, Reinf = equations
  u1, u2, u3, u4, u5 = u

  p = (gamma-1)*(u3-0.5*u2^2/u1)  # pressure

  s = (log(p) - gamma*log(u1))*u1 / (gamma-1)

  w1 = (gamma - s) / (gamma-1) - 0.5 * u1/p * u2^2/u1^2
  w2 = u2/p
  w3 = -u1/p

  return SVector(w1, w2, w3, zero(w1), zero(w1))
end


# Calculate entropy for a conservative state `u` (here: same as total energy)
@inline entropy(u, equations::HyperbolicNavierStokesEquations1D) = energy_total(u, equations)


# Calculate total energy for a conservative state `u`
@inline function energy_total(u, equations::HyperbolicNavierStokesEquations1D)
  return u[3]/u[1]
end

"""
    calc_viscous_shock_solution

    The fuction calls a fortran routine which calculates the solution of the
    viscous shock example. The grid on which the solution is calculated is the same
    as the grid used for Trixis DGSEM solver. The grid is defined by the input
    variables.
"""

const lib_path = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "NavierStokes1d_fortran", "ns_shock_lobatto.so")

@inline function calc_viscous_shock_solution(polydeg, refinement_level)

    n_nodes = (polydeg+1)*2^refinement_level
    epsilon = 0.0000000001

    solution = Vector{Float64}(zeros(6*n_nodes))

    ccall((:calc_solution, lib_path),
        Cvoid,
        (Ref{Int64}, Ref{Int64}, Ref{Float64}, Ptr{Float64}),
        polydeg, refinement_level, epsilon, solution)

    x = solution[1:n_nodes]
    rho = solution[n_nodes+1:2*n_nodes]
    rho_v1 = solution[2*n_nodes+1:3*n_nodes]
    rho_e = solution[3*n_nodes+1:4*n_nodes]
    tau = solution[4*n_nodes+1:5*n_nodes]
    q = solution[5*n_nodes+1:6*n_nodes]

    return x, rho, rho_v1, rho_e, tau, q
end


end # @muladd
