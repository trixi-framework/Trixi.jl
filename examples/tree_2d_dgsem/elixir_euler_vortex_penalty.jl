
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

"""
    initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)

The classical isentropic vortex test case of
- Chi-Wang Shu (1997)
  Essentially Non-Oscillatory and Weighted Essentially Non-Oscillatory
  Schemes for Hyperbolic Conservation Laws
  [NASA/CR-97-206253](https://ntrs.nasa.gov/citations/19980007543)
"""
#@inline function initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)
#  # needs appropriate mesh size, e.g. [-10,-10]x[10,10]
#  # make sure that the inicenter does not exit the domain, e.g. T=10.0
#  # initial center of the vortex
#  inicenter = SVector(0.0, 0.0)
#  # size and strength of the vortex
#  iniamplitude = 0.2
#  # base flow
#  rho = 1.0
#  v1 = 1.0
#  v2 = 1.0
#  vel = SVector(v1, v2)
#  p = 10.0
#  rt = p / rho                  # ideal gas equation
#  cent = inicenter + vel*t      # advection of center
#  cent = x - cent               # distance to centerpoint
#  #cent=cross(iniaxis,cent)     # distance to axis, tangent vector, length r
#  # cross product with iniaxis = [0,0,1]
#  cent = SVector(-cent[2], cent[1])
#  r2 = cent[1]^2 + cent[2]^2
#  du = iniamplitude/(2*π)*exp(0.5*(1-r2)) # vel. perturbation
#  dtemp = -(equations.gamma-1)/(2*equations.gamma*rt)*du^2            # isentrop
#  rho = rho * (1+dtemp)^(1\(equations.gamma-1))
#  vel = vel + du*cent
#  v1, v2 = vel
#  p = p * (1+dtemp)^(equations.gamma/(equations.gamma-1))
#  prim = SVector(rho, v1, v2, p)
#  return prim2cons(prim, equations)
#end
function initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)
  ϱ0 = 1.0               # background density
  v0 = SVector(1.0, 1.0) # background velocity
  p0 = 10.0              # background pressure
  ε  = 1.0               # vortex strength
  L = 20.0               # size of the domain per coordinate direction

  T0 = p0 / ϱ0           # background temperature
  γ = equations.gamma    # ideal gas constant

  x0 = v0 * t            # current center of the vortex
  dx = vortex_center.(x - x0, L)
  r2 = sum(abs2, dx)

  # perturbed primitive variables
  T = T0 - (γ - 1) * ε^2 / (8 * γ * π^2) * exp(1 - r2)
  v = v0 + ε / (2 * π) * exp(0.5 * (1 - r2)) * SVector(-dx[2], dx[1])
  ϱ = ϱ0 * (T / T0)^(1 / (γ - 1))
  p = ϱ * T

  return prim2cons(SVector(ϱ, v..., p), equations)
end

vortex_center(x, L) = mod(x + L/2, L) - L/2


@inline function source_terms_vortex_penalty_es(u, x, t, equations::CompressibleEulerEquations2D)
  @unpack gamma=equations 

  inverse_penalty_parameter = 1.0/(1.0e-2)

  du1 = zero(eltype(u))
  du2 = zero(eltype(u))
  du3 = zero(eltype(u))
  du4 = zero(eltype(u))

  if (abs(x[1])<=1.0)
    u_ex = initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)
    prim_ex = cons2prim(u_ex, equations)
    prim    = cons2prim(u, equations)
    beta_ex = 0.5 * prim_ex[1] / prim_ex[4]
    beta    = 0.5 * prim[1] / prim[4]
    vel2_bar = 2 * (avg(prim_ex[2],prim[2])^2 + avg(prim_ex[3],prim[3])^2) - (avg(prim_ex[2]^2,prim[2]^2) + avg(prim_ex[3]^2,prim[3]^2))
    rho_ln = ln_mean(prim_ex[1],prim[1])
    inv_beta_ln=inv_ln_mean(beta_ex,beta) 

    du1 = inverse_penalty_parameter * (u_ex[1] - u[1])
    du2 = inverse_penalty_parameter * (u_ex[2] - u[2])
    du3 = inverse_penalty_parameter * (u_ex[3] - u[3])
    du4 = inverse_penalty_parameter * (  jmp(prim_ex[1],prim[1])  * (0.5/(gamma - 1) * inv_beta_ln + 0.5 * vel2_bar)
                                       - jmp(beta_ex,beta) * (0.5 * rho_ln / (gamma - 1)*inv_beta_ln^2)
                                       + avg(prim_ex[1],prim[1])  * (avg(prim_ex[2],prim[2]) * jmp(prim_ex[2],prim[2]) + avg(prim_ex[3],prim[3]) * jmp(prim_ex[3],prim[3]))) 
  end

  return SVector(du1, du2, du3, du4)
end

avg(a,b) = 0.5 * (a + b)
jmp(a,b) = (a - b)

@inline function source_terms_vortex_penalty(u, x, t, equations::CompressibleEulerEquations2D)
  inverse_penalty_parameter = 1.0/(1.0e-2)
  du1 = zero(eltype(u))
  du2 = zero(eltype(u))
  du3 = zero(eltype(u))
  du4 = zero(eltype(u))
  if (abs(x[1])<=1.0)
    u_ex = initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)
    vel_ex = u_ex[2:3] / u_ex[1]
    vel    = u[2:3] / u[1]

    du1 = inverse_penalty_parameter * (u_ex[1] - u[1])
    du2 = inverse_penalty_parameter * (u_ex[2] - u[2])
    du3 = inverse_penalty_parameter * (u_ex[3] - u[3])
    du4 = inverse_penalty_parameter * (u_ex[4] - u[4])
  end
  return SVector(du1, du2, du3, du4)
end

initial_condition = initial_condition_isentropic_vortex

volume_flux = flux_ranocha
#solver = DGSEM(polydeg=3,surface_flux=flux_lax_friedrichs,volume_integral=VolumeIntegralFluxDifferencing(volume_flux))
solver = DGSEM(polydeg=3,surface_flux=flux_ranocha,volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-10.0, -10.0)
coordinates_max = ( 10.0,  10.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=10_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, source_terms=source_terms_vortex_penalty_es)
#semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, source_terms=source_terms_vortex_penalty)
#semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true,
                                     extra_analysis_errors=(:conservation_error,),
                                     extra_analysis_integrals=(entropy, energy_total,
                                                               energy_kinetic, energy_internal))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.3)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

# run the simulation
# use adaptive time stepping based on error estimates, time step roughly dt = 5e-3
sol = solve(ode, SSPRK43(),
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary

###############################################################################
# run the simulation

#sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
#            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
#            save_everystep=false, callback=callbacks);
#summary_callback() # print the timer summary
