
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
function initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)
    # needs appropriate mesh size, e.g. [-10,-10]x[10,10]
    # make sure that the inicenter does not exit the domain, e.g. T=10.0
    # initial center of the vortex
    inicenter = SVector(0.0, 0.0)
    # size and strength of the vortex
    iniamplitude = 0.2
    # base flow
    rho = 1.0
    v1 = 1.0
    v2 = 1.0
    vel = SVector(v1, v2)
    p = 10.0
    rt = p / rho                  # ideal gas equation
    cent = inicenter + vel * t      # advection of center
    cent = x - cent               # distance to centerpoint
    #cent=cross(iniaxis,cent)     # distance to axis, tangent vector, length r
    # cross product with iniaxis = [0,0,1]
    cent = SVector(-cent[2], cent[1])
    r2 = cent[1]^2 + cent[2]^2
    du = iniamplitude / (2 * π) * exp(0.5 * (1 - r2)) # vel. perturbation
    dtemp = -(equations.gamma - 1) / (2 * equations.gamma * rt) * du^2            # isentrop
    rho = rho * (1 + dtemp)^(1 \ (equations.gamma - 1))
    vel = vel + du * cent
    v1, v2 = vel
    p = p * (1 + dtemp)^(equations.gamma / (equations.gamma - 1))
    prim = SVector(rho, v1, v2, p)
    return prim2cons(prim, equations)
end
initial_condition = initial_condition_isentropic_vortex
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-10.0, -10.0)
coordinates_max = (10.0, 10.0)
mesh = P4estMesh((1, 1), polydeg = Trixi.polydeg(solver),
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 initial_refinement_level = 4)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true,
                                     extra_analysis_errors = (:conservation_error,),
                                     extra_analysis_integrals = (entropy, energy_total,
                                                                 energy_kinetic,
                                                                 energy_internal))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

###############################################################################
# run the simulation

sol = solve(ode, BS3(),
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
