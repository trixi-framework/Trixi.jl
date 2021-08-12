
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_isentropic_vortex
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = (-10.0, -10.0)
coordinates_max = ( 10.0,  10.0)
mesh = P4estMesh((1, 1), polydeg=Trixi.polydeg(solver),
                 coordinates_min=coordinates_min, coordinates_max=coordinates_max,
                 initial_refinement_level=4)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true,
                                     extra_analysis_errors=(:conservation_error,),
                                     extra_analysis_integrals=(entropy, energy_total,
                                                               energy_kinetic, energy_internal))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)


###############################################################################
# run the simulation

sol = solve(ode, BS3(),
            save_everystep=false, callback=callbacks, maxiters=1e5);
summary_callback() # print the timer summary
