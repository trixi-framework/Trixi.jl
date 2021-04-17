
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

initial_condition = initial_condition_taylor_green_vortex

# TODO: Change back

# TGV run "VolumeIntegralLocalFluxComparison+secret"
# surface_flux = FluxComparedToCentral(flux_ranocha)
# volume_flux = flux_ranocha
# volume_integral = VolumeIntegralLocalFluxComparison(flux_central, volume_flux)

# TGV run "VolumeIntegralLocalFluxComparison"
surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
volume_integral = VolumeIntegralLocalFluxComparison(flux_central, volume_flux)

# TGV run "VolumeIntegralFluxComparison"
# surface_flux = flux_lax_friedrichs
# volume_flux = flux_ranocha
# volume_integral = VolumeIntegralFluxComparison(flux_central, volume_flux)

# TGV run "secret+LLF"
# surface_flux = flux_lax_friedrichs
# volume_flux = FluxComparedToCentral(flux_ranocha)
# volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

# TGV run "secret+secret"
# surface_flux = FluxComparedToCentral(flux_ranocha)
# volume_flux = FluxComparedToCentral(flux_ranocha)
# volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

# TGV run "mod1/VolumeIntegralLocalComparison"
# surface_flux = flux_lax_friedrichs
# volume_flux = flux_ranocha
# volume_integral = VolumeIntegralLocalComparison(VolumeIntegralFluxDifferencing(volume_flux))

solver = DGSEM(polydeg = 3, surface_flux = surface_flux, volume_integral = volume_integral)

coordinates_min = (-pi, -pi, -pi)
coordinates_max = ( pi,  pi,  pi)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=100_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 20
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     save_analysis=true,
                                     extra_analysis_integrals=(Trixi.energy_kinetic,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_restart = SaveRestartCallback(interval=100,
                                   save_final_restart=true)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=1.4)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_restart, save_solution,
                        stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
