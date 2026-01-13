using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7, 0.5)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

polydeg = 4 # Governs in this case only the number of subcells
basis = LobattoLegendreBasis(polydeg)
surface_flux = flux_godunov
volume_integral = VolumeIntegralPureLGLFiniteVolumeO2(basis,
                                                      volume_flux_fv = surface_flux,
                                                      reconstruction_mode = reconstruction_O2_full,
                                                      slope_limiter = monotonized_central)
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

coordinates_min = (-1.0, -1.0, -1.0)
coordinates_max = (1.0, 1.0, 1.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                n_cells_max = 100_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver)

###############################################################################
# ODE solvers, callbacks etc.

ode = semidiscretize(semi, (0.0, 1.0))

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 100)

stepsize_callback = StepsizeCallback(cfl = 1.2)

callbacks = CallbackSet(summary_callback, analysis_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, ParsaniKetchesonDeconinck3S82(); dt = 1.0,
            ode_default_options()..., callback = callbacks);
