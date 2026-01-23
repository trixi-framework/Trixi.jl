using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test

polydeg = 2 # governs in this case only the number of subcells
basis = LobattoLegendreBasis(polydeg)
surface_flux = flux_hll

volume_integral = VolumeIntegralPureLGLFiniteVolumeO2(basis,
                                                      volume_flux_fv = surface_flux,
                                                      reconstruction_mode = reconstruction_O2_full,
                                                      slope_limiter = monotonized_central)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

coordinates_min = (0.0, 0.0, 0.0)
coordinates_max = (2.0, 2.0, 2.0)
cells_per_dimension = (2, 2, 2)
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max,
                      periodicity = false)

boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = boundary_condition_default(mesh, boundary_condition)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)
###############################################################################
# run the simulation

sol = solve(ode, ParsaniKetchesonDeconinck3S82();
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
