using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_weak_blast_wave

basis = LobattoLegendreBasis(5)
surface_flux = flux_lax_friedrichs

indicator = Trixi.IndicatorEntropyViolation(basis; threshold = 1e-6)
volume_integral = Trixi.VolumeIntegralAdaptive(indicator;
                                               volume_integral_default = VolumeIntegralWeakForm(),
                                               volume_integral_stabilized = VolumeIntegralFluxDifferencing(flux_ranocha))

solver = DGSEM(basis, surface_flux, volume_integral)
#solver = DGSEM(basis, surface_flux, VolumeIntegralFluxDifferencing(flux_ranocha))

coordinates_min = (-2.0,)
coordinates_max = (2.0,)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 8,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.8)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        #save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
