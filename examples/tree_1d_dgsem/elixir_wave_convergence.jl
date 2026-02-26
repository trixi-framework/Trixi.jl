using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the wave equations

equations = WaveEquations1D(1.0)

initial_condition = initial_condition_gauss

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0)
coordinates_max = (1.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000, periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_wall)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 4.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.5)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false); dt = 1.0,
            ode_default_options()..., callback = callbacks);
