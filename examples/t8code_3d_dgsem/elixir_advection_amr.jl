# The same setup as tree_3d_dgsem/elixir_advection_amr.jl
# to verify the T8codeMesh implementation against TreeMesh.

using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7, 0.5)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

initial_condition = initial_condition_gauss
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-5.0, -5.0, -5.0)
coordinates_max = (5.0, 5.0, 5.0)
trees_per_dimension = (1, 1, 1)

# Note that it is not necessary to use mesh polydeg lower than the solver polydeg
# on a Cartesian mesh.
# See https://doi.org/10.1007/s10915-018-00897-9, Section 6.
mesh = T8codeMesh(trees_per_dimension, polydeg = 1,
                  coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                  initial_refinement_level = 4)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.3)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable = first),
                                      base_level = 4,
                                      med_level = 5, med_threshold = 0.1,
                                      max_level = 6, max_threshold = 0.6)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true,
                           dynamic_load_balancing = false)
# We disable `dynamic_load_balancing` for now, since t8code does not support
# partitioning for coarsening yet. That is, a complete family of elements always
# stays on rank and is not split up due to partitioning. Without this feature
# dynamic AMR simulations are not perfectly deterministic regarding to
# convergent tests. Once this feature is available in t8code load balancing is
# enabled again.

stepsize_callback = StepsizeCallback(cfl = 1.2)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        amr_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
