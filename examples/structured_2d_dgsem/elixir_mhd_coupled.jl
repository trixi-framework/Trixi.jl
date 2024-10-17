using OrdinaryDiffEq
using Trixi

###############################################################################
# Two semidiscretizations of the ideal GLM-MHD systems using converter functions such that
# they are coupled across the domain boundaries to generate a periodic system.
#
# In this elixir, we have a square domain that is divided into a left and right half.
# On each half of the domain, an independent SemidiscretizationHyperbolic is created for
# each set of ideal GLM-MHD equations. The two systems are coupled in the x-direction
# and are periodic in the y-direction.
# For a high-level overview, see also the figure below:
#
# (-2,  2)                                   ( 2,  2)
#     ┌────────────────────┬────────────────────┐
#     │    ↑ periodic ↑    │    ↑ periodic ↑    │
#     │                    │                    │
#     │     =========      │     =========      │
#     │     system #1      │     system #2      │
#     │     =========      │     =========      │
#     │                    │                    │
#     │<-- coupled         │<-- coupled         │
#     │         coupled -->│         coupled -->│
#     │                    │                    │
#     │    ↓ periodic ↓    │    ↓ periodic ↓    │
#     └────────────────────┴────────────────────┘
# (-2, -2)                                   ( 2, -2)

gamma = 5 / 3
equations = IdealGlmMhdEquations2D(gamma)

cells_per_dimension = (32, 64)

volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
solver = DGSEM(polydeg = 3,
               surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

###########
# system #1
###########

initial_condition1 = initial_condition_convergence_test
coordinates_min1 = (-1 / sin(pi / 4), -1 / sin(pi / 4))
coordinates_max1 = (0.0, 1 / sin(pi / 4))
mesh1 = StructuredMesh(cells_per_dimension,
                       coordinates_min1,
                       coordinates_max1,
                       periodicity = (false, true))

coupling_function1 = (x, u, equations_other, equations_own) -> u
boundary_conditions1 = (x_neg = BoundaryConditionCoupled(2, (:end, :i_forward), Float64,
                                                         coupling_function1),
                        x_pos = BoundaryConditionCoupled(2, (:begin, :i_forward), Float64,
                                                         coupling_function1),
                        y_neg = boundary_condition_periodic,
                        y_pos = boundary_condition_periodic)

semi1 = SemidiscretizationHyperbolic(mesh1, equations, initial_condition1, solver,
                                     boundary_conditions = boundary_conditions1)

###########
# system #2
###########

initial_condition2 = initial_condition_convergence_test
coordinates_min2 = (0.0, -1 / sin(pi / 4))
coordinates_max2 = (1 / sin(pi / 4), 1 / sin(pi / 4))
mesh2 = StructuredMesh(cells_per_dimension,
                       coordinates_min2,
                       coordinates_max2,
                       periodicity = (false, true))

coupling_function2 = (x, u, equations_other, equations_own) -> u
boundary_conditions2 = (x_neg = BoundaryConditionCoupled(1, (:end, :i_forward), Float64,
                                                         coupling_function2),
                        x_pos = BoundaryConditionCoupled(1, (:begin, :i_forward), Float64,
                                                         coupling_function2),
                        y_neg = boundary_condition_periodic,
                        y_pos = boundary_condition_periodic)

semi2 = SemidiscretizationHyperbolic(mesh2, equations, initial_condition2, solver,
                                     boundary_conditions = boundary_conditions2)

# Create a semidiscretization that bundles all the semidiscretizations.
semi = SemidiscretizationCoupled(semi1, semi2)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100

analysis_callback1 = AnalysisCallback(semi1, interval = 100)
analysis_callback2 = AnalysisCallback(semi2, interval = 100)
analysis_callback = AnalysisCallbackCoupled(semi, analysis_callback1, analysis_callback2)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 50,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

cfl = 1.0

stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl,
                                      semi_indices = [1, 2])

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 0.01, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
