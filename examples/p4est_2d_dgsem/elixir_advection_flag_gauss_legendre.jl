using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

solver = DGSEM(polydeg = 4, surface_flux = flux_lax_friedrichs,
               basis_type = GaussLegendreBasis)

# Deformed rectangle that looks like a waving flag,
# lower and upper faces are sinus curves, left and right are vertical lines.
f1(s) = SVector(-1.0, s - 1.0)
f2(s) = SVector(1.0, s + 1.0)
f3(s) = SVector(s, -1.0 + sinpi(0.5 * s))
f4(s) = SVector(s, 1.0 + sinpi(0.5 * s))

# Create P4estMesh with 3 x 2 trees and 6 x 4 elements,
# approximate the geometry with a smaller polydeg for testing.
trees_per_dimension = (3, 2)
mesh = P4estMesh(trees_per_dimension, polydeg = 3,
                 faces = (f1, f2, f3, f4),
                 periodicity = true,
                 initial_refinement_level = 1)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

ode = semidiscretize(semi, (0.0, 0.2))

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval = 100)
stepsize_callback = StepsizeCallback(cfl = 0.8)

callbacks = CallbackSet(summary_callback, analysis_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
