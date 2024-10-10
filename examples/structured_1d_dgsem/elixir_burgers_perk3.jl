# Convex and ECOS are imported because they are used for finding the optimal time step and optimal 
# monomial coefficients in the stability polynomial of P-ERK time integrators.
using Convex, ECOS

# NLsolve is imported to solve the system of nonlinear equations to find a coefficients
# in the Butcher tableau in the third order P-ERK time integrator.
using NLsolve

using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the (inviscid) Burgers equation

equations = InviscidBurgersEquation1D()

initial_condition = initial_condition_convergence_test

# Create DG solver with polynomial degree = 4 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 4, surface_flux = flux_lax_friedrichs)

coordinates_min = (0.0,) # minimum coordinate
coordinates_max = (1.0,) # maximum coordinate
cells_per_dimension = (64,)

mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 200
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(dt = 0.1,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 3.7)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

# Optimize 8-stage, third order P-ERK scheme for this semidiscretization
ode_algorithm = Trixi.PairedExplicitRK3(8, tspan, semi)

sol = Trixi.solve(ode, ode_algorithm,
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep = false, callback = callbacks);

# Print the timer summary
summary_callback()
