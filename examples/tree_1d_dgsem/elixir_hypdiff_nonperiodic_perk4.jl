
using OrdinaryDiffEq
using Trixi

# Convex and ECOS are imported because they are used for finding the optimal time step and optimal 
# monomial coefficients in the stability polynomial of P-ERK time integrators.
using Convex, ECOS

###############################################################################
# semidiscretization of the hyperbolic diffusion equations

equations = HyperbolicDiffusionEquations1D()

initial_condition = initial_condition_poisson_nonperiodic

boundary_conditions = boundary_condition_poisson_nonperiodic

solver = DGSEM(polydeg = 4, surface_flux = flux_lax_friedrichs)

coordinates_min = 0.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 30_000,
                periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions,
                                    source_terms = source_terms_poisson_nonperiodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

resid_tol = 5.0e-12
steady_state_callback = SteadyStateCallback(abstol = resid_tol, reltol = 0.0)

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

# Construct fourth order paired explicit Runge-Kutta method with 11 stages for given simulation setup.
# Pass `tspan` to calculate maximum time step allowed for the bisection algorithm used 
# in calculating the polynomial coefficients in the ODE algorithm.                                     
ode_algorithm = Trixi.PairedExplicitRK4(11, tspan, semi)

cfl_number = Trixi.calculate_cfl(ode_algorithm, ode)
stepsize_callback = StepsizeCallback(cfl = 0.9 * cfl_number)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = Trixi.solve(ode, ode_algorithm,
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
