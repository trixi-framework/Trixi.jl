
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the hyperbolic diffusion equations

equations = HyperbolicDiffusionEquations2D()

initial_condition = initial_condition_poisson_nonperiodic

solver = DGSEM(polydeg = 6, surface_flux = flux_lax_friedrichs)

boundary_conditions = (x_neg = boundary_condition_poisson_nonperiodic,
                       x_pos = boundary_condition_poisson_nonperiodic,
                       y_neg = boundary_condition_periodic,
                       y_pos = boundary_condition_periodic)

###############################################################################
# Get the curved quad mesh from a mapping function
#
# Mapping as described in https://arxiv.org/abs/2012.12040, but reduced to 2D
function mapping(xi_, eta_)
    # Transform input variables between -1 and 1 onto [0,3]
    xi = 1.5 * xi_ + 1.5
    eta = 1.5 * eta_ + 1.5

    y = eta + 3 / 8 * (cos(1.5 * pi * (2 * xi - 3) / 3) *
                       cos(0.5 * pi * (2 * eta - 3) / 3))

    x = xi + 3 / 8 * (cos(0.5 * pi * (2 * xi - 3) / 3) *
                      cos(2 * pi * (2 * y - 3) / 3))

    return SVector(x, y)
end

# Create curved mesh with 8 x 8 elements
cells_per_dimension = (8, 8)
mesh = StructuredMesh(cells_per_dimension, mapping, periodicity = (false, true))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_poisson_nonperiodic,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 30.0)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

resid_tol = 5.0e-12
steady_state_callback = SteadyStateCallback(abstol = resid_tol, reltol = 0.0)

analysis_interval = 4000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy, energy_total))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 4000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 1.9)

callbacks = CallbackSet(summary_callback, steady_state_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = Trixi.solve(ode, Trixi.HypDiffN3Erk3Sstar52(),
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
