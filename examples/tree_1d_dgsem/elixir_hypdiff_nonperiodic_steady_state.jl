using NonlinearSolveFirstOrder
using ADTypes
using OrdinaryDiffEqLowStorageRK
using Trixi

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
                periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_conditions,
                                    source_terms = source_terms_poisson_nonperiodic)

###############################################################################
# ODE/Steady State problem

tspan = (0.0, 5.0) # Does not matter for steady-state solve
ode = semidiscretize(semi, tspan)

# Construct steady steate problem
steady_state_prob = SteadyStateProblem(ode)

alg = NewtonRaphson(autodiff = AutoFiniteDiff())

resid_tol = 1e-12
sol_steady_state = NonlinearSolveFirstOrder.solve(steady_state_prob, alg; abstol = resid_tol)

# Supply steady state solution as initial condition for time-dependent run
@assert SciMLBase.successful_retcode(sol_steady_state)
ode.u0 .= sol_steady_state.u

###############################################################################
# Callbacks

summary_callback = SummaryCallback()

steady_state_callback = SteadyStateCallback(abstol = resid_tol, reltol = 0.0)

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback,
                        steady_state_callback,
                        analysis_callback,
                        stepsize_callback)

###############################################################################
# run the simulation - IC already satisfies the `SteadyStateCallback`

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false); dt = 1,
            ode_default_options()..., callback = callbacks);
