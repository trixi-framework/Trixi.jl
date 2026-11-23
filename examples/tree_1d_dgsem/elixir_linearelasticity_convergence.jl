using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear elasticity equations

# Material parameters with dilatational wave speed c1 = sqrt((lambda + 2 * mu) / rho) = sqrt(2)
rho = 2.0
lambda = 2.0
mu = 1.0
equations = LinearElasticityEquations1D(rho = rho, mu = mu, lambda = lambda)

solver = DGSEM(polydeg = 3, surface_flux = flux_hll)

coordinate_min = 0.0
coordinate_max = 1.0

mesh = TreeMesh(coordinate_min, coordinate_max,
                initial_refinement_level = 4,
                periodicity = true)

initial_condition = initial_condition_convergence_test

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (energy_kinetic,
                                                                 energy_internal,
                                                                 entropy))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode,
            CarpenterKennedy2N54(williamson_condition = false, thread = Trixi.Threaded());
            dt = 42.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
