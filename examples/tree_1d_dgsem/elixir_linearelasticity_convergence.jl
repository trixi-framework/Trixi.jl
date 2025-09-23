using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear elasticity equations

# "Material parameters" need to be integers such that the manufactured solutions stays periodic on [0,1]
rho = 3
c1 = 2
c1_squared = c1^2
equations = LinearElasticityEquations1D(rho, c1_squared, c1)

solver = DGSEM(polydeg = 3, surface_flux = flux_hll)

coordinate_min = 0.0
coordinate_max = 1.0

mesh = TreeMesh(coordinate_min, coordinate_max,
                initial_refinement_level = 4,
                n_cells_max = 10_000)

initial_condition = initial_condition_convergence_test

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0) # Relatively short simulation time due to high wave speeds (~5990 m/s)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 42.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
