
using Downloads: download
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_convergence_test

source_terms = source_terms_convergence_test

# BCs must be passed as Dict
boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict(
  :x_neg => boundary_condition,
  :x_pos => boundary_condition,
  :y_neg => boundary_condition,
  :y_pos => boundary_condition
)

solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

###############################################################################
# Get the uncurved mesh from a file (downloads the file if not available locally)

coordinates_min = (0.0, 0.0)
coordinates_max = (2.0, 2.0)

trees_per_dimension = (4, 4)
mesh = P4estMesh(trees_per_dimension,
                 polydeg=1, initial_refinement_level=2,
                 coordinates_min=coordinates_min, coordinates_max=coordinates_max,
                 periodicity=false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms=source_terms,
                                    boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_restart = SaveRestartCallback(interval=100,
                                   save_final_restart=true)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_restart, save_solution,
                        stepsize_callback)
###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
