# TODO: Taal refactor, rename to
# - euler_source_terms.jl
# or something similar?

using OrdinaryDiffEq
using Trixi

equations = CompressibleEulerEquations2D(1.4)

initial_conditions = initial_conditions_convergence_test

surface_flux = flux_lax_friedrichs
solver = DGSEM(3, surface_flux)

coordinates_min = (0, 0)
coordinates_max = (2, 2)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=10_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_conditions, solver,
                                    source_terms=source_terms_convergence_test)

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

# TODO: Taal implement, printing stuff (Logo etc.) at the beginning (optionally)

analysis_interval = 100
alive_callback = AliveCallback(analysis_interval=analysis_interval)
analysis_callback = AnalysisCallback(semi, analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=1.0)

save_solution = SaveSolutionCallback(solution_interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=:primitive)

# TODO: Taal, restart
# restart_interval = 100

callbacks = CallbackSet(stepsize_callback, analysis_callback, save_solution, alive_callback)


sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), dt=stepsize_callback(ode),
            save_everystep=false, callback=callbacks);

nothing
