
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

initial_condition = initial_condition_convergence_test

boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict(:Bottom => boundary_condition,
                           :Top => boundary_condition,
                           :Circle => boundary_condition,
                           :Cut => boundary_condition)

solver = DGSEM(polydeg = 4, surface_flux = flux_lax_friedrichs)

# Unstructured 3D half circle mesh from HOHQMesh
mesh_file = Trixi.download("https://gist.githubusercontent.com/andrewwinters5000/11461efbfb02c42e06aca338b3d0b645/raw/81deeb1ebc4945952c30af5bb75fe222a18d975c/abaqus_half_circle_3d.inp",
                           joinpath(@__DIR__, "abaqus_half_circle_3d.inp"))

mesh = P4estMesh{3}(mesh_file, initial_refinement_level = 0)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_convergence_test,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 50,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback);

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

summary_callback() # print the timer summary
