
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test

boundary_condition_convergence_test = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict(:Slant => boundary_condition_convergence_test,
                           :Bezier => boundary_condition_convergence_test,
                           :Right => boundary_condition_convergence_test,
                           :Bottom => boundary_condition_convergence_test,
                           :Top => boundary_condition_convergence_test)

###############################################################################
# Get the DG approximation space

solver = DGSEM(polydeg = 8, surface_flux = flux_lax_friedrichs)

###############################################################################
# Get the curved quad mesh from a file (downloads the file if not available locally)
mesh_file = Trixi.download("https://gist.githubusercontent.com/andrewwinters5000/52056f1487853fab63b7f4ed7f171c80/raw/9d573387dfdbb8bce2a55db7246f4207663ac07f/mesh_trixi_unstructured_mesh_docs.mesh",
                           joinpath(@__DIR__, "mesh_trixi_unstructured_mesh_docs.mesh"))

mesh = UnstructuredMesh2D(mesh_file)

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_restart = SaveRestartCallback(interval = 50,
                                   save_final_restart = true)

save_solution = SaveSolutionCallback(interval = 10,
                                     save_initial_solution = true,
                                     save_final_solution = true)

stepsize_callback = StepsizeCallback(cfl = 0.9)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_restart,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
