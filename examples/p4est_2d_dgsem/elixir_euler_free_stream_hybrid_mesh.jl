
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

# Test free stream preservation with constant initial condition
initial_condition = initial_condition_constant

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

###############################################################################

# Hybrid mesh composed of a second-order and first-order quadrilateral element
mesh_file = Trixi.download("https://gist.githubusercontent.com/DanielDoehring/84d876776e379322c38e9c0bfcadee8e/raw/0068805b853a8faa0fe229280d353016359d8a7d/hybrid_quadmesh.inp",
                           joinpath(@__DIR__, "hybrid_quadmesh.inp"))

# Refine the given mesh twice
mesh = P4estMesh{2}(mesh_file, initial_refinement_level = 2)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions =
                                    Dict(:all => BoundaryConditionDirichlet(initial_condition)))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 2.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
