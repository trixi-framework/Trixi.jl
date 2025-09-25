using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

# Test free stream preservation with constant initial condition
initial_condition = initial_condition_constant

solver = DGSEM(polydeg = 3, surface_flux = flux_hll)

target_mesh_file = joinpath(@__DIR__, "hybrid_hexmesh.inp")
isfile(target_mesh_file) ||
    Trixi.download("https://gist.githubusercontent.com/DanielDoehring/70c1d59e3c6378ee4d7e21769e430fce/raw/25d063663199a20813d2f94ec04135fc2d9d1713/hybrid_hexmesh.inp",
                   target_mesh_file)
mesh_file = target_mesh_file

# Refine the given mesh twice
mesh = P4estMesh{3}(mesh_file, initial_refinement_level = 2)

boundary_conditions = Dict(:all => BoundaryConditionDirichlet(initial_condition))
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 200
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 5.0)

save_solution = SaveSolutionCallback(interval = 500,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback, save_solution)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
